"""Foundry Local backend.

Foundry Local 2.x serves only the OpenAI-compatible /v1 API over HTTP; model management
(catalog, download, load, unload) goes through the `foundry` CLI, which talks to the same
daemon over IPC. We drive the CLI with `-o json`, so anything loaded here is also visible to
any other client of the daemon.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import time

from .base import BackendError, ModelVariant, ServiceStatus

log = logging.getLogger(__name__)

_CATALOG_TTL = 300.0


def _diagnose(log_text: str) -> tuple[str | None, str | None]:
    """Turn a daemon log excerpt into (root-cause line, plain-English hint) for known failures."""
    flat = " ".join(line.strip() for line in log_text.splitlines())
    m = re.search(r"Error creating execution provider:?\s*(.+?)(?:\s+at\s+Microsoft\.|$)", flat)
    if m:
        return (
            f"Error creating execution provider: {m.group(1).strip()}",
            "The hardware runtime (execution provider) for this variant failed to start. This is usually a "
            "version mismatch between Foundry Local's ONNX Runtime and the execution-provider package Windows "
            "installed (e.g. the Intel OpenVINO EP). Check Windows Update and update Foundry Local "
            "(winget upgrade Microsoft.FoundryLocal); meanwhile the CPU or WebGPU variant of the same model should work.",
        )
    m = re.search(r"failed to load model [^:]+:\s*(.+?)(?:\s+at\s+Microsoft\.|$)", flat)
    if m:
        return m.group(1).strip(), None
    return None, None


def _last_json(text: str) -> dict | None:
    """Parse CLI output: usually one JSON document, sometimes preceded by progress lines."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
    return None


class FoundryLocalBackend:
    name = "foundry-local"

    def __init__(self, executable: str | None = None):
        self.executable = executable or shutil.which("foundry")
        self._catalog: list[ModelVariant] | None = None
        self._catalog_at = 0.0
        self._endpoint: str | None = None
        self._system: dict | None = None
        self._lock = asyncio.Lock()  # serialise state-changing CLI calls

    # -- CLI plumbing -----------------------------------------------------------------------

    def _run_sync(self, args: list[str], timeout: float) -> dict:
        if not self.executable:
            raise BackendError("Foundry Local is not installed.", code="NOT_INSTALLED",
                               hint="Install it with: winget install Microsoft.FoundryLocal")
        try:
            proc = subprocess.run(
                [self.executable, "-o", "json", *args],
                capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired as e:
            raise BackendError(f"`foundry {' '.join(args)}` timed out after {timeout:.0f}s", code="TIMEOUT") from e
        # Results go to stdout; errors arrive as {"error": {...}} on stderr.
        for stream in (proc.stdout, proc.stderr):
            data = _last_json(stream)
            if isinstance(data, dict) and "error" in data:
                err = data["error"]
                raise BackendError(err.get("message", "Foundry Local error"), code=err.get("code"), hint=err.get("hint"))
        if proc.returncode != 0:
            raise BackendError(f"`foundry {' '.join(args)}` failed (exit {proc.returncode})",
                               detail=(proc.stderr or proc.stdout)[-2000:])
        data = _last_json(proc.stdout)
        if data is None and proc.stdout.strip():
            raise BackendError(f"Unexpected output from `foundry {' '.join(args)}`", detail=proc.stdout[-2000:])
        return data or {}

    async def _run(self, *args: str, timeout: float = 60.0) -> dict:
        return await asyncio.to_thread(self._run_sync, list(args), timeout)

    async def recent_logs(self, lines: int = 60) -> str:
        if not self.executable:
            return ""

        def run() -> str:
            proc = subprocess.run([self.executable, "server", "logs", "-n", str(lines)], capture_output=True,
                                  text=True, encoding="utf-8", timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
            return proc.stdout

        try:
            return await asyncio.to_thread(run)
        except (OSError, subprocess.SubprocessError):
            return ""

    async def _explain(self, err: BackendError) -> BackendError:
        """Internal errors from the runtime are opaque; dig the real cause out of the daemon log."""
        if err.code == "INTERNAL_ERROR":
            cause, hint = _diagnose(await self.recent_logs())
            if cause:
                err.detail = cause
            if hint:
                err.hint = hint
        return err

    # -- service ----------------------------------------------------------------------------

    async def status(self) -> ServiceStatus:
        if not self.executable:
            return ServiceStatus(installed=False, running=False, endpoint=None)
        data = await self._run("server", "status", timeout=20)
        urls = data.get("webUrls") or []
        self._endpoint = urls[0].rstrip("/") if urls else None
        return ServiceStatus(
            installed=True,
            running=bool(data.get("running")),
            endpoint=self._endpoint,
            version=(self._system or {}).get("service", {}).get("cliVersion"),
            detail={k: data.get(k) for k in ("state", "pid", "startedAt", "uptime")},
        )

    async def system_info(self, refresh: bool = False) -> dict:
        """Full `foundry status` report: hardware Foundry sees, runtime versions, warnings."""
        if self._system is None or refresh:
            self._system = await self._run("status", timeout=60)
        return self._system

    async def start(self) -> ServiceStatus:
        async with self._lock:
            await self._run("server", "start", timeout=120)
        return await self.status()

    async def endpoint(self) -> str:
        if self._endpoint is None:
            st = await self.status()
            if not st.running:
                st = await self.start()
            if not st.endpoint:
                raise BackendError("Foundry Local service has no endpoint.", code="NOT_RUNNING")
        return self._endpoint  # type: ignore[return-value]

    def forget_endpoint(self) -> None:
        self._endpoint = None

    # -- models -----------------------------------------------------------------------------

    async def catalog(self, refresh: bool = False) -> list[ModelVariant]:
        if refresh or self._catalog is None or time.monotonic() - self._catalog_at > _CATALOG_TTL:
            data = await self._run("model", "list", "--variants", timeout=120)
            self._catalog = [
                ModelVariant(
                    id=v["variantId"],
                    alias=v.get("alias", ""),
                    name=v.get("variantName", v["variantId"]),
                    device=(v.get("device") or "").lower(),
                    runtime=v.get("executionProvider"),
                    task=(v.get("type") or "").lower() or None,
                    size_mb=v.get("fileSizeMb"),
                    license=v.get("license"),
                    cached=bool(v.get("cached")),
                    extra={"source": v.get("source")},
                )
                for v in data.get("variants", [])
            ]
            self._catalog_at = time.monotonic()
        loaded = await self.loaded_ids()
        for v in self._catalog:
            v.loaded = v.id in loaded
        return self._catalog

    async def loaded_ids(self) -> set[str]:
        data = await self._run("model", "list", "--loaded", timeout=30)
        return {m["id"] for m in data.get("models", [])}

    def _mark(self, model_id: str, **fields) -> None:
        for v in self._catalog or []:
            if v.id == model_id:
                for k, val in fields.items():
                    setattr(v, k, val)

    async def download(self, model_id: str) -> str:
        async with self._lock:
            data = await self._run("model", "download", model_id, timeout=3600)
        self._mark(model_id, cached=True)
        return data.get("message", f"Downloaded {model_id}")

    async def load(self, model_id: str) -> str:
        await self.endpoint()  # make sure the daemon is up
        async with self._lock:
            try:
                data = await self._run("model", "load", model_id, timeout=1800)  # NPU compile can be slow
            except BackendError as e:
                raise await self._explain(e)
        self._mark(model_id, loaded=True, cached=True)
        return data.get("message", f"Loaded {model_id}")

    async def unload(self, model_id: str) -> str:
        async with self._lock:
            data = await self._run("model", "unload", model_id, timeout=120)
        self._mark(model_id, loaded=False)
        return data.get("message", f"Unloaded {model_id}")

    async def remove(self, model_id: str) -> str:
        if not model_id.strip():
            # `foundry cache rm` with no model clears the entire cache.
            raise BackendError("A model id is required.", code="INVALID_ARGUMENT")
        async with self._lock:
            data = await self._run("cache", "rm", model_id, timeout=300)
        self._mark(model_id, cached=False, loaded=False)
        return data.get("message", f"Removed {model_id}")
