"""OpenVINO GenAI backend: runs Foundry Local's cached OpenVINO models directly on the NPU (or GPU).

Foundry's `*-openvino-npu` / `*-openvino-gpu` variants are OpenVINO IR models wrapped in a tiny ONNX
"EPContext" file that points ONNX Runtime's OpenVINO execution provider at the IR. When that
execution provider can't start (e.g. a version mismatch between Foundry's ONNX Runtime and the
EP package Windows installed), we can still load the IR ourselves with OpenVINO GenAI, which talks
to the NPU driver without ONNX Runtime in the way.

Models run inside the harness process. Compiled blobs are cached, so only the first load of a
model pays the NPU compile cost.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from .base import BackendError

log = logging.getLogger(__name__)

try:
    import openvino as ov
    import openvino_genai as og
except ImportError:  # optional: Intel-only dependency
    ov = og = None


def default_data_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NPU-Harness" / "openvino"


@dataclass
class _Loaded:
    id: str
    device: str
    pipe: object  # og.LLMPipeline
    defaults: dict
    max_prompt_len: int | None
    loaded_at: float
    load_seconds: float
    lock: threading.Lock = field(default_factory=threading.Lock)  # one generation at a time per model


def _flatten_content(content) -> str:
    """OpenAI messages may carry a list of content parts; keep the text ones."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def _sse(obj: dict | str) -> bytes:
    return f"data: {obj if isinstance(obj, str) else json.dumps(obj)}\n\n".encode()


class OpenVinoBackend:
    name = "openvino-genai"

    def __init__(self, data_dir: Path | None = None):
        data_dir = data_dir or default_data_dir()
        self.staging_dir = data_dir / "models"
        self.cache_dir = data_dir / "compiled"
        self._loaded: dict[str, _Loaded] = {}
        self._lock = asyncio.Lock()

    # -- capability -------------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return og is not None

    def info(self) -> dict:
        if not self.available:
            return {"available": False}
        return {
            "available": True,
            "version": og.__version__.split("-")[0],
            "devices": ov.Core().available_devices,
            "cache_dir": str(self.cache_dir),
            "loaded": [
                {"id": m.id, "device": m.device, "load_seconds": round(m.load_seconds, 1)} for m in self._loaded.values()
            ],
        }

    @staticmethod
    def can_run(device: str, runtime: str | None) -> bool:
        return og is not None and runtime == "OpenVINOExecutionProvider" and device in ("npu", "gpu")

    def is_loaded(self, model_id: str | None) -> bool:
        return model_id in self._loaded

    def loaded_ids(self) -> set[str]:
        return set(self._loaded)

    # -- staging ----------------------------------------------------------------------------

    def _stage(self, model_id: str, src: Path) -> Path:
        """Lay out Foundry's files the way LLMPipeline expects them, using hard links (no copy)."""
        xml = next((p for p in sorted(src.glob("openvino_model*.xml"))), None)
        if xml is None or not xml.with_suffix(".bin").exists():
            raise BackendError(f"No OpenVINO model found in {src}", code="NOT_OPENVINO")

        dst = self.staging_dir / model_id.replace(":", "-").replace("/", "_")
        shutil.rmtree(dst, ignore_errors=True)  # removing hard links never touches Foundry's copies
        dst.mkdir(parents=True)

        def link(source: Path, name: str) -> None:
            target = dst / name
            try:
                os.link(source, target)
            except OSError:  # different volume, or links unsupported: fall back to a copy
                shutil.copy2(source, target)

        for f in src.iterdir():
            if f.is_file() and f.suffix in (".json", ".txt"):
                link(f, f.name)
        link(xml, "openvino_model.xml")
        link(xml.with_suffix(".bin"), "openvino_model.bin")
        for part in ("openvino_tokenizer", "openvino_detokenizer"):
            for ext in (".xml", ".bin"):
                for candidate in (src / part / f"{part}{ext}", src / f"{part}{ext}"):
                    if candidate.exists():
                        link(candidate, f"{part}{ext}")
                        break
        if not (dst / "openvino_tokenizer.xml").exists():
            raise BackendError(f"No OpenVINO tokenizer found in {src}", code="NOT_OPENVINO")
        return dst

    @staticmethod
    def _foundry_settings(src: Path, device: str) -> tuple[dict, dict]:
        """Read the compile properties and sampling defaults Foundry ships in genai_config.json."""
        try:
            cfg = json.loads((src / "genai_config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}, {}
        props: dict = {}
        for opt in cfg.get("model", {}).get("decoder", {}).get("session_options", {}).get("provider_options", []):
            load_config = opt.get("OpenVINO", {}).get("load_config")
            if load_config:
                for key, value in json.loads(load_config).get(device.upper(), {}).items():
                    props[key] = int(value) if str(value).isdigit() else value
        return props, cfg.get("search", {})

    # -- lifecycle --------------------------------------------------------------------------

    async def load(self, model_id: str, src: Path, device: str) -> str:
        if not self.available:
            raise BackendError("OpenVINO GenAI is not installed.", code="NOT_INSTALLED",
                               hint="Install it with: uv add openvino-genai")
        async with self._lock:
            if model_id in self._loaded:
                return f"{model_id} is already loaded"
            loaded = await asyncio.to_thread(self._load_sync, model_id, src, device)
            self._loaded[model_id] = loaded
        cached = " (from compile cache)" if loaded.load_seconds < 10 else ""
        return f"Loaded {model_id} on the {device.upper()} with OpenVINO GenAI in {loaded.load_seconds:.1f}s{cached}"

    def _load_sync(self, model_id: str, src: Path, device: str) -> _Loaded:
        path = self._stage(model_id, src)
        props, search = self._foundry_settings(src, device)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            pipe = og.LLMPipeline(str(path), device.upper(), CACHE_DIR=str(self.cache_dir), **props)
        except RuntimeError as e:
            raise BackendError(f"OpenVINO GenAI could not load {model_id} on the {device.upper()}",
                               code="OPENVINO_LOAD_FAILED", detail=str(e).strip().splitlines()[-1][:500]) from e
        return _Loaded(
            id=model_id, device=device, pipe=pipe, defaults=search,
            max_prompt_len=props.get("MAX_PROMPT_LEN"),
            loaded_at=time.time(), load_seconds=time.monotonic() - started,
        )

    async def unload(self, model_id: str) -> str:
        async with self._lock:
            loaded = self._loaded.pop(model_id, None)
            if loaded is None:
                raise BackendError(f"{model_id} is not loaded in OpenVINO GenAI", code="NOT_LOADED")

            def release() -> None:
                with loaded.lock:  # wait for any in-flight generation
                    loaded.pipe = None
                gc.collect()

            await asyncio.to_thread(release)
        return f"Unloaded {model_id} from the {loaded.device.upper()}"

    def discard_staging(self, model_id: str) -> None:
        shutil.rmtree(self.staging_dir / model_id.replace(":", "-").replace("/", "_"), ignore_errors=True)

    # -- inference (OpenAI chat completions) ------------------------------------------------

    def _generation_config(self, m: _Loaded, payload: dict):
        cfg = m.pipe.get_generation_config()
        d = m.defaults
        cfg.do_sample = bool(d.get("do_sample", False))
        for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if key in d:
                setattr(cfg, key, d[key])
        if (t := payload.get("temperature")) is not None:
            cfg.temperature = float(t)
            cfg.do_sample = float(t) > 0
        if (p := payload.get("top_p")) is not None:
            cfg.top_p = float(p)
        cfg.max_new_tokens = int(payload.get("max_completion_tokens") or payload.get("max_tokens") or 512)
        stop = payload.get("stop")
        if stop:
            cfg.stop_strings = {stop} if isinstance(stop, str) else set(stop)
        cfg.apply_chat_template = False  # we template the messages ourselves
        return cfg

    def _prompt(self, m: _Loaded, payload: dict) -> str:
        messages = [{"role": msg.get("role", "user"), "content": _flatten_content(msg.get("content"))}
                    for msg in payload.get("messages", [])]
        if not messages:
            raise BackendError("`messages` is required", code="INVALID_REQUEST")
        tokenizer = m.pipe.get_tokenizer()
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if m.max_prompt_len:
            n = tokenizer.encode(prompt).input_ids.get_shape()[-1]
            if n > m.max_prompt_len:
                raise BackendError(
                    f"Prompt is {n} tokens; this NPU model was compiled for at most {m.max_prompt_len}.",
                    code="CONTEXT_LENGTH_EXCEEDED")
        return prompt

    async def chat(self, payload: dict, usage_sink: dict | None = None) -> dict | AsyncIterator[bytes]:
        """Returns a chat.completion dict, or an async iterator of SSE bytes when `stream` is set.

        Streamed chunks are word-sized and can hold several tokens, so callers wanting exact counts
        pass `usage_sink`, which is filled with the final usage when generation ends."""
        m = self._loaded.get(payload.get("model"))
        if m is None or m.pipe is None:
            raise BackendError(f"{payload.get('model')} is not loaded", code="NOT_LOADED")
        prompt = self._prompt(m, payload)
        cfg = self._generation_config(m, payload)
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()

        def streamer(text: str):
            loop.call_soon_threadsafe(queue.put_nowait, ("token", text))
            return og.StreamingStatus.CANCEL if cancelled.is_set() else og.StreamingStatus.RUNNING

        def run() -> None:
            try:
                with m.lock:
                    if m.pipe is None:
                        raise BackendError(f"{m.id} was unloaded", code="NOT_LOADED")
                    # A list input returns DecodedResults (with perf metrics); a bare str returns only text.
                    result = m.pipe.generate([prompt], cfg, streamer)
                pm = result.perf_metrics
                usage = {"prompt_tokens": pm.get_num_input_tokens(), "completion_tokens": pm.get_num_generated_tokens()}
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                finish = "length" if usage["completion_tokens"] >= cfg.max_new_tokens else "stop"
                loop.call_soon_threadsafe(queue.put_nowait, ("done", (usage, finish)))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", e))

        threading.Thread(target=run, name=f"ov-generate-{m.id}", daemon=True).start()
        base = {"id": cid, "created": created, "model": m.id}

        if not payload.get("stream"):
            parts: list[str] = []
            while True:
                kind, value = await queue.get()
                if kind == "token":
                    parts.append(value)
                elif kind == "error":
                    raise value if isinstance(value, BackendError) else BackendError(str(value), code="GENERATION_FAILED")
                else:
                    usage, finish = value
                    return {**base, "object": "chat.completion", "usage": usage, "choices": [{
                        "index": 0, "finish_reason": finish,
                        "message": {"role": "assistant", "content": "".join(parts)}}]}

        include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

        async def stream() -> AsyncIterator[bytes]:
            chunk = {**base, "object": "chat.completion.chunk"}
            yield _sse({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            try:
                while True:
                    kind, value = await queue.get()
                    if kind == "token":
                        yield _sse({**chunk, "choices": [{"index": 0, "delta": {"content": value}, "finish_reason": None}]})
                    elif kind == "error":
                        yield _sse({"error": {"message": str(value)}})
                        break
                    else:
                        usage, finish = value
                        if usage_sink is not None:
                            usage_sink.update(usage)
                        yield _sse({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                        if include_usage:
                            yield _sse({**chunk, "choices": [], "usage": usage})
                        break
                yield _sse("[DONE]")
            finally:
                cancelled.set()  # client went away: stop generating at the next token

        return stream()
