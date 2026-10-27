"""Stable OpenAI-compatible endpoint: forwards /v1/* to the backend and records per-request stats.

Foundry Local picks a new port each time its daemon starts; clients pointed at the harness
(http://127.0.0.1:8765/v1) keep working across restarts.
"""

from __future__ import annotations

import collections
import json
import time
from dataclasses import asdict, dataclass

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .backends.base import BackendError
from .backends.foundry import FoundryLocalBackend

_HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "accept-encoding"}


@dataclass
class RequestStat:
    ts: float
    path: str
    model: str | None
    status: int
    stream: bool
    ttft_ms: float | None = None  # time to first content token
    duration_ms: float | None = None
    completion_tokens: int | None = None
    tokens_per_sec: float | None = None


class InferenceProxy:
    def __init__(self, backend: FoundryLocalBackend, keep: int = 100):
        self.backend = backend
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=600, write=60, pool=5))
        self.stats: collections.deque[RequestStat] = collections.deque(maxlen=keep)

    async def close(self) -> None:
        await self.client.aclose()

    def recent(self) -> list[dict]:
        return [asdict(s) for s in reversed(self.stats)]

    async def _send(self, request: Request, path: str, body: bytes) -> httpx.Response:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
        for attempt in range(2):
            base = await self.backend.endpoint()
            req = self.client.build_request(request.method, f"{base}/v1/{path}", params=request.query_params,
                                            headers=headers, content=body)
            try:
                return await self.client.send(req, stream=True)
            except httpx.ConnectError:
                # Daemon restarted on a new port (or stopped); rediscover once.
                self.backend.forget_endpoint()
                if attempt:
                    raise
        raise AssertionError("unreachable")

    async def forward(self, request: Request, path: str) -> Response:
        body = await request.body()
        payload: dict = {}
        if body and "json" in request.headers.get("content-type", ""):
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                pass
        started = time.time()  # before sending: non-streaming responses arrive only once generation is done
        try:
            upstream = await self._send(request, path, body)
        except BackendError as e:
            return JSONResponse({"error": e.to_dict()}, status_code=503)
        except httpx.HTTPError as e:
            return JSONResponse({"error": {"message": f"Backend unreachable: {e}"}}, status_code=502)

        stat = RequestStat(ts=started, path=path, model=payload.get("model"), status=upstream.status_code,
                           stream=bool(payload.get("stream")))
        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_HEADERS}
        track = path.endswith("completions") and upstream.status_code == 200

        if not stat.stream:
            content = await upstream.aread()
            await upstream.aclose()
            stat.duration_ms = (time.time() - stat.ts) * 1000
            if track:
                try:
                    usage = json.loads(content).get("usage") or {}
                    stat.completion_tokens = usage.get("completion_tokens")
                except (json.JSONDecodeError, AttributeError):
                    pass
                self._finish(stat)
            return Response(content, status_code=upstream.status_code, headers=resp_headers)

        async def relay():
            tokens = 0
            buffer = b""
            try:
                async for chunk in upstream.aiter_raw():
                    if track:
                        buffer += chunk
                        *lines, buffer = buffer.split(b"\n")
                        for line in lines:
                            if line.startswith(b"data:") and b'"content":"' in line and b'"content":""' not in line:
                                tokens += 1  # Foundry streams roughly one token per chunk
                                if stat.ttft_ms is None:
                                    stat.ttft_ms = (time.time() - stat.ts) * 1000
                    yield chunk
            finally:
                await upstream.aclose()
                if track:
                    stat.duration_ms = (time.time() - stat.ts) * 1000
                    stat.completion_tokens = tokens
                    self._finish(stat)

        return StreamingResponse(relay(), status_code=upstream.status_code, headers=resp_headers)

    def _finish(self, stat: RequestStat) -> None:
        if stat.completion_tokens and stat.duration_ms:
            gen_ms = stat.duration_ms - (stat.ttft_ms or 0)
            if gen_ms > 0:
                stat.tokens_per_sec = round(stat.completion_tokens / (gen_ms / 1000), 1)
        self.stats.append(stat)
