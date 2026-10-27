"""FastAPI app: NPU telemetry, model management and the OpenAI-compatible proxy."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .backends.base import BackendError
from .backends.foundry import FoundryLocalBackend
from .backends.openvino import OpenVinoBackend
from .backends.router import ModelRouter
from .jobs import JobManager
from .proxy import InferenceProxy
from .telemetry.monitor import NpuMonitor, sample_to_dict

STATIC = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    monitor = NpuMonitor()
    backend = FoundryLocalBackend()
    openvino = OpenVinoBackend()
    router = ModelRouter(backend, openvino)
    jobs = JobManager()
    proxy = InferenceProxy(backend, openvino)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        monitor.start()
        yield
        monitor.stop()
        await proxy.close()

    app = FastAPI(title="NPU Harness", version=__version__, lifespan=lifespan)

    def backend_error(e: BackendError) -> HTTPException:
        return HTTPException(status_code=502, detail=e.to_dict())

    # -- NPU telemetry ----------------------------------------------------------------------

    @app.get("/api/npu")
    async def npu():
        latest = monitor.latest()
        return {
            "device": asdict(monitor.device) if monitor.device else None,
            "error": monitor.error,
            "interval": monitor.interval,
            "latest": sample_to_dict(latest) if latest else None,
        }

    @app.get("/api/npu/history")
    async def npu_history():
        return [sample_to_dict(s, include_processes=False) for s in monitor.history()]

    @app.get("/api/npu/stream")
    async def npu_stream(request: Request):
        """Server-sent events: one telemetry sample per monitor tick."""
        async def events():
            last = None
            while not await request.is_disconnected():
                s = monitor.latest()
                if s is not None and s.ts != last:
                    last = s.ts
                    yield f"data: {json.dumps(sample_to_dict(s))}\n\n"
                await asyncio.sleep(monitor.interval / 2)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # -- backend service --------------------------------------------------------------------

    @app.get("/api/backend")
    async def backend_status():
        try:
            st = await backend.status()
            info = await backend.system_info() if st.installed else {}
        except BackendError as e:
            raise backend_error(e)
        return {"name": backend.name, **asdict(st), "system": info, "openvino": router.status()}

    @app.post("/api/backend/start")
    async def backend_start():
        try:
            return asdict(await backend.start())
        except BackendError as e:
            raise backend_error(e)

    @app.get("/api/backend/logs")
    async def backend_logs(lines: int = 100):
        return {"text": await backend.recent_logs(min(max(lines, 1), 1000))}

    # -- models -----------------------------------------------------------------------------

    @app.get("/api/models")
    async def models(refresh: bool = False):
        try:
            return [asdict(m) for m in await router.catalog(refresh=refresh)]
        except BackendError as e:
            raise backend_error(e)

    actions = {"download": router.download, "load": router.load, "unload": router.unload, "remove": router.remove}

    @app.post("/api/models/{action}/{model_id:path}")
    async def model_action(action: str, model_id: str):
        if action not in actions:
            raise HTTPException(status_code=404, detail={"message": f"Unknown action {action!r}"})
        if not model_id.strip():
            raise HTTPException(status_code=400, detail={"message": "A model id is required."})
        fn = actions[action]
        return asdict(jobs.submit(action, model_id, lambda: fn(model_id)))

    @app.get("/api/jobs")
    async def list_jobs():
        return jobs.list()

    # -- inference --------------------------------------------------------------------------

    @app.get("/api/requests")
    async def recent_requests():
        return proxy.recent()

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def openai_proxy(path: str, request: Request):
        return await proxy.forward(request, path)

    # -- UI ---------------------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
