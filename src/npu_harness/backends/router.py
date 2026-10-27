"""Routes model operations between Foundry Local and the in-process OpenVINO GenAI backend.

Foundry Local is always tried first. If it can't load an OpenVINO variant, the same cached files
are loaded directly with OpenVINO GenAI. The failure is remembered per runtime for the rest of the
session, so later loads skip straight to the direct path instead of failing again first.
"""

from __future__ import annotations

from .base import BackendError, ModelVariant
from .foundry import FoundryLocalBackend
from .openvino import OpenVinoBackend


class ModelRouter:
    def __init__(self, foundry: FoundryLocalBackend, openvino: OpenVinoBackend):
        self.foundry = foundry
        self.openvino = openvino
        self._foundry_broken: dict[str, str] = {}  # runtime -> why Foundry couldn't start it

    async def catalog(self, refresh: bool = False) -> list[ModelVariant]:
        variants = await self.foundry.catalog(refresh=refresh)
        direct = self.openvino.loaded_ids()
        for v in variants:
            if v.id in direct:
                v.loaded = True
                v.extra["engine"] = "openvino-genai"
            else:
                v.extra["engine"] = "foundry-local" if v.loaded else None
            v.extra["direct_capable"] = self.openvino.can_run(v.device, v.runtime)
        return variants

    async def _variant(self, model_id: str) -> ModelVariant | None:
        return next((v for v in await self.foundry.catalog() if v.id == model_id), None)

    def status(self) -> dict:
        return {**self.openvino.info(), "foundry_fallbacks": dict(self._foundry_broken)}

    async def download(self, model_id: str) -> str:
        return await self.foundry.download(model_id)

    async def load(self, model_id: str) -> str | tuple[str, str]:
        v = await self._variant(model_id)
        direct = v is not None and self.openvino.can_run(v.device, v.runtime)
        if direct and v.runtime in self._foundry_broken:
            return await self._load_direct(v, self._foundry_broken[v.runtime])
        try:
            return await self.foundry.load(model_id)
        except BackendError as e:
            if not direct:
                raise
            reason = e.detail or str(e)
            try:
                result = await self._load_direct(v, reason)
            except BackendError as direct_error:
                # Both paths failed: report Foundry's error, with the direct attempt alongside.
                e.hint = f"{e.hint or ''} Loading directly with OpenVINO GenAI also failed: {direct_error}" \
                         f"{f' ({direct_error.detail})' if direct_error.detail else ''}".strip()
                raise e from direct_error
            self._foundry_broken[v.runtime] = reason
            return result

    async def _load_direct(self, v: ModelVariant, reason: str) -> tuple[str, str]:
        path = await self.foundry.model_path(v.id)
        if path is None:
            raise BackendError(f"{v.id} isn't downloaded yet.", code="NOT_CACHED", hint="Download it first.")
        message = await self.openvino.load(v.id, path, v.device)
        notice = (f"Foundry Local couldn't start {v.runtime} ({reason}), so the harness is running this model "
                  f"directly with OpenVINO GenAI on the {v.device.upper()}. It's served at the same /v1 endpoint.")
        return message, notice

    async def unload(self, model_id: str) -> str:
        if self.openvino.is_loaded(model_id):
            return await self.openvino.unload(model_id)
        return await self.foundry.unload(model_id)

    async def remove(self, model_id: str) -> str:
        if self.openvino.is_loaded(model_id):
            await self.openvino.unload(model_id)
        self.openvino.discard_staging(model_id)
        return await self.foundry.remove(model_id)
