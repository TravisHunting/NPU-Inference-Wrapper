"""The interface every inference backend (Foundry Local today; OpenVINO, Lemonade, ... later) implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class BackendError(Exception):
    def __init__(self, message: str, *, code: str | None = None, hint: str | None = None, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint
        self.detail = detail

    def to_dict(self) -> dict:
        return {"message": str(self), "code": self.code, "hint": self.hint, "detail": self.detail}


@dataclass
class ModelVariant:
    id: str  # unique id the backend loads by, e.g. "qwen2.5-0.5b-instruct-openvino-npu:5"
    alias: str  # family name shared by all variants, e.g. "qwen2.5-0.5b"
    name: str
    device: str  # "npu" | "gpu" | "cpu"
    runtime: str | None  # execution provider / runtime, e.g. "OpenVINOExecutionProvider"
    task: str | None  # "chat" | "speech" | "embedding"
    size_mb: int | None
    license: str | None = None
    cached: bool = False
    loaded: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class ServiceStatus:
    installed: bool
    running: bool
    endpoint: str | None  # base URL of the backend's OpenAI-compatible server
    version: str | None = None
    detail: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str

    async def status(self) -> ServiceStatus: ...
    async def start(self) -> ServiceStatus: ...
    async def catalog(self, refresh: bool = False) -> list[ModelVariant]: ...
    async def download(self, model_id: str) -> str: ...
    async def load(self, model_id: str) -> str: ...
    async def unload(self, model_id: str) -> str: ...
    async def remove(self, model_id: str) -> str: ...
