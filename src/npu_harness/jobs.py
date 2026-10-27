"""Tracks long-running backend operations (download, load, ...) so the UI can poll them."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from .backends.base import BackendError


@dataclass
class Job:
    id: int
    action: str
    model_id: str
    state: str = "running"  # running | succeeded | failed
    message: str | None = None
    notice: str | None = None  # something the user should know even though the job succeeded
    error: dict | None = None
    started: float = 0.0
    finished: float | None = None


class JobManager:
    def __init__(self, keep: int = 50):
        self._jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._keep = keep
        self._tasks: set[asyncio.Task] = set()

    def active(self, action: str, model_id: str) -> Job | None:
        return next((j for j in self._jobs.values()
                     if j.state == "running" and j.action == action and j.model_id == model_id), None)

    def submit(self, action: str, model_id: str, fn: Callable[[], Awaitable[str | tuple[str, str]]]) -> Job:
        if existing := self.active(action, model_id):
            return existing
        job = Job(next(self._ids), action, model_id, started=time.time())
        self._jobs[job.id] = job

        async def run() -> None:
            try:
                result = await fn()
                job.message, job.notice = result if isinstance(result, tuple) else (result, None)
                job.state = "succeeded"
            except BackendError as e:
                job.state, job.error = "failed", e.to_dict()
            except Exception as e:  # surface anything unexpected to the UI rather than losing it
                job.state, job.error = "failed", {"message": str(e) or type(e).__name__}
            finally:
                job.finished = time.time()
                self._trim()

        task = asyncio.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def _trim(self) -> None:
        done = [j for j in self._jobs.values() if j.state != "running"]
        for j in sorted(done, key=lambda j: j.id)[: max(0, len(done) - self._keep)]:
            del self._jobs[j.id]

    def list(self) -> list[dict]:
        return [asdict(j) for j in sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)]
