"""Background NPU sampler: utilization, memory and per-process usage over time."""

from __future__ import annotations

import collections
import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field

import psutil

from .d3dkmt import Adapter, enumerate_adapters
from .pdh import CounterQuery, PdhError

log = logging.getLogger(__name__)

_ENGINE_RE = re.compile(r"pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+_eng_(\d+)_engtype_(\w*)", re.I)
_PROC_MEM_RE = re.compile(r"pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+", re.I)
_ADAPTER_MEM_RE = re.compile(r"(luid_0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+", re.I)


@dataclass
class NpuDevice:
    luid: str
    name: str
    vendor: str | None = None
    driver_version: str | None = None
    driver_date: str | None = None
    wddm_version: int | None = None
    pnp_id: str | None = None


@dataclass
class ProcessUsage:
    pid: int
    name: str
    utilization: float
    shared_bytes: int


@dataclass
class NpuSample:
    ts: float
    utilization: float
    shared_bytes: int
    dedicated_bytes: int
    system_memory_total: int
    system_memory_available: int
    processes: list[ProcessUsage] = field(default_factory=list)


def _pnp_compute_accelerators() -> list[dict]:
    """Friendly name and driver details for NPUs, from the PnP driver store (one-off, ~1s)."""
    script = (
        "Get-CimInstance Win32_PnPSignedDriver -Filter \"DeviceClass='COMPUTEACCELERATOR'\" | "
        "Select-Object DeviceName,Manufacturer,DriverVersion,"
        "@{n='DriverDate';e={if($_.DriverDate){$_.DriverDate.ToString('yyyy-MM-dd')}}},DeviceID | "
        "ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout.strip()
        data = json.loads(out) if out else []
        return data if isinstance(data, list) else [data]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        log.warning("Could not query PnP driver info: %s", e)
        return []


def detect_npu() -> NpuDevice | None:
    """Fast path: find the NPU adapter. Driver details are filled in later by enrich_device()."""
    npus: list[Adapter] = [a for a in enumerate_adapters() if a.is_npu]
    if not npus:
        return None
    adapter = npus[0]
    return NpuDevice(luid=adapter.luid, name=adapter.name or "NPU", wddm_version=adapter.wddm_version)


def enrich_device(device: NpuDevice) -> None:
    pnp = _pnp_compute_accelerators()
    if not pnp:
        return
    info = pnp[0]
    if device.name == "NPU" and info.get("DeviceName"):
        device.name = info["DeviceName"]
    device.vendor = info.get("Manufacturer")
    device.driver_version = info.get("DriverVersion")
    device.driver_date = info.get("DriverDate")
    device.pnp_id = info.get("DeviceID")


class NpuMonitor:
    def __init__(self, interval: float = 1.0, history_seconds: int = 300):
        self.interval = interval
        self.device: NpuDevice | None = None
        self.error: str | None = None
        self._history: collections.deque[NpuSample] = collections.deque(maxlen=int(history_seconds / interval))
        self._names: dict[int, str] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="npu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> NpuSample | None:
        return self._history[-1] if self._history else None

    def history(self) -> list[NpuSample]:
        return list(self._history)

    def _process_name(self, pid: int) -> str:
        if pid not in self._names:
            try:
                self._names[pid] = psutil.Process(pid).name()
            except (psutil.Error, OSError):
                self._names[pid] = "System" if pid == 4 else f"pid {pid}"
        return self._names[pid]

    def _run(self) -> None:
        try:
            self.device = detect_npu()
        except OSError as e:
            self.error = f"Adapter enumeration failed: {e}"
        if self.device is None:
            self.error = self.error or "No NPU (compute-only adapter) found on this machine."
            return
        threading.Thread(target=enrich_device, args=(self.device,), name="npu-enrich", daemon=True).start()

        luid = self.device.luid.lower()
        try:
            query = CounterQuery({
                "engine": r"\GPU Engine(*)\Utilization Percentage",
                "adapter_shared": r"\GPU Adapter Memory(*)\Shared Usage",
                "adapter_dedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
                "process_shared": r"\GPU Process Memory(*)\Shared Usage",
            })
        except PdhError as e:
            self.error = f"Performance counters unavailable: {e}"
            return

        try:
            query.collect()  # prime rate counters
            while not self._stop.wait(self.interval):
                try:
                    self._history.append(self._sample(query.collect(), luid))
                except PdhError as e:
                    log.warning("NPU sample failed: %s", e)
        finally:
            query.close()

    def _sample(self, data: dict[str, dict[str, float]], luid: str) -> NpuSample:
        # Utilization: sum each engine's usage across processes, then take the busiest engine
        # (the same way Task Manager reports a device's overall utilization).
        per_engine: dict[str, float] = collections.defaultdict(float)
        per_process: dict[int, float] = collections.defaultdict(float)
        for inst, value in data["engine"].items():
            m = _ENGINE_RE.match(inst)
            if m and m.group(2).lower() == luid:
                per_engine[m.group(3)] += value
                per_process[int(m.group(1))] = max(per_process[int(m.group(1))], value)

        shared_by_pid: dict[int, int] = {}
        for inst, value in data["process_shared"].items():
            m = _PROC_MEM_RE.match(inst)
            if m and m.group(2).lower() == luid and value > 0:
                shared_by_pid[int(m.group(1))] = int(value)

        def adapter_total(key: str) -> int:
            return int(sum(v for inst, v in data[key].items()
                           if (m := _ADAPTER_MEM_RE.match(inst)) and m.group(1).lower() == luid))

        pids = {p for p, u in per_process.items() if u > 0} | set(shared_by_pid)
        processes = sorted(
            (ProcessUsage(pid, self._process_name(pid), round(per_process.get(pid, 0.0), 2), shared_by_pid.get(pid, 0))
             for pid in pids),
            key=lambda p: (p.utilization, p.shared_bytes), reverse=True,
        )
        vm = psutil.virtual_memory()
        return NpuSample(
            ts=time.time(),
            utilization=round(min(100.0, max(per_engine.values(), default=0.0)), 2),
            shared_bytes=adapter_total("adapter_shared"),
            dedicated_bytes=adapter_total("adapter_dedicated"),
            system_memory_total=vm.total,
            system_memory_available=vm.available,
            processes=processes,
        )


def sample_to_dict(s: NpuSample, include_processes: bool = True) -> dict:
    d = asdict(s)
    if not include_processes:
        d.pop("processes")
    return d
