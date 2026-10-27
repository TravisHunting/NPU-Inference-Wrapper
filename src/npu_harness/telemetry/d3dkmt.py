"""Enumerate graphics/compute adapters through the Windows kernel thunk API (gdi32 D3DKMT*).

NPUs on Windows are exposed as compute-only (MCDM) adapters. Enumerating adapters here
gives us each adapter's LUID, which is the key used by the "GPU Engine" and
"GPU Adapter Memory" performance counters, so we can pick the NPU's counters out exactly
instead of guessing.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

# D3DKMT_QUERYADAPTERINFO types (d3dkmthk.h)
KMTQAITYPE_ADAPTERREGISTRYINFO = 8
KMTQAITYPE_DRIVERVERSION = 13
KMTQAITYPE_ADAPTERTYPE = 15

# D3DKMT_ADAPTERTYPE bitfield
_RENDER_SUPPORTED = 1 << 0
_DISPLAY_SUPPORTED = 1 << 1
_SOFTWARE_DEVICE = 1 << 2
_COMPUTE_ONLY = 1 << 11

MAX_PATH = 260


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class D3DKMT_ADAPTERINFO(ctypes.Structure):
    _fields_ = [
        ("hAdapter", ctypes.c_uint),
        ("AdapterLuid", LUID),
        ("NumOfSources", wintypes.ULONG),
        ("bPrecisePresentRegionsPreferred", wintypes.BOOL),
    ]


# D3DKMT_ENUMADAPTERS_FILTER bits. EnumAdapters2 skips compute-only adapters, so NPUs
# only show up through EnumAdapters3 with IncludeComputeOnly set.
_FILTER_INCLUDE_COMPUTE_ONLY = 1 << 0


class D3DKMT_ENUMADAPTERS3(ctypes.Structure):
    _fields_ = [
        ("Filter", ctypes.c_uint64),
        ("NumAdapters", ctypes.c_uint),
        ("pAdapters", ctypes.POINTER(D3DKMT_ADAPTERINFO)),
    ]


class D3DKMT_QUERYADAPTERINFO(ctypes.Structure):
    _fields_ = [
        ("hAdapter", ctypes.c_uint),
        ("Type", ctypes.c_int),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", ctypes.c_uint),
    ]


class D3DKMT_ADAPTERREGISTRYINFO(ctypes.Structure):
    _fields_ = [
        ("AdapterString", ctypes.c_wchar * MAX_PATH),
        ("BiosString", ctypes.c_wchar * MAX_PATH),
        ("DacType", ctypes.c_wchar * MAX_PATH),
        ("ChipType", ctypes.c_wchar * MAX_PATH),
    ]


class D3DKMT_CLOSEADAPTER(ctypes.Structure):
    _fields_ = [("hAdapter", ctypes.c_uint)]


@dataclass(frozen=True)
class Adapter:
    luid: str  # formatted like the perf-counter instance names: luid_0x00000000_0x0001423a
    name: str
    wddm_version: int | None
    render: bool
    display: bool
    software: bool
    compute_only: bool

    @property
    def is_npu(self) -> bool:
        # A hardware adapter that can only do compute is an NPU (MCDM device).
        return self.compute_only and not self.software and not self.display


def format_luid(high: int, low: int) -> str:
    return f"luid_0x{high & 0xFFFFFFFF:08x}_0x{low & 0xFFFFFFFF:08x}"


def _query(gdi32, h_adapter: int, qtype: int, buf) -> bool:
    q = D3DKMT_QUERYADAPTERINFO(h_adapter, qtype, ctypes.cast(ctypes.byref(buf), ctypes.c_void_p), ctypes.sizeof(buf))
    return gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(q)) == 0


def enumerate_adapters() -> list[Adapter]:
    gdi32 = ctypes.WinDLL("gdi32")
    enum = D3DKMT_ENUMADAPTERS3(_FILTER_INCLUDE_COMPUTE_ONLY, 0, None)
    if gdi32.D3DKMTEnumAdapters3(ctypes.byref(enum)) != 0:
        return []
    infos = (D3DKMT_ADAPTERINFO * enum.NumAdapters)()
    enum.pAdapters = infos
    if gdi32.D3DKMTEnumAdapters3(ctypes.byref(enum)) != 0:
        return []

    adapters: list[Adapter] = []
    for info in infos[: enum.NumAdapters]:
        h = info.hAdapter
        try:
            kind = ctypes.c_uint(0)
            _query(gdi32, h, KMTQAITYPE_ADAPTERTYPE, kind)
            reg = D3DKMT_ADAPTERREGISTRYINFO()
            name = reg.AdapterString if _query(gdi32, h, KMTQAITYPE_ADAPTERREGISTRYINFO, reg) else ""
            wddm = ctypes.c_int(0)
            wddm_version = wddm.value if _query(gdi32, h, KMTQAITYPE_DRIVERVERSION, wddm) else None
            bits = kind.value
            adapters.append(
                Adapter(
                    luid=format_luid(info.AdapterLuid.HighPart, info.AdapterLuid.LowPart),
                    name=name,
                    wddm_version=wddm_version,
                    render=bool(bits & _RENDER_SUPPORTED),
                    display=bool(bits & _DISPLAY_SUPPORTED),
                    software=bool(bits & _SOFTWARE_DEVICE),
                    compute_only=bool(bits & _COMPUTE_ONLY),
                )
            )
        finally:
            gdi32.D3DKMTCloseAdapter(ctypes.byref(D3DKMT_CLOSEADAPTER(h)))
    return adapters
