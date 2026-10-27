"""Minimal ctypes wrapper over the Windows Performance Data Helper (PDH) API.

Only what we need: wildcard counters read back as {instance_name: value}. No admin rights required.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

PDH_FMT_DOUBLE = 0x00000200
PDH_FMT_NOCAP100 = 0x00008000
PDH_MORE_DATA = 0x800007D2
PDH_CSTATUS_VALID_DATA = 0x0
PDH_CSTATUS_NEW_DATA = 0x1
# Statuses that mean "nothing to report yet" (e.g. a rate counter before its second sample,
# or a wildcard with no instances right now) rather than a real failure.
PDH_NO_DATA = 0x800007D5
_EMPTY_STATUSES = {0xC0000BBA, 0xC0000BC6, PDH_NO_DATA}  # PDH_CSTATUS_INVALID_DATA, PDH_INVALID_DATA

_pdh = ctypes.WinDLL("pdh")


class _FmtCounterValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class _FmtCounterValueItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _FmtCounterValue)]


class PdhError(OSError):
    pass


def _check(status: int, what: str) -> None:
    if status & 0xFFFFFFFF != 0:
        raise PdhError(f"{what} failed: 0x{status & 0xFFFFFFFF:08x}")


class CounterQuery:
    """A PDH query holding a set of named wildcard counters."""

    def __init__(self, paths: dict[str, str]):
        self._query = wintypes.HANDLE()
        _check(_pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query)), "PdhOpenQuery")
        self._counters: dict[str, wintypes.HANDLE] = {}
        for key, path in paths.items():
            h = wintypes.HANDLE()
            _check(_pdh.PdhAddEnglishCounterW(self._query, path, None, ctypes.byref(h)), f"PdhAddEnglishCounter({path})")
            self._counters[key] = h

    def collect(self) -> dict[str, dict[str, float]]:
        """Sample all counters. Returns {counter_key: {instance_name: value}}."""
        status = _pdh.PdhCollectQueryData(self._query) & 0xFFFFFFFF
        if status not in (0, PDH_NO_DATA):
            raise PdhError(f"PdhCollectQueryData failed: 0x{status:08x}")
        return {key: self._read_array(h) for key, h in self._counters.items()}

    @staticmethod
    def _read_array(counter: wintypes.HANDLE) -> dict[str, float]:
        fmt = PDH_FMT_DOUBLE | PDH_FMT_NOCAP100
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        buf = None
        # Instances come and go between calls, so the required size can grow; retry a few times.
        for _ in range(4):
            status = _pdh.PdhGetFormattedCounterArrayW(counter, fmt, ctypes.byref(size), ctypes.byref(count), buf) & 0xFFFFFFFF
            if status in _EMPTY_STATUSES:
                return {}
            if status == PDH_MORE_DATA:
                buf = ctypes.create_string_buffer(size.value)
                continue
            _check(status, "PdhGetFormattedCounterArray")
            break
        else:
            return {}
        if buf is None:
            return {}
        items = ctypes.cast(buf, ctypes.POINTER(_FmtCounterValueItem))
        out: dict[str, float] = {}
        for i in range(count.value):
            item = items[i]
            if item.FmtValue.CStatus in (PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA):
                # Instance names can repeat (e.g. several engines per process); sum duplicates.
                out[item.szName] = out.get(item.szName, 0.0) + item.FmtValue.doubleValue
        return out

    def close(self) -> None:
        if self._query:
            _pdh.PdhCloseQuery(self._query)
            self._query = wintypes.HANDLE()
