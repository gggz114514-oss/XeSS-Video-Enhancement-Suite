# -*- coding: utf-8 -*-
"""RAM / VRAM / disk probe with fallback chain.

Primary VRAM source is DXGI QueryVideoMemoryInfo (dxgi_vram.py).  On this
machine CreateDXGIFactory1 returns E_NOINTERFACE from *every* caller (ctypes,
C# P/Invoke) while the process sits in the interactive console session - a
system-level restriction (GPU virtualization / remote-desktop hooks).  When
DXGI COM fails we fall back to the GPU Adapter Memory performance counters,
which expose real dedicated usage.  The fallback is reported explicitly in the
JSON so no consumer can mistake it for DXGI data.

Usage:
    python memory_probe.py [--json-one] [--watch SECONDS]
"""
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from ctypes import wintypes

DEFAULT_WORK = os.environ.get(
    "XESS_RT_WORK",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "..", "..", "work")))
_DXGI_IMPORTED = False
_dxgi_probe = None
_HERE = os.path.dirname(os.path.abspath(__file__))


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def ram_mib():
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return {
        "load_pct": st.dwMemoryLoad,
        "total_phys_mib": st.ullTotalPhys / (1 << 20),
        "avail_phys_mib": st.ullAvailPhys / (1 << 20),
        "avail_pagefile_mib": st.ullAvailPageFile / (1 << 20),
    }


def _try_dxgi():
    """Return list of adapter dicts via DXGI COM, or None if unavailable."""
    global _DXGI_IMPORTED, _dxgi_probe
    if not _DXGI_IMPORTED:
        _DXGI_IMPORTED = True
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        try:
            import dxgi_vram
            _dxgi_probe = dxgi_vram
        except Exception:
            _dxgi_probe = None
    if _dxgi_probe is None:
        return None
    try:
        infos = _dxgi_probe.enumerate_adapters()
        out = []
        for i in infos:
            if "error" in i:
                continue
            out.append({
                "adapter": i["adapter"], "name": i["name"],
                "vendor": i["vendor"], "device": i["device"],
                "budget_mib": i.get("local_budget_mib"),
                "usage_mib": i.get("local_usage_mib"),
            })
        return out or None
    except Exception:
        return None


_COUNTER_QUERY = (
    "Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' "
    "| Select-Object -ExpandProperty CounterSamples "
    "| ForEach-Object { '{0}:{1}' -f $_.InstanceName, [long]$_.CookedValue }"
)

_vram_cache = {"t": 0.0, "rows": None}
_VRAM_TTL_S = 5.0


def _query_counter_once():
    """One Get-Counter invocation (~1.1 s); returns usage bytes per adapter."""
    try:
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _COUNTER_QUERY],
            capture_output=True, text=True, timeout=15)
        if ps.returncode != 0:
            return None
        rows = []
        for line in (ps.stdout or "").splitlines():
            m = re.match(r"luid_\S+_phys_\d+:(\d+)\s*$", line.strip())
            if m:
                rows.append(int(m.group(1)))
        return rows or None
    except Exception:
        return None


def shutdown():
    """Back-compat no-op: there is no background sampler to stop."""
    return None


def _perf_counter_vram():
    """Fallback: dedicated VRAM usage via the perf counter, cached for
    _VRAM_TTL_S so callers can poll freely without spawning PowerShell."""
    now = time.monotonic()
    if now - _vram_cache["t"] < _VRAM_TTL_S:
        rows = _vram_cache["rows"]
    else:
        rows = _query_counter_once()
        _vram_cache["t"] = now
        _vram_cache["rows"] = rows
    if rows:
        total = _nominal_vram_mib()
        return {
            "source": "perf-counter",
            "adapter_count": len(rows),
            "usage_mib": [r / (1 << 20) for r in rows],
            "budget_mib": max(0.0, (total - max(rows) / (1 << 20))) if total else None,
            "nominal_total_mib": total,
            "note": ("DXGI COM unavailable; budget approximated as "
                     "nominal dedicated size minus live usage"),
        }
    return {"source": "unavailable", "note": "no counter matched"}


def _nominal_vram_mib():
    """Nominal dedicated VRAM via DXGI GetDesc (adapter enumeration may be
    blocked on this machine); MiB or None.  Cached - it never changes."""
    global _nominal_cache
    try:
        return _nominal_cache
    except NameError:
        pass
    value = None
    try:
        import dxgi_vram
        for info in dxgi_vram.enumerate_adapters():
            if "error" not in info and info.get("dedicated_video_bytes"):
                value = info["dedicated_video_bytes"] / (1 << 20)
                break
    except Exception:
        value = None
    if not value:
        # DXGI factory is blocked on this machine; read the display class
        # registry key instead (HardwareInformation.qwMemorySize, full QWORD).
        import winreg
        try:
            key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(root, i)
                        i += 1
                    except OSError:
                        break
                    try:
                        with winreg.OpenKey(root, sub) as k:
                            qws, _kind = winreg.QueryValueEx(
                                k, "HardwareInformation.qwMemorySize")
                            size = qws[0] if isinstance(qws, tuple) else qws
                            if isinstance(size, int) and size > (1 << 28):
                                value = max(value or 0, size / (1 << 20))
                    except OSError:
                        continue
        except OSError:
            pass
    _nominal_cache = value
    return value


def vram():
    """Unified shape: source, usage_mib[list], budget_mib or None."""
    dxgi = _try_dxgi()
    if dxgi:
        first = dxgi[0]
        return {
            "source": "dxgi",
            "adapter_count": len(dxgi),
            "usage_mib": [a.get("budget_mib") and a.get("usage_mib")
                          for a in dxgi] and
                         [a.get("usage_mib") or 0.0 for a in dxgi],
            "budget_mib": first.get("budget_mib"),
            "note": "DXGI QueryVideoMemoryInfo",
        }
    return _perf_counter_vram()


def disk_mib(path=DEFAULT_WORK):
    try:
        u = shutil.disk_usage(path)
        return {
            "path": path,
            "total_mib": u.total / (1 << 20),
            "free_mib": u.free / (1 << 20),
            "is_system_drive": os.path.splitdrive(path)[0].lower()
            == os.path.splitdrive(os.environ.get("SystemDrive", "C:"))[0].lower(),
        }
    except OSError:
        return None


def snapshot(work_dir=DEFAULT_WORK):
    return {
        "t": time.time(),
        "ram": ram_mib(),
        "vram": vram(),
        "disk": disk_mib(work_dir),
    }


def main():
    if "--watch" in sys.argv:
        i = sys.argv.index("--watch")
        seconds = float(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 5.0
        work = sys.argv[i + 2] if i + 2 < len(sys.argv) else DEFAULT_WORK
        end = time.time() + seconds
        while time.time() < end:
            print(json.dumps(snapshot(work), ensure_ascii=False))
            time.sleep(0.5)
    else:
        work = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORK
        print(json.dumps(snapshot(work), ensure_ascii=False))


if __name__ == "__main__":
    main()
