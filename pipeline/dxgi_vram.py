# -*- coding: utf-8 -*-
"""DXGI video memory budget/usage probe (ctypes, no extra deps).

Reads QueryVideoMemoryInfo through IDXGIFactory1 -> IDXGIAdapter2 exactly the
way the plan's MemoryBudget will, so Phase 0 metrics and Phase 10 share one
implementation.  Uses DXGI_MEMORY_SEGMENT_GROUP_LOCAL (dedicated VRAM).

Usage:
    python dxgi_vram.py                 -> JSON line per adapter
    python dxgi_vram.py --adapter 0     -> only adapter 0
"""
import ctypes
import json
import sys
from ctypes import wintypes

DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL = 1

# {770aae78-f26f-4dba-a829-c1c78e4f9527} IDXGIFactory1 (fields little-endian)
_GUID_FACTORY1 = bytes.fromhex("78ae0a776ff2ba4da829c1c78e4f9527")
# {50c83a1c-e072-4c48-87b0-3630fa36a6d0} IDXGIAdapter2 (fields little-endian)
_GUID_ADAPTER2 = bytes.fromhex("1c3ac85072e0484c87b03630fa36a6d0")
# {645967a4-1392-4310-a798-8053ce3e93fd} IDXGIAdapter3 (fields little-endian)
_GUID_ADAPTER3 = bytes.fromhex("a467596592133010a7988053ce3e93fd")

# vtable indices verified against Windows SDK 10.0.26100.0 headers
# (dxgi.h, dxgi1_2.h, dxgi1_4.h):
#   IDXGIFactory1: 0..2 IUnknown, 3 SetPrivateData, 4 SetPrivateDataInterface,
#                  5 GetPrivateData, 6 GetParent, 7 EnumAdapters
#   IDXGIAdapter2: 7 EnumOutputs, 8 GetDesc, 9 CheckInterfaceSupport,
#                  10 GetDesc1, 11 GetDesc2
#   IDXGIAdapter3: +12 RegisterHardwareContentProtectionTeardownStatusEvent,
#                  13 Unregister..., 14 QueryVideoMemoryInfo
FACTORY_ENUM_ADAPTERS = 7
ADAPTER2_GETDESC1 = 10
ADAPTER3_QUERY_VIDEO_MEMORY_INFO = 14


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
    ]


class DXGI_ADAPTER_DESC_BASE(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", ctypes.c_uint64),
    ]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", ctypes.c_uint64),
        ("Flags", wintypes.UINT),
    ]


def _guid(hexstr: bytes) -> GUID:
    return GUID(
        int.from_bytes(hexstr[0:4], "little"),
        int.from_bytes(hexstr[4:6], "little"),
        int.from_bytes(hexstr[6:8], "little"),
        (ctypes.c_ubyte * 8).from_buffer_copy(hexstr[8:16]),
    )


def _vtbl(iface):
    """Return the vtable pointer array of a COM interface."""
    return ctypes.cast(iface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents


def _call(iface, index, restype, argtypes):
    # COM vtable methods are __stdcall on Windows.
    fn = ctypes.cast(_vtbl(iface)[index], ctypes.WINFUNCTYPE(restype, *argtypes))
    return fn


HRESULT = ctypes.c_long


def enumerate_adapters(adapter_filter=None):
    dxgi = ctypes.windll.dxgi
    create = dxgi.CreateDXGIFactory1
    create.restype = HRESULT
    create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    factory = ctypes.c_void_p()
    hr = create(ctypes.byref(_guid(_GUID_FACTORY1)), ctypes.byref(factory))
    if hr != 0:
        raise OSError(f"CreateDXGIFactory1 failed hr=0x{hr & 0xffffffff:08x}")

    results = []
    index = 0
    while True:
        adapter = ctypes.c_void_p()
        enum = _call(factory, FACTORY_ENUM_ADAPTERS, HRESULT,  # EnumAdapters
                     [ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p)])
        hr = enum(factory, index, ctypes.byref(adapter))
        if hr == 0x887A002D:  # DXGI_ERROR_NOT_FOUND
            break
        if hr != 0:
            raise OSError(f"EnumAdapters({index}) hr=0x{hr & 0xffffffff:08x}")
        try:
            results.append(_adapter_info(adapter, index, adapter_filter))
        except OSError as e:
            results.append({"adapter": index, "error": str(e)})
        index += 1
    return results


def _adapter_info(adapter, index, adapter_filter):
    # QI to IDXGIAdapter2 for GetDesc1 (the enumerator returns IDXGIAdapter).
    a2 = ctypes.c_void_p()
    qi = _call(adapter, 0, HRESULT,  # IUnknown::QueryInterface
               [ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)])
    hr = qi(adapter, ctypes.byref(_guid(_GUID_ADAPTER2)), ctypes.byref(a2))
    if hr != 0 or not a2:
        # fall back to IDXGIAdapter::GetDesc (vtable index 4)
        desc_base = DXGI_ADAPTER_DESC_BASE()
        getdesc = _call(adapter, 4, HRESULT,
                        [ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC_BASE)])
        if getdesc(adapter, ctypes.byref(desc_base)) != 0:
            raise OSError("GetDesc failed")
        name = desc_base.Description
        dedicated = desc_base.DedicatedVideoMemory
        has_query = False
    else:
        desc = DXGI_ADAPTER_DESC1()
        getdesc1 = _call(a2, ADAPTER2_GETDESC1, HRESULT,
                         [ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1)])
        if getdesc1(a2, ctypes.byref(desc)) != 0:
            raise OSError("GetDesc1 failed")
        name = desc.Description
        dedicated = desc.DedicatedVideoMemory
        # QI to IDXGIAdapter3, which owns QueryVideoMemoryInfo (vtable 14).
        a3 = ctypes.c_void_p()
        hr3 = qi(a2, ctypes.byref(_guid(_GUID_ADAPTER3)), ctypes.byref(a3))
        has_query = hr3 == 0 and bool(a3)

    if adapter_filter is not None and int(adapter_filter) != index:
        return None

    info = {"adapter": index, "name": name,
            "dedicated_video_bytes": dedicated}
    if has_query:
        q = DXGI_QUERY_VIDEO_MEMORY_INFO()
        query = _call(a3, ADAPTER3_QUERY_VIDEO_MEMORY_INFO, HRESULT,
                      [ctypes.c_void_p, wintypes.UINT, wintypes.UINT,
                       ctypes.POINTER(DXGI_QUERY_VIDEO_MEMORY_INFO)])
        for seg, label in ((DXGI_MEMORY_SEGMENT_GROUP_LOCAL, "local"),
                           (DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, "nonlocal")):
            if query(a3, 0, seg, ctypes.byref(q)) == 0:
                info[f"{label}_budget_mib"] = q.Budget / (1 << 20)
                info[f"{label}_usage_mib"] = q.CurrentUsage / (1 << 20)
                info[f"{label}_avail_resv_mib"] = q.AvailableForReservation / (1 << 20)
                info[f"{label}_cur_resv_mib"] = q.CurrentReservation / (1 << 20)
            else:
                info[f"{label}_error"] = "query failed"
    else:
        info["budget_source"] = "getdesc-only (no Adapter3)"
    return info


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--adapter")]
    adapter_filter = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--adapter" and i + 1 < len(sys.argv):
            adapter_filter = sys.argv[i + 1]
    infos = [r for r in enumerate_adapters(adapter_filter) if r is not None]
    print(json.dumps(infos, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
