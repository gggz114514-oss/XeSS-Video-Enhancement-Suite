#!/usr/bin/env python3
"""Single-producer/single-consumer Windows shared-memory packet ring."""

from __future__ import annotations

import ctypes
import mmap
import os
import struct
import time
import uuid


MAGIC = b"XSRG"
VERSION = 1
HEADER = struct.Struct("<4sIIIQQII24x")
SLOT_HEADER = struct.Struct("<II")
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
EVENT_MODIFY_STATE = 0x0002
SYNCHRONIZE = 0x00100000


class SharedRingError(RuntimeError):
    pass


def _windows_api():
    if os.name != "nt":
        raise SharedRingError("shared-memory ring is supported only on Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateEventW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p)
    kernel32.CreateEventW.restype = ctypes.c_void_p
    kernel32.OpenEventW.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p)
    kernel32.OpenEventW.restype = ctypes.c_void_p
    kernel32.SetEvent.argtypes = (ctypes.c_void_p,)
    kernel32.SetEvent.restype = ctypes.c_int
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    return kernel32


def _event_name(name: str, suffix: str) -> str:
    return f"{name}-{suffix}"


def _mapping_size(slots: int, slot_size: int) -> int:
    return HEADER.size + slots * (SLOT_HEADER.size + slot_size)


class RingOwner:
    """Creates the mapping/events and keeps them alive while children run."""

    def __init__(self, *, slots: int, slot_size: int, prefix: str = "xess"):
        if slots < 2 or slot_size <= 0:
            raise SharedRingError("invalid shared ring dimensions")
        self.slots = slots
        self.slot_size = slot_size
        self.name = f"Local\\{prefix}-{os.getpid()}-{uuid.uuid4().hex}"
        self._api = _windows_api()
        total = _mapping_size(slots, slot_size)
        self.mapping = mmap.mmap(-1, total, tagname=self.name, access=mmap.ACCESS_WRITE)
        self.data_event = self._api.CreateEventW(None, False, False, _event_name(self.name, "data"))
        self.space_event = self._api.CreateEventW(None, False, False, _event_name(self.name, "space"))
        if not self.data_event or not self.space_event:
            self.close()
            raise SharedRingError(f"CreateEventW failed: {ctypes.get_last_error()}")
        HEADER.pack_into(self.mapping, 0, MAGIC, VERSION, slots, slot_size, 0, 0, 0, 0)

    def arguments(self) -> list[str]:
        return ["--shm-name", self.name, "--shm-slots", str(self.slots),
                "--shm-slot-size", str(self.slot_size)]

    def close(self) -> None:
        mapping = getattr(self, "mapping", None)
        if mapping is not None:
            try:
                mapping.close()
            finally:
                self.mapping = None
        api = getattr(self, "_api", None)
        for attribute in ("data_event", "space_event"):
            handle = getattr(self, attribute, None)
            if api and handle:
                api.CloseHandle(handle)
                setattr(self, attribute, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class RingWriter:
    def __init__(self, name: str, slots: int, slot_size: int, timeout_seconds: float = 30.0):
        self.name = name
        self.slots = slots
        self.slot_size = slot_size
        self.timeout_ms = max(1, int(timeout_seconds * 1000))
        self.wait_seconds = 0.0  # time blocked because the consumer lagged
        self._api = _windows_api()
        self.mapping = mmap.mmap(-1, _mapping_size(slots, slot_size), tagname=name,
                                 access=mmap.ACCESS_WRITE)
        values = HEADER.unpack_from(self.mapping, 0)
        if values[:4] != (MAGIC, VERSION, slots, slot_size):
            self.mapping.close()
            raise SharedRingError(f"shared ring header mismatch: {values[:4]}")
        rights = EVENT_MODIFY_STATE | SYNCHRONIZE
        self.data_event = self._api.OpenEventW(rights, False, _event_name(name, "data"))
        self.space_event = self._api.OpenEventW(rights, False, _event_name(name, "space"))
        if not self.data_event or not self.space_event:
            self.close()
            raise SharedRingError(f"OpenEventW failed: {ctypes.get_last_error()}")

    def _sequences(self) -> tuple[int, int]:
        return struct.unpack_from("<QQ", self.mapping, 16)

    def write(self, packet: bytes) -> int:
        if len(packet) > self.slot_size:
            raise SharedRingError(
                f"packet is {len(packet)} bytes but shared slot capacity is {self.slot_size}"
            )
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while True:
            write_sequence, read_sequence = self._sequences()
            if write_sequence - read_sequence < self.slots:
                slot = write_sequence % self.slots
                offset = HEADER.size + slot * (SLOT_HEADER.size + self.slot_size)
                SLOT_HEADER.pack_into(self.mapping, offset, len(packet), 0)
                self.mapping[offset + SLOT_HEADER.size:offset + SLOT_HEADER.size + len(packet)] = packet
                struct.pack_into("<Q", self.mapping, 16, write_sequence + 1)
                if not self._api.SetEvent(self.data_event):
                    raise SharedRingError(f"SetEvent(data) failed: {ctypes.get_last_error()}")
                return len(packet)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SharedRingError("timed out waiting for a free shared-memory slot")
            waited_from = time.perf_counter()
            result = self._api.WaitForSingleObject(self.space_event, min(self.timeout_ms, int(remaining * 1000)))
            self.wait_seconds += time.perf_counter() - waited_from
            if result not in (WAIT_OBJECT_0, WAIT_TIMEOUT):
                raise SharedRingError(f"WaitForSingleObject(space) failed: {result}")

    def flush(self) -> None:
        return

    def close(self) -> None:
        mapping = getattr(self, "mapping", None)
        if mapping is not None:
            try:
                mapping.close()
            finally:
                self.mapping = None
        api = getattr(self, "_api", None)
        for attribute in ("data_event", "space_event"):
            handle = getattr(self, attribute, None)
            if api and handle:
                api.CloseHandle(handle)
                setattr(self, attribute, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def packet_slot_size(width: int, height: int, *, depth: bool, mask: bool) -> int:
    from stream_protocol import HEADER as PACKET_HEADER
    pixels = width * height
    return PACKET_HEADER.size + pixels * (3 + 8 + (4 if depth else 0) + (1 if mask else 0))
