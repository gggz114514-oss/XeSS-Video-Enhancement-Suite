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


def _map_part(part):
    """Byte measure of a payload part (bytes or contiguous numpy view)."""
    size = getattr(part, "nbytes", None)
    return size if size is not None else len(part)


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

    def _wait_slot(self) -> int:
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while True:
            write_sequence, read_sequence = self._sequences()
            if write_sequence - read_sequence < self.slots:
                return write_sequence
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SharedRingError("timed out waiting for a free shared-memory slot")
            waited_from = time.perf_counter()
            result = self._api.WaitForSingleObject(self.space_event,
                                                   min(self.timeout_ms, int(remaining * 1000)))
            self.wait_seconds += time.perf_counter() - waited_from
            if result not in (WAIT_OBJECT_0, WAIT_TIMEOUT):
                raise SharedRingError(f"WaitForSingleObject(space) failed: {result}")

    def write(self, packet: bytes) -> int:
        if len(packet) > self.slot_size:
            raise SharedRingError(
                f"packet is {len(packet)} bytes but shared slot capacity is {self.slot_size}"
            )
        write_sequence = self._wait_slot()
        slot = write_sequence % self.slots
        offset = HEADER.size + slot * (SLOT_HEADER.size + self.slot_size)
        SLOT_HEADER.pack_into(self.mapping, offset, len(packet), 0)
        self.mapping[offset + SLOT_HEADER.size:offset + SLOT_HEADER.size + len(packet)] = packet
        struct.pack_into("<Q", self.mapping, 16, write_sequence + 1)
        if not self._api.SetEvent(self.data_event):
            raise SharedRingError(f"SetEvent(data) failed: {ctypes.get_last_error()}")
        return len(packet)

    def write_parts(self, header: bytes, parts) -> int:
        """Write a pre-split packet (protocol header + payload parts) into one slot.

        ``parts`` may mix ``bytes`` and contiguous numpy arrays; each is
        written into the mmap slot directly, so the payload is never
        concatenated or copied through an intermediate ``bytes`` object.
        The header must already carry the incremental CRC over the parts.
        """
        total = len(header) + sum(_map_part(part) for part in parts)
        if total > self.slot_size:
            raise SharedRingError(
                f"packet is {total} bytes but shared slot capacity is {self.slot_size}"
            )
        write_sequence = self._wait_slot()
        slot = write_sequence % self.slots
        offset = HEADER.size + slot * (SLOT_HEADER.size + self.slot_size)
        SLOT_HEADER.pack_into(self.mapping, offset, total, 0)
        cursor = offset + SLOT_HEADER.size
        for part in (header, *parts):
            size = _map_part(part)
            if size:
                self.mapping[cursor:cursor + size] = part
                cursor += size
        struct.pack_into("<Q", self.mapping, 16, write_sequence + 1)
        if not self._api.SetEvent(self.data_event):
            raise SharedRingError(f"SetEvent(data) failed: {ctypes.get_last_error()}")
        return total

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


class RingReader:
    """Consumer side of the shared ring (mirror of the C++ reader).

    ``read()`` returns the full slot payload as bytes and advances the ring.
    Used by downstream components (e.g. sr_postprocess) that consume raw
    per-frame buffers written by a C++ writer.
    """

    def __init__(self, name: str, slots: int, slot_size: int, timeout_seconds: float = 30.0):
        self.name = name
        self.slots = slots
        self.slot_size = slot_size
        self.timeout_ms = max(1, int(timeout_seconds * 1000))
        self.wait_seconds = 0.0
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

    def read(self) -> bytes:
        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while True:
            write_sequence, read_sequence = self._sequences()
            if read_sequence < write_sequence:
                slot = read_sequence % self.slots
                offset = HEADER.size + slot * (SLOT_HEADER.size + self.slot_size)
                (packet_size,) = SLOT_HEADER.unpack_from(self.mapping, offset)
                if not packet_size or packet_size > self.slot_size:
                    raise SharedRingError(f"invalid packet size {packet_size}")
                payload = bytes(self.mapping[offset + SLOT_HEADER.size:
                                             offset + SLOT_HEADER.size + packet_size])
                struct.pack_into("<Q", self.mapping, 24, read_sequence + 1)
                if not self._api.SetEvent(self.space_event):
                    raise SharedRingError(f"SetEvent(space) failed: {ctypes.get_last_error()}")
                return payload
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SharedRingError(f"timed out waiting for a packet (frames lagging)")
            waited_from = time.perf_counter()
            result = self._api.WaitForSingleObject(
                self.data_event, min(self.timeout_ms, int(remaining * 1000)))
            self.wait_seconds += time.perf_counter() - waited_from
            if result not in (WAIT_OBJECT_0, WAIT_TIMEOUT):
                raise SharedRingError(f"WaitForSingleObject(data) failed: {result}")

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
