#!/usr/bin/env python3
"""Versioned binary frame protocol used by XeSS stream workers."""

from __future__ import annotations

import dataclasses
import enum
import io
import struct
import zlib


MAGIC = b"XSPK"
VERSION = 1
PIXEL_RGB24 = 1
HEADER = struct.Struct("<4sHHIIIIIIIIII")
MAX_PAYLOAD = 512 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


class Flags(enum.IntFlag):
    NONE = 0
    RESET = 1 << 0
    SCENE_CUT = 1 << 1
    EOS = 1 << 2


@dataclasses.dataclass(slots=True)
class FramePacket:
    index: int
    width: int
    height: int
    pixel_format: int = PIXEL_RGB24
    flags: Flags = Flags.NONE
    color: bytes = b""
    motion: bytes = b""
    depth: bytes = b""
    mask: bytes = b""

    @property
    def payload(self) -> bytes:
        return self.color + self.motion + self.depth + self.mask


def _read_exact(stream: io.BufferedIOBase, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ProtocolError(f"unexpected EOF: wanted {count} bytes, got {count - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode(packet: FramePacket) -> bytes:
    if packet.index < 0 or packet.width < 0 or packet.height < 0:
        raise ProtocolError("negative frame metadata")
    payload = packet.payload
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError("payload exceeds protocol limit")
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = HEADER.pack(MAGIC, VERSION, HEADER.size, packet.index, packet.width,
                         packet.height, packet.pixel_format, int(packet.flags),
                         len(packet.color), len(packet.motion), len(packet.depth),
                         len(packet.mask), checksum)
    return header + payload


def write_packet(stream: io.BufferedIOBase, packet: FramePacket) -> None:
    stream.write(encode(packet))
    stream.flush()


def read_packet(stream: io.BufferedIOBase, *, expected_index: int | None = None) -> FramePacket:
    values = HEADER.unpack(_read_exact(stream, HEADER.size))
    (magic, version, header_size, index, width, height, pixel_format, flags,
     color_size, motion_size, depth_size, mask_size, checksum) = values
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: {magic!r}")
    if version != VERSION or header_size != HEADER.size:
        raise ProtocolError(f"unsupported protocol {version}/{header_size}")
    if expected_index is not None and index != expected_index:
        raise ProtocolError(f"out-of-order frame: got {index}, expected {expected_index}")
    sizes = (color_size, motion_size, depth_size, mask_size)
    total = sum(sizes)
    if total > MAX_PAYLOAD:
        raise ProtocolError("declared payload exceeds protocol limit")
    payload = _read_exact(stream, total)
    if (zlib.crc32(payload) & 0xFFFFFFFF) != checksum:
        raise ProtocolError("payload checksum mismatch")
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return FramePacket(index=index, width=width, height=height, pixel_format=pixel_format,
                       flags=Flags(flags), color=payload[offsets[0]:offsets[1]],
                       motion=payload[offsets[1]:offsets[2]],
                       depth=payload[offsets[2]:offsets[3]],
                       mask=payload[offsets[3]:offsets[4]])


def eos(index: int) -> FramePacket:
    return FramePacket(index=index, width=0, height=0, flags=Flags.EOS)
