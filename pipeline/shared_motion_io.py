"""Bounded on-disk side channel for the offline CPU SR→FG bridge.

The side channel carries full flow/depth/mask/confidence arrays in system
memory and bounded disk files, never a source or SR RGB frame. These arrays
are not small metadata. A producer publishes one atomically-renamed packet and waits when
the bounded directory already contains ``slots`` packets.  The bridge removes
each packet after turning it into an output-resolution FramePacket.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
import time

import numpy as np


MAGIC = b"SMOT"
VERSION = 2
# magic, version, frame id, source width/height, pts prev/current numerator,
# pts denominator, flow/depth/mask byte counts, stream flags
HEADER_V1 = struct.Struct("<4sIIIIqqqIIII")
HEADER = struct.Struct("<4sIIIIqqqIIIII")


@dataclass(slots=True)
class SidecarPacket:
    frame_id: int
    width: int
    height: int
    prev_pts_num: int
    current_pts_num: int
    pts_den: int
    flags: int
    flow: np.ndarray
    depth: np.ndarray | None
    mask: np.ndarray | None
    confidence: np.ndarray | None = None

    @property
    def path_key(self) -> str:
        return f"packet_{self.frame_id:08d}"


def _packet_path(directory: str | os.PathLike[str], frame_id: int) -> Path:
    return Path(directory) / f"packet_{int(frame_id):08d}.smot"


def _wait_for_capacity(directory: Path, slots: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        count = len(tuple(directory.glob("packet_*.smot")))
        if count < slots:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("shared motion sidecar ring is full")
        time.sleep(0.005)


def write_sidecar(directory: str | os.PathLike[str], packet: SidecarPacket,
                  *, slots: int = 4, timeout: float = 30.0) -> Path:
    if slots < 2 or timeout <= 0 or packet.pts_den <= 0:
        raise ValueError("invalid sidecar capacity, timeout or time base")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = _packet_path(root, packet.frame_id)
    _wait_for_capacity(root, slots, timeout)
    flow = np.ascontiguousarray(packet.flow, dtype=np.float32)
    if flow.shape != (packet.height, packet.width, 2):
        raise ValueError(f"sidecar flow shape {flow.shape} does not match source")
    depth = (np.ascontiguousarray(packet.depth, dtype=np.float32)
             if packet.depth is not None else None)
    mask = (np.ascontiguousarray(packet.mask, dtype=np.uint8)
            if packet.mask is not None else None)
    if depth is not None and depth.shape != (packet.height, packet.width):
        raise ValueError("sidecar depth shape does not match source")
    if mask is not None and mask.shape != (packet.height, packet.width):
        raise ValueError("sidecar mask shape does not match source")
    confidence = (np.ascontiguousarray(packet.confidence, dtype=np.float32)
                  if packet.confidence is not None else None)
    if confidence is not None and confidence.shape != (packet.height, packet.width):
        raise ValueError("sidecar confidence shape does not match source")
    header = HEADER.pack(
        MAGIC, VERSION, int(packet.frame_id), int(packet.width), int(packet.height),
        int(packet.prev_pts_num), int(packet.current_pts_num), int(packet.pts_den),
        flow.nbytes, 0 if depth is None else depth.nbytes,
        0 if mask is None else mask.nbytes, int(packet.flags),
        0 if confidence is None else confidence.nbytes)
    partial = root / f".{target.name}.{os.getpid()}.partial"
    with partial.open("wb") as handle:
        handle.write(header)
        # The synchronous file write already consumes the contiguous buffer;
        # materializing a second full-array bytes object adds no ownership or
        # durability guarantee. Keep flush/fsync and atomic publication below.
        handle.write(memoryview(flow).cast("B"))
        if depth is not None:
            handle.write(memoryview(depth).cast("B"))
        if mask is not None:
            handle.write(memoryview(mask).cast("B"))
        if confidence is not None:
            handle.write(memoryview(confidence).cast("B"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, target)
    return target


def read_sidecar(path: str | os.PathLike[str]) -> SidecarPacket:
    source = Path(path)
    raw = source.read_bytes()
    if len(raw) < HEADER_V1.size:
        raise RuntimeError(f"truncated shared motion packet: {source}")
    version = struct.unpack_from("<I", raw, 4)[0]
    header = HEADER if version == VERSION else HEADER_V1
    if len(raw) < header.size:
        raise RuntimeError(f"truncated shared motion packet: {source}")
    values = header.unpack_from(raw, 0)
    (magic, version, frame_id, width, height, prev_pts_num,
     current_pts_num, pts_den, flow_bytes, depth_bytes, mask_bytes, flags) = values[:12]
    confidence_bytes = values[12] if version == VERSION else 0
    if magic != MAGIC or version not in (1, VERSION) or pts_den <= 0 or min(width, height) <= 0:
        raise RuntimeError(f"invalid shared motion packet header: {source}")
    expected_flow = width * height * 2 * np.dtype(np.float32).itemsize
    if flow_bytes != expected_flow:
        raise RuntimeError(f"invalid shared motion flow byte count: {source}")
    expected = header.size + flow_bytes + depth_bytes + mask_bytes + confidence_bytes
    if len(raw) != expected:
        raise RuntimeError(f"shared motion packet size mismatch: {source}")
    cursor = header.size
    flow = np.frombuffer(raw, np.float32, flow_bytes // 4, cursor).reshape(height, width, 2).copy()
    cursor += flow_bytes
    depth = None
    if depth_bytes:
        if depth_bytes != width * height * 4:
            raise RuntimeError(f"invalid shared motion depth byte count: {source}")
        depth = np.frombuffer(raw, np.float32, depth_bytes // 4, cursor).reshape(height, width).copy()
        cursor += depth_bytes
    mask = None
    if mask_bytes:
        if mask_bytes != width * height:
            raise RuntimeError(f"invalid shared motion mask byte count: {source}")
        mask = np.frombuffer(raw, np.uint8, mask_bytes, cursor).reshape(height, width).copy()
        cursor += mask_bytes
    confidence = None
    if confidence_bytes:
        if confidence_bytes != width * height * 4:
            raise RuntimeError(f"invalid shared motion confidence byte count: {source}")
        confidence = np.frombuffer(raw, np.float32, confidence_bytes // 4, cursor).reshape(height, width).copy()
    return SidecarPacket(frame_id, width, height, prev_pts_num, current_pts_num,
                         pts_den, flags, flow, depth, mask, confidence)


def wait_sidecar(directory: str | os.PathLike[str], frame_id: int,
                 *, timeout: float = 30.0) -> tuple[Path, SidecarPacket]:
    target = _packet_path(directory, frame_id)
    deadline = time.monotonic() + timeout
    while not target.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for shared motion packet {frame_id}")
        time.sleep(0.005)
    packet = read_sidecar(target)
    if packet.frame_id != frame_id:
        raise RuntimeError(f"sidecar frame mismatch: {packet.frame_id} != {frame_id}")
    return target, packet
