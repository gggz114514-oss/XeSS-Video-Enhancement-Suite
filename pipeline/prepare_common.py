#!/usr/bin/env python3
"""Common streaming/file preparation driver used by independent SR and FG entrypoints."""

from __future__ import annotations

import argparse
import contextlib
from collections import deque
from dataclasses import dataclass
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import zlib

from motion_core import DepthEstimator, DisFlow, FrameAnalyzer, write_debug
from shm_ring import RingWriter
from stage_timer import StageTimer
from stream_protocol import (Flags, FramePacket, MAGIC, VERSION, PIXEL_RGB24,
                             HEADER as PACKET_HEADER, eos, encode, write_packet)


ROOT = os.path.dirname(os.path.abspath(__file__))


def _part_size(part) -> int:
    """Byte size of a payload section (bytes or C-contiguous numpy array)."""
    size = getattr(part, "nbytes", None)
    return size if size is not None else len(part)

_MIGRATION_NOTICE = ("[prepare] SEA-RAFT has been retired from the mainline; "
                     "this job runs on native Fast DIS instead")
_MIGRATION_PRINTED = False


def resolve_engine(name: str) -> str:
    """Map the retired ``sea-raft`` engine choice onto native Fast DIS."""
    global _MIGRATION_PRINTED
    if name == "sea-raft":
        if not _MIGRATION_PRINTED:
            print(_MIGRATION_NOTICE, file=sys.stderr, flush=True)
            _MIGRATION_PRINTED = True
        return "dis"
    return name


def normalize_legacy_engine(args) -> None:
    """Retire ``sea-raft`` engine choices before analysis.

    Old callers used ``--engine sea-raft --bidirectional``; the retired engine
    must not silently run the expert bidirectional DIS path.  The choice is
    forced to one-way Fast DIS with a single per-process notice.  Expert
    ``--engine dis --bidirectional`` keeps its behavior.
    """
    if args.engine != "sea-raft":
        return
    global _MIGRATION_PRINTED
    if not _MIGRATION_PRINTED:
        extra = (" --bidirectional was forced off: the retired engine is "
                 "one-way Fast DIS" if args.bidirectional else "")
        print(_MIGRATION_NOTICE + extra, file=sys.stderr, flush=True)
        _MIGRATION_PRINTED = True
    args.engine = "dis"
    args.bidirectional = False


def add_common_arguments(parser: argparse.ArgumentParser, *, kind: str) -> None:
    parser.add_argument("--raw", default="", help="rgb24 file input; omit with --stream")
    parser.add_argument("--stream", action="store_true", help="read rgb24 frames from stdin and write packets to stdout")
    parser.add_argument("--shm-name", default="", help="write packets to a named shared-memory ring")
    parser.add_argument("--shm-slots", type=int, default=0)
    parser.add_argument("--shm-slot-size", type=int, default=0)
    parser.add_argument("--in-w", type=int, required=True)
    parser.add_argument("--in-h", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--engine", choices=("dis", "sea-raft"), default="dis",
                        help="'sea-raft' is accepted for old callers and runs DIS")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--depth-model", default="")
    parser.add_argument("--depth-device", default="GPU")
    parser.add_argument("--temporal", type=float, default=0.25)
    parser.add_argument("--consistency", type=float, default=1.5)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--depth-edge", type=float, default=0.04)
    parser.add_argument("--motion-window", type=int, choices=(2, 5), default=2)
    parser.add_argument("--temporal-motion-strength", type=float, default=0.65)
    parser.add_argument("--temporal-depth-strength", type=float, default=0.18)
    parser.add_argument("--responsive-mask", action="store_true")
    parser.add_argument("--responsive-max", type=float, default=0.8)
    parser.add_argument("--mv-path", choices=("highres", "lowres-depth"), default="highres")
    parser.add_argument("--mv-out", default="")
    parser.add_argument("--depth-out", default="")
    parser.add_argument("--mask-out", default="")
    parser.add_argument("--overlay-mask", default="", help="optional static grayscale UI/subtitle mask")
    parser.add_argument("--debug-dir", default="")
    parser.set_defaults(kind=kind)


def validate(args: argparse.Namespace) -> None:
    if args.in_w <= 0 or args.in_h <= 0 or args.frames < 2:
        raise SystemExit("[prepare] invalid dimensions or frame count")
    if args.stream == bool(args.raw):
        raise SystemExit("[prepare] choose exactly one of --stream or --raw")
    if args.shm_name and (not args.stream or args.shm_slots < 2 or args.shm_slot_size <= 0):
        raise SystemExit("[prepare] shared-memory output requires --stream and valid ring dimensions")
    if not 0.0 <= args.temporal <= 0.8:
        raise SystemExit("[prepare] --temporal must be in 0..0.8")
    if args.consistency <= 0 or not 0 <= args.dilate <= 4:
        raise SystemExit("[prepare] invalid consistency/dilation settings")
    if not 0.0 <= args.responsive_max <= 1.0:
        raise SystemExit("[prepare] --responsive-max must be in 0..1")
    if not 0.0 <= args.temporal_motion_strength <= 1.0:
        raise SystemExit("[prepare] --temporal-motion-strength must be in 0..1")
    if not 0.0 <= args.temporal_depth_strength <= 0.5:
        raise SystemExit("[prepare] --temporal-depth-strength must be in 0..0.5")
    if args.kind == "sr" and args.mv_path == "lowres-depth" and not args.depth_model:
        raise SystemExit("[prepare] lowres-depth requires --depth-model")
    if not args.stream and not args.mv_out:
        raise SystemExit("[prepare] file mode requires --mv-out")
    if args.overlay_mask and not os.path.isfile(args.overlay_mask):
        raise SystemExit(f"[prepare] overlay mask not found: {args.overlay_mask}")


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"wanted {size} bytes, received {size - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def frame_iterator(args: argparse.Namespace, timer: StageTimer | None = None):
    frame_bytes = args.in_w * args.in_h * 3
    if args.stream:
        source = sys.stdin.buffer
        for index in range(args.frames):
            with (timer.span("decoder_read") if timer else contextlib.nullcontext()):
                try:
                    data = read_exact(source, frame_bytes)
                except EOFError as exc:
                    raise RuntimeError(f"decoder ended at frame {index}: {exc}") from exc
            yield np.frombuffer(data, np.uint8).reshape(args.in_h, args.in_w, 3).copy()
        return
    available = os.path.getsize(args.raw) // frame_bytes
    if available < args.frames:
        raise RuntimeError(f"raw input has {available} frames, expected {args.frames}")
    mapped = np.memmap(args.raw, dtype=np.uint8, mode="r",
                       shape=(available, args.in_h, args.in_w, 3))
    for index in range(args.frames):
        yield np.array(mapped[index], copy=True)


def create_analyzer(args: argparse.Namespace) -> FrameAnalyzer:
    engine = DisFlow(args.bidirectional)
    depth = DepthEstimator(args.depth_model, args.depth_device) if args.depth_model else None
    return FrameAnalyzer(engine, depth, temporal=args.temporal,
                         consistency=args.consistency, dilation=args.dilate,
                         depth_edge=args.depth_edge, responsive_max=args.responsive_max,
                         photometric_confidence=(args.kind == "fg"))


def ensure_outputs(args: argparse.Namespace) -> None:
    if args.stream:
        return
    for directory in (args.mv_out, args.depth_out, args.mask_out, args.debug_dir):
        if directory:
            os.makedirs(directory, exist_ok=True)


def emit_file(args: argparse.Namespace, index: int, flow: bytes,
              depth: bytes, mask: bytes) -> None:
    Path(args.mv_out, f"mv_{index:06d}.bin").write_bytes(flow)
    if args.depth_out and depth:
        Path(args.depth_out, f"depth_{index:06d}.bin").write_bytes(depth)
    if args.mask_out and mask:
        Path(args.mask_out, f"mask_{index:06d}.bin").write_bytes(mask)


@dataclass
class PendingFrame:
    index: int
    rgb: np.ndarray
    result: object
    flags: Flags


def run_preparer(args: argparse.Namespace) -> None:
    validate(args)
    normalize_legacy_engine(args)
    ensure_outputs(args)
    timer = StageTimer()
    analyzer = create_analyzer(args)
    overlay_mask = None
    if args.overlay_mask:
        overlay_mask = cv2.imread(args.overlay_mask, cv2.IMREAD_GRAYSCALE)
        if overlay_mask is None:
            raise RuntimeError(f"cannot decode overlay mask: {args.overlay_mask}")
        overlay_mask = cv2.resize(overlay_mask, (args.in_w, args.in_h),
                                  interpolation=cv2.INTER_NEAREST)
    ring = (RingWriter(args.shm_name, args.shm_slots, args.shm_slot_size)
            if args.shm_name else None)
    output = ring if ring is not None else (sys.stdout.buffer if args.stream else None)
    scene_cuts: list[int] = []
    started = time.perf_counter()
    pending: deque[PendingFrame] = deque(maxlen=5)

    def emit(entry: PendingFrame, window: list[PendingFrame]) -> None:
        with timer.span("packet_write"):
            result = entry.result
            if args.kind == "fg" and args.motion_window == 5:
                from five_frame_fg import refine_five_frame
                result = refine_five_frame(
                    window, entry.index,
                    motion_strength=args.temporal_motion_strength,
                    depth_strength=args.temporal_depth_strength)
            result.flow[..., 0] = np.clip(result.flow[..., 0], -args.in_w, args.in_w)
            result.flow[..., 1] = np.clip(result.flow[..., 1], -args.in_h, args.in_h)
            with timer.span("packet_encode"):
                color = np.ascontiguousarray(entry.rgb, dtype=np.uint8)
                flow = np.ascontiguousarray(result.flow, dtype=np.float32)
                output_depth = result.depth
                if output_depth is None and args.kind == "fg":
                    output_depth = np.full((args.in_h, args.in_w), 0.5, np.float32)
                depth = (np.ascontiguousarray(output_depth, dtype=np.float32)
                         if output_depth is not None else b"")
                if overlay_mask is not None and args.kind == "fg":
                    mask = np.ascontiguousarray(overlay_mask, dtype=np.uint8)
                else:
                    mask = (np.clip(result.mask * 255.0, 0, 255).astype(np.uint8)
                            if result.mask is not None else b"")
                # 直接构造协议头：CRC 按 color/motion/depth/mask 分块增量计算，
                # 各段以缓冲区直写 transport，整帧不再拼成一个大 bytes。
                parts = [color, flow, depth, mask]
                sizes = [_part_size(part) for part in parts]
                crc = 0
                for part in parts:
                    crc = zlib.crc32(part, crc)
                header = PACKET_HEADER.pack(
                    MAGIC, VERSION, PACKET_HEADER.size, entry.index,
                    args.in_w, args.in_h, PIXEL_RGB24, int(entry.flags),
                    sizes[0], sizes[1], sizes[2], sizes[3], crc & 0xFFFFFFFF)
            if output is not None:
                # write_packet minus the CRC/encode pass so the blocking
                # transport portion (worker_read_wait) stays separable.
                with timer.span("transport_write"):
                    if isinstance(output, RingWriter):
                        output.write_parts(header, parts)
                    else:
                        output.write(header)
                        for part in parts:
                            output.write(part)
                        output.flush()
            else:
                emit_file(args, entry.index, flow.tobytes(),
                          depth.tobytes() if depth is not b"" else b"",
                          mask.tobytes() if mask is not b"" else b"")
            if args.debug_dir:
                write_debug(args.debug_dir, entry.index, result)

    frames_iter = frame_iterator(args, timer)
    index = -1
    while True:
        try:
            rgb = next(frames_iter)
        except StopIteration:
            break
        index += 1
        with timer.span("analyze_total"):
            if index == 0:
                result = analyzer.first(rgb, args.responsive_mask)
                flags = Flags.RESET
            else:
                result = analyzer.next(rgb, args.responsive_mask,
                                       dilate_highres=(args.mv_path == "highres" or args.kind == "fg"))
                flags = Flags.NONE
                if result.scene_cut:
                    scene_cuts.append(index)
                    flags |= Flags.RESET | Flags.SCENE_CUT
        pending.append(PendingFrame(index, rgb.copy(), result, flags))
        if args.kind != "fg" or args.motion_window == 2:
            emit(pending[-1], [pending[-1]])
        elif index >= 2:
            target = index - 2
            emit(next(entry for entry in pending if entry.index == target), list(pending))
        if index and (index % 10 == 0 or index + 1 == args.frames):
            elapsed = time.perf_counter() - started
            reliable = result.metrics.get("reliable_fraction", 1.0)
            print(f"[prepare-{args.kind}] {index + 1}/{args.frames}, "
                  f"reliable={reliable:.1%}, {elapsed / index:.3f}s/pair",
                  file=sys.stderr, flush=True)
    if args.kind == "fg" and args.motion_window == 5:
        for target in range(max(0, args.frames - 2), args.frames):
            emit(next(entry for entry in pending if entry.index == target), list(pending))
    if output is not None:
        write_packet(output, eos(args.frames))
        if ring is not None:
            ring.close()
    else:
        metadata = {"version": "1.2",
                    "kind": args.kind, "width": args.in_w, "height": args.in_h,
                    "frames": args.frames, "engine": args.engine,
                    "bidirectional": args.bidirectional,
                    "motion_direction": "current_to_previous", "motion_units": "pixels",
                    "mv_path": args.mv_path, "depth": bool(args.depth_model),
                    "motion_window": args.motion_window,
                    "responsive_mask": args.responsive_mask, "scene_cuts": scene_cuts}
        for directory in (args.mv_out, args.depth_out, args.mask_out):
            if directory:
                Path(directory, "meta.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(args.mv_out, "reset_frames.txt").write_text(
            "".join(f"{index}\n" for index in scene_cuts), encoding="ascii")
    if ring is not None:
        timer.observe("worker_read_wait", ring.wait_seconds)
    timer.totals["prepare_total"] = time.perf_counter() - started
    timer.report(f"prepare-{args.kind}")
    print(f"[prepare-{args.kind}] complete: {args.frames} frames",
          file=sys.stderr, flush=True)
