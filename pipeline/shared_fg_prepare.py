#!/usr/bin/env python3
"""Bridge SR RGB output to XeFG using the source-resolution MotionPackets.

The bridge never runs DIS or depth. It waits for the bounded sidecar emitted by
the single SR preparer, adapts MV/depth/mask to the SR output grid, and writes
the normal FramePacket ABI expected by ``xess-fg``.
"""

from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np

from motion_core import MotionResult, adapt_motion
from shared_motion_io import wait_sidecar
from stream_protocol import Flags, FramePacket, eos, write_packet


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"SR output ended early: wanted {size}, got {size - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _adapt_plane(plane: np.ndarray | None, width: int, height: int,
                 interpolation: int):
    if plane is None:
        return None
    return np.ascontiguousarray(cv2.resize(plane, (width, height), interpolation=interpolation))


def run(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.frames < 2:
        raise SystemExit("[shared-fg] invalid output dimensions or frame count")
    if args.timeout <= 0:
        raise SystemExit("[shared-fg] timeout must be positive")
    sidecar_dir = Path(args.motion_dir)
    if not sidecar_dir.is_dir():
        raise SystemExit(f"[shared-fg] motion sidecar directory is missing: {sidecar_dir}")
    frame_bytes = args.width * args.height * 3
    created = []
    pending = deque(maxlen=5)
    motion_window = getattr(args, "motion_window", 2)
    if motion_window not in (2, 5):
        raise ValueError("shared motion window must be 2 or 5")
    counts = {"backend": "cpu-dis", "source_pair_analysis_count": 0,
              "directional_dispatch_count": 0, "sr_consume_count": 0,
              "fg_consume_count": 0, "sr_color_frames_received": 0,
              "fg_packets_emitted": 0, "sidecar_bytes_read": 0,
              "cpu_rgb_bytes_read": 0, "mask_tagged": False,
              "motion_window_applied": motion_window,
              "lookahead_frames": 2 if motion_window == 5 else 0,
              "confidence_from_source": True, "source_pts": [],
              "consumer_counts_stage": "SR output received / FG packet delivered; native completion checked by driver"}

    def emit(entry):
        result = entry.result
        if motion_window == 5:
            from five_frame_fg import refine_five_frame
            result = refine_five_frame(list(pending), entry.index,
                                      motion_strength=args.temporal_motion_strength,
                                      depth_strength=args.temporal_depth_strength)
        write_packet(sys.stdout.buffer, FramePacket(
            index=entry.index, width=args.width, height=args.height,
            flags=Flags(entry.flags), color=entry.color,
            motion=result.flow.tobytes(),
            depth=(result.depth.astype(np.float32, copy=False).tobytes()
                   if result.depth is not None else b""), mask=b""))
        counts["fg_packets_emitted"] += 1
        counts["fg_consume_count"] += int(entry.index > 0)

    try:
        previous_pts = None
        for index in range(args.frames):
            sidecar_path, packet = wait_sidecar(sidecar_dir, index, timeout=args.timeout)
            created.append(sidecar_path)
            color = read_exact(sys.stdin.buffer, frame_bytes)
            current_pts = Fraction(packet.current_pts_num, packet.pts_den)
            if previous_pts is not None and (current_pts <= previous_pts or
                    Fraction(packet.prev_pts_num, packet.pts_den) != previous_pts):
                raise RuntimeError(f"shared packet {index} has non-adjacent PTS")
            previous_pts = current_pts
            counts["source_pts"].append([packet.current_pts_num, packet.pts_den])
            counts["sr_color_frames_received"] += 1
            counts["sr_consume_count"] += int(index > 0)
            counts["sidecar_bytes_read"] += sidecar_path.stat().st_size
            counts["cpu_rgb_bytes_read"] += len(color)
            if packet.width <= 0 or packet.height <= 0:
                raise RuntimeError(f"invalid source dimensions in sidecar {sidecar_path}")
            motion = adapt_motion(packet.flow, (packet.width, packet.height),
                                  (args.width, args.height))
            depth = _adapt_plane(packet.depth, args.width, args.height, cv2.INTER_LINEAR)
            if depth is None:
                # Explicit constant-depth option retains standalone semantics.
                depth = np.full((args.height, args.width), 0.5, np.float32)
            # Responsive masks are an SR-side diagnostic.  XeFG's public
            # stream ABI has no responsive-mask resource; only an explicit UI
            # mask route may populate this field, and that route is not part of
            # the shared CPU gate.
            confidence = _adapt_plane(packet.confidence, args.width, args.height, cv2.INTER_LINEAR)
            if confidence is None:
                counts["confidence_from_source"] = False
                if motion_window == 5:
                    raise RuntimeError("five-frame shared FG requires source confidence (sidecar v2)")
            result = MotionResult(motion, depth, None, confidence,
                                  bool(Flags(packet.flags) & Flags.SCENE_CUT), {})
            pending.append(SimpleNamespace(index=index, result=result, color=color, flags=packet.flags))
            if motion_window == 2:
                emit(pending[-1])
            elif index >= 2:
                emit(next(entry for entry in pending if entry.index == index - 2))
            sidecar_path.unlink(missing_ok=True)
            created.pop()
        if motion_window == 5:
            for index in range(max(0, args.frames - 2), args.frames):
                emit(next(entry for entry in pending if entry.index == index))
        write_packet(sys.stdout.buffer, eos(args.frames))
        report_path = Path(args.report) if getattr(args, "report", "") else sidecar_dir / "consumer_counters.json"
        report_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
        print(f"[shared-motion-consumers] {json.dumps(counts)}", file=sys.stderr, flush=True)
    finally:
        # If downstream FG fails, leave no stale sidecar that could be consumed
        # by a later run in a kept workspace.
        for path in created:
            path.unlink(missing_ok=True)
        for path in sidecar_dir.glob("packet_*.smot"):
            path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="shared source MV bridge for XeFG")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--motion-dir", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--motion-window", type=int, choices=(2, 5), default=2)
    parser.add_argument("--temporal-motion-strength", type=float, default=.65)
    parser.add_argument("--temporal-depth-strength", type=float, default=.18)
    parser.add_argument("--report", default="")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
