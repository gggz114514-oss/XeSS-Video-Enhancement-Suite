#!/usr/bin/env python3
"""Bounded-memory five-frame motion-compensated RGB24 fusion.

The filter keeps two past and two future frames, aligns every neighbour to the
current frame with DIS optical flow, then applies a conservative robust blend.
It is intended to run before SR so the extra optical-flow work stays at source
resolution and downstream motion/depth analysis sees the same fused image.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np


BASE_WEIGHTS = {1: 0.34, 2: 0.17}


def read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            break
        chunks.extend(block)
    return bytes(chunks)


class FiveFrameFusion:
    def __init__(self, width: int, height: int, strength: float):
        self.width = width
        self.height = height
        self.strength = strength
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                     np.arange(height, dtype=np.float32))
        self.grid_x = grid_x
        self.grid_y = grid_y

    def fuse(self, center_index: int, frames: OrderedDict[int, np.ndarray]) -> np.ndarray:
        center_u8 = frames[center_index]
        if self.strength <= 0:
            return center_u8
        center = center_u8.astype(np.float32)
        center_gray = cv2.cvtColor(center_u8, cv2.COLOR_RGB2GRAY)
        total = np.ones((self.height, self.width, 1), dtype=np.float32)
        result = center.copy()
        for neighbour_index, neighbour_u8 in frames.items():
            distance = abs(neighbour_index - center_index)
            if not distance or distance > 2:
                continue
            neighbour_gray = cv2.cvtColor(neighbour_u8, cv2.COLOR_RGB2GRAY)
            flow = self.dis.calc(center_gray, neighbour_gray, None)
            map_x = self.grid_x + flow[..., 0]
            map_y = self.grid_y + flow[..., 1]
            aligned = cv2.remap(neighbour_u8, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT101).astype(np.float32)
            aligned_gray = cv2.cvtColor(aligned.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            error = np.abs(aligned_gray.astype(np.float32) - center_gray.astype(np.float32))
            # Large post-warp changes are normally cuts, occlusions, or bad flow.
            if float(np.mean(error)) > 30.0:
                continue
            confidence = np.exp(-error / 11.0)
            confidence *= np.clip((26.0 - error) / 18.0, 0.0, 1.0)
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            confidence *= np.clip((96.0 - magnitude) / 64.0, 0.0, 1.0)
            weight = (self.strength * BASE_WEIGHTS[distance] * confidence)[..., None]
            # A robust colour clamp prevents one misaligned neighbour from drawing
            # a bright/dark duplicate across a silhouette.
            aligned = np.clip(aligned, center - 24.0, center + 24.0)
            result += aligned * weight
            total += weight
        return np.clip(result / total + 0.5, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-frame motion-compensated RGB24 fusion")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--strength", type=float, default=0.35)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.frames <= 0:
        raise SystemExit("[fusion] invalid dimensions or frame count")
    if not 0.0 <= args.strength <= 1.0:
        raise SystemExit("[fusion] --strength must be in 0..1")

    frame_bytes = args.width * args.height * 3
    fusion = FiveFrameFusion(args.width, args.height, args.strength)
    buffered: OrderedDict[int, np.ndarray] = OrderedDict()
    next_output = 0
    written = 0
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    try:
        for index in range(args.frames):
            payload = read_exact(stdin, frame_bytes)
            if len(payload) != frame_bytes:
                raise RuntimeError(f"short input at frame {index}: {len(payload)}/{frame_bytes} bytes")
            buffered[index] = np.frombuffer(payload, dtype=np.uint8).reshape(
                args.height, args.width, 3).copy()
            while next_output <= index - 2:
                stdout.write(fusion.fuse(next_output, buffered).tobytes())
                written += 1
                next_output += 1
                oldest_needed = next_output - 2
                while buffered and next(iter(buffered)) < oldest_needed:
                    buffered.popitem(last=False)
                if written % 24 == 0 or written == args.frames:
                    print(f"[fusion] {written}/{args.frames}", file=sys.stderr, flush=True)
        while next_output < args.frames:
            stdout.write(fusion.fuse(next_output, buffered).tobytes())
            written += 1
            next_output += 1
        stdout.flush()
    except BrokenPipeError:
        os._exit(1)
    except Exception as exc:
        print(f"[fusion] error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    print(f"[fusion] complete: {written} frames, strength={args.strength:g}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
