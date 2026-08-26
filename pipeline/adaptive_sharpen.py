#!/usr/bin/env python3
"""Bounded streaming adaptive sharpener for final RGB24 output."""

import argparse
import sys

import cv2
import numpy as np

from stage_timer import StageTimer


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser(description="Adaptive final-stage RGB24 sharpener")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--static", type=float, default=0.35)
    parser.add_argument("--motion", type=float, default=0.18)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.frames <= 0:
        raise SystemExit("[sharpen] invalid dimensions/frame count")
    if not 0.0 <= args.motion <= 1.0 or not 0.0 <= args.static <= 1.0:
        raise SystemExit("[sharpen] strengths must be in 0..1")
    frame_bytes = args.width * args.height * 3
    source, output = sys.stdin.buffer, sys.stdout.buffer
    timer = StageTimer()
    previous_luma = None
    for index in range(args.frames):
        with timer.span("input_wait"):
            data = read_exact(source, frame_bytes)
        if data is None:
            raise SystemExit(f"[sharpen] input ended at frame {index}")
        with timer.span("sharpen"):
            frame_u8 = np.frombuffer(data, np.uint8).reshape(args.height, args.width, 3)
            frame = frame_u8.astype(np.float32)
            luma = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if previous_luma is None:
                motion = np.zeros_like(luma)
            else:
                motion = np.clip(np.abs(luma - previous_luma) / 32.0, 0.0, 1.0)
                motion = cv2.GaussianBlur(motion, (5, 5), 0.9)
            local_min = cv2.erode(luma, np.ones((3, 3), np.uint8))
            local_max = cv2.dilate(luma, np.ones((3, 3), np.uint8))
            contrast = local_max - local_min
            noise_guard = np.clip((contrast - 72.0) / 80.0, 0.0, 0.65)
            strength = args.static * (1.0 - motion) + args.motion * motion
            strength *= 1.0 - noise_guard
            blurred = cv2.GaussianBlur(frame, (0, 0), 0.8)
            detail = np.clip(frame - blurred, -24.0, 24.0)
            sharpened = np.clip(frame + detail * (strength[..., None] * 1.65), 0.0, 255.0)
        with timer.span("output_write"):
            output.write(sharpened.astype(np.uint8).tobytes())
        previous_luma = luma
        if index and index % 60 == 0:
            print(f"[sharpen] {index}/{args.frames}", file=sys.stderr, flush=True)
    output.flush()
    timer.report("sharpen")


if __name__ == "__main__":
    main()
