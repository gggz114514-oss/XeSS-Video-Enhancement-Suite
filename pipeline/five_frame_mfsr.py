#!/usr/bin/env python3
"""Streaming five-frame multi-frame SR residual injection for XeSS RGB24 output.

Neighbouring source frames are projected directly onto the output-resolution
grid using sub-pixel optical-flow coordinates.  Only coherent, photometrically
reliable luma detail is injected into the already-upscaled XeSS frame.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np


TEMPORAL_WEIGHTS = {1: 0.80, 2: 0.50}


def read_exact(stream, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            return None
        chunks.extend(block)
    return bytes(chunks)


class MultiFrameReconstructor:
    def __init__(self, in_w: int, in_h: int, out_w: int, out_h: int,
                 strength: float, detail_boost: float, max_injection: float):
        self.in_w = in_w
        self.in_h = in_h
        self.out_w = out_w
        self.out_h = out_h
        self.strength = strength
        self.detail_boost = detail_boost
        self.max_injection = max_injection
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        lr_x, lr_y = np.meshgrid(np.arange(in_w, dtype=np.float32),
                                 np.arange(in_h, dtype=np.float32))
        self.lr_x = lr_x
        self.lr_y = lr_y
        hr_x, hr_y = np.meshgrid(np.arange(out_w, dtype=np.float32),
                                 np.arange(out_h, dtype=np.float32))
        self.hr_to_lr_x = (hr_x + 0.5) * (in_w / out_w) - 0.5
        self.hr_to_lr_y = (hr_y + 0.5) * (in_h / out_h) - 0.5

    def reconstruct(self, center_index: int, sources: OrderedDict[int, np.ndarray],
                    xess_u8: np.ndarray) -> np.ndarray:
        center_u8 = sources[center_index]
        center_gray = cv2.cvtColor(center_u8, cv2.COLOR_RGB2GRAY)
        base_hr = cv2.resize(center_u8, (self.out_w, self.out_h),
                             interpolation=cv2.INTER_CUBIC).astype(np.float32)
        weighted = base_hr.copy()
        squared = base_hr * base_hr
        weight_sum = np.ones((self.out_h, self.out_w, 1), np.float32)

        for neighbour_index, neighbour_u8 in sources.items():
            distance = abs(neighbour_index - center_index)
            if not distance or distance > 2:
                continue
            neighbour_gray = cv2.cvtColor(neighbour_u8, cv2.COLOR_RGB2GRAY)
            flow = self.dis.calc(center_gray, neighbour_gray, None)
            flow = np.nan_to_num(flow.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            aligned_lr = cv2.remap(neighbour_u8, self.lr_x + flow[..., 0],
                                   self.lr_y + flow[..., 1], cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT101)
            aligned_gray = cv2.cvtColor(aligned_lr, cv2.COLOR_RGB2GRAY)
            error = np.abs(aligned_gray.astype(np.float32) - center_gray.astype(np.float32))
            if float(np.mean(error)) > 30.0:
                continue
            confidence = np.exp(-error / 10.0)
            confidence *= np.clip((25.0 - error) / 17.0, 0.0, 1.0)
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            confidence *= np.clip((80.0 - magnitude) / 56.0, 0.0, 1.0)
            valid = ((self.lr_x + flow[..., 0] >= 0.0) &
                     (self.lr_x + flow[..., 0] <= self.in_w - 1.0) &
                     (self.lr_y + flow[..., 1] >= 0.0) &
                     (self.lr_y + flow[..., 1] <= self.in_h - 1.0))
            confidence *= valid.astype(np.float32)

            flow_hr = cv2.resize(flow, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
            aligned_hr = cv2.remap(
                neighbour_u8,
                self.hr_to_lr_x + flow_hr[..., 0],
                self.hr_to_lr_y + flow_hr[..., 1],
                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101,
            ).astype(np.float32)
            confidence_hr = cv2.resize(confidence, (self.out_w, self.out_h),
                                       interpolation=cv2.INTER_LINEAR)
            weight = (TEMPORAL_WEIGHTS[distance] * confidence_hr)[..., None]
            weighted += aligned_hr * weight
            squared += aligned_hr * aligned_hr * weight
            weight_sum += weight

        reconstruction = weighted / np.maximum(weight_sum, 1e-6)
        variance = np.maximum(squared / np.maximum(weight_sum, 1e-6) - reconstruction ** 2, 0.0)
        coherence = np.exp(-np.mean(variance, axis=2) / 150.0)
        support = np.clip((weight_sum[..., 0] - 1.0) / 1.20, 0.0, 1.0)
        confidence_hr = cv2.GaussianBlur(support * coherence, (0, 0), 0.65)

        reconstruction_y = cv2.cvtColor(np.clip(reconstruction, 0, 255).astype(np.uint8),
                                         cv2.COLOR_RGB2GRAY).astype(np.float32)
        base_y = cv2.cvtColor(base_hr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        correction = reconstruction_y - base_y
        subpixel_residual = correction - cv2.GaussianBlur(correction, (0, 0), 1.35)
        stable_detail = reconstruction_y - cv2.GaussianBlur(reconstruction_y, (0, 0), 0.75)
        signal_guard = np.clip((reconstruction_y - 7.0) / 18.0, 0.0, 1.0)
        injection = confidence_hr * signal_guard * (
            self.strength * np.clip(subpixel_residual, -10.0, 10.0) +
            self.detail_boost * np.clip(stable_detail, -14.0, 14.0)
        )
        injection = np.clip(injection, -self.max_injection, self.max_injection)
        refined = xess_u8.astype(np.float32) + injection[..., None]
        return np.clip(refined + 0.5, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-frame sub-pixel MFSR residual injector")
    parser.add_argument("--video", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--in-w", type=int, required=True)
    parser.add_argument("--in-h", type=int, required=True)
    parser.add_argument("--out-w", type=int, required=True)
    parser.add_argument("--out-h", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--strength", type=float, default=1.80)
    parser.add_argument("--detail-boost", type=float, default=0.90)
    parser.add_argument("--max-injection", type=float, default=22.0)
    args = parser.parse_args()
    if min(args.in_w, args.in_h, args.out_w, args.out_h, args.frames) <= 0:
        raise SystemExit("[mfsr] invalid dimensions/frame count")
    if not 0.0 <= args.strength <= 8.0 or not 0.0 <= args.detail_boost <= 4.0:
        raise SystemExit("[mfsr] invalid strength/detail boost")
    if not 0.0 <= args.max_injection <= 128.0:
        raise SystemExit("[mfsr] --max-injection must be in 0..128")

    source_command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                      "-i", args.video, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24",
                      "-s", f"{args.in_w}x{args.in_h}", "-vframes", str(args.frames), "-"]
    decoder = subprocess.Popen(source_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert decoder.stdout is not None
    source_bytes = args.in_w * args.in_h * 3
    output_bytes = args.out_w * args.out_h * 3
    sources: OrderedDict[int, np.ndarray] = OrderedDict()
    next_source = 0
    reconstructor = MultiFrameReconstructor(
        args.in_w, args.in_h, args.out_w, args.out_h,
        args.strength, args.detail_boost, args.max_injection,
    )
    started = time.perf_counter()
    try:
        for index in range(args.frames):
            wanted = min(args.frames - 1, index + 2)
            while next_source <= wanted:
                payload = read_exact(decoder.stdout, source_bytes)
                if payload is None:
                    raise RuntimeError(f"source decoder ended at frame {next_source}")
                sources[next_source] = np.frombuffer(payload, np.uint8).reshape(
                    args.in_h, args.in_w, 3).copy()
                next_source += 1
            xess_payload = read_exact(sys.stdin.buffer, output_bytes)
            if xess_payload is None:
                raise RuntimeError(f"XeSS stream ended at frame {index}")
            xess_frame = np.frombuffer(xess_payload, np.uint8).reshape(
                args.out_h, args.out_w, 3)
            refined = reconstructor.reconstruct(index, sources, xess_frame)
            sys.stdout.buffer.write(refined.tobytes())
            oldest_needed = index - 1
            while sources and next(iter(sources)) < oldest_needed:
                sources.popitem(last=False)
            if (index + 1) % 24 == 0:
                elapsed = time.perf_counter() - started
                print(f"[mfsr] {index + 1}/{args.frames}, {elapsed / (index + 1):.3f}s/frame",
                      file=sys.stderr, flush=True)
        sys.stdout.buffer.flush()
        decoder.stdout.close()
        code = decoder.wait(timeout=10)
        if code:
            raise RuntimeError(f"source decoder failed with exit code {code}")
    except BrokenPipeError:
        decoder.terminate()
        os._exit(1)
    except Exception as exc:
        if decoder.poll() is None:
            decoder.terminate()
        print(f"[mfsr] error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    print(f"[mfsr] complete: {args.frames} frames, {(time.perf_counter() - started):.2f}s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
