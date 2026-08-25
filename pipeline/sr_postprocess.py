#!/usr/bin/env python3
"""Fused SR post-processor: final sharpening + vertical ringing guard, one pass.

Replaces the old adaptive_sharpen -> edge_ringing_guard process pair.  Each
output frame is traversed once through scratch buffers allocated up front; the
intermediate raw round-trip between the former processes is gone.  Fixed
(fast-preset) sharpening skips the motion-map work because static == motion
cancels out of the linear combination, and every remaining operation keeps the
exact float32 evaluation order of the retired chain so results stay
byte-identical.

The ringing-guard guide analysis depends only on the source video, so a
background thread decodes and analyses ahead of the main loop; the cv2/numpy
work releases the GIL and overlaps the upstream wait.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading

import cv2
import numpy as np

from stage_timer import StageTimer


class _SharpenBuffers:
    """Scratch space for the sharpen pass, allocated once."""

    def __init__(self, height: int, width: int):
        f32, u8 = np.float32, np.uint8
        hw = (height, width)
        self.luma = np.empty(hw, f32)
        self.gray_u8 = np.empty(hw, u8)
        self.scratch = np.empty(hw, f32)
        self.noise_inv = np.empty(hw, f32)
        self.strength = np.empty(hw, f32)
        self.frame = np.empty((height, width, 3), f32)
        self.detail = np.empty((height, width, 3), f32)
        self.product = np.empty((height, width, 3), f32)
        self.sharp_u8 = np.empty((height, width, 3), u8)
        self.result = np.empty((height, width, 3), f32)
        self.out_u8 = np.empty((height, width, 3), u8)
        self.guide_scaled = np.empty((height, width, 3), f32)
        self.kernel3 = np.ones((3, 3), u8)
        self.prev_luma = np.empty(hw, f32)
        self.have_prev = False
        self.motion = np.empty(hw, f32)


class _GuideBuffers:
    """Scratch space for one ringing-guard guide analysis (pool-allocated)."""

    def __init__(self, height: int, width: int):
        f32 = np.float32
        hw = (height, width)
        self.guide_u8 = np.empty((height, width, 3), np.uint8)
        self.guide_f = np.empty((height, width, 3), f32)
        self.guide_y = np.empty(hw, f32)
        self.sobel = np.empty(hw, f32)
        self.blend = np.empty(hw, f32)
        self.inv_blend = np.empty(hw, f32)


def read_exact(stream, size: int, buf: bytearray) -> bool:
    view = memoryview(buf)[:size]
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            return False
        offset += count
    return True


def compute_blend(gb: _GuideBuffers, source: np.ndarray, width: int,
                  height: int, strength: float) -> None:
    """Fill gb.blend / gb.inv_blend from the resized guide frame."""
    guide = cv2.resize(source, (width, height),
                       interpolation=cv2.INTER_CUBIC, dst=gb.guide_u8)
    np.copyto(gb.guide_f, guide, casting="unsafe")
    cv2.cvtColor(gb.guide_f, cv2.COLOR_RGB2GRAY, dst=gb.guide_y)
    sobel = cv2.Sobel(gb.guide_y, cv2.CV_32F, 1, 0, dst=gb.sobel, ksize=3)
    np.abs(sobel, out=sobel)
    sobel /= 8.0
    blend = gb.blend
    np.subtract(sobel, 0.5, out=blend)
    np.divide(blend, 4.0, out=blend)
    np.clip(blend, 0.0, 1.0, out=blend)
    cv2.GaussianBlur(blend, (0, 0), 2.5, dst=blend)
    np.multiply(blend, strength, out=blend)
    np.clip(blend, 0.0, 0.90, out=blend)
    np.subtract(1.0, blend, out=gb.inv_blend)


def sharpen_frame(buf: _SharpenBuffers, frame_u8: np.ndarray, mode: str,
                  static: float, motion_level: float) -> np.ndarray:
    """Sharpen through reusable buffers; returns the quantized u8 result."""
    luma = buf.luma
    cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY, dst=buf.gray_u8)
    np.copyto(luma, buf.gray_u8, casting="unsafe")
    if mode == "adaptive":
        motion = buf.motion
        if not buf.have_prev:
            motion[:] = 0.0
            buf.have_prev = True
        else:
            np.subtract(luma, buf.prev_luma, out=motion)
            np.abs(motion, out=motion)
            np.divide(motion, 32.0, out=motion)
            np.clip(motion, 0.0, 1.0, out=motion)
            cv2.GaussianBlur(motion, (5, 5), 0.9, dst=motion)
        np.copyto(buf.prev_luma, luma)
    else:
        motion = None
    cv2.erode(luma, buf.kernel3, dst=buf.scratch)
    local_min = buf.scratch
    local_max = cv2.dilate(luma, buf.kernel3, dst=luma)
    contrast = np.subtract(local_max, local_min, out=buf.noise_inv)
    np.subtract(contrast, 72.0, out=contrast)
    np.divide(contrast, 80.0, out=contrast)
    np.clip(contrast, 0.0, 0.65, out=contrast)
    noise_inv = np.subtract(1.0, contrast, out=buf.noise_inv)
    strength = buf.strength
    if motion is None:
        # static == motion cancels out of the linear combination, so the fast
        # path reduces to static * (1 - noise_guard) without any motion map.
        np.multiply(noise_inv, static, out=strength)
    else:
        np.multiply(motion, motion_level, out=buf.scratch)
        np.subtract(1.0, motion, out=motion)
        np.multiply(motion, static, out=motion)
        np.add(motion, buf.scratch, out=strength)
        np.multiply(strength, noise_inv, out=strength)
    frame = buf.frame
    np.copyto(frame, frame_u8, casting="unsafe")
    detail = buf.detail
    cv2.GaussianBlur(frame, (0, 0), 0.8, dst=detail)
    np.subtract(frame, detail, out=detail)
    np.clip(detail, -24.0, 24.0, out=detail)
    np.multiply(strength, 1.65, out=strength)
    product = buf.product
    np.multiply(detail, strength[:, :, None], out=product)
    np.add(frame, product, out=product)
    np.clip(product, 0.0, 255.0, out=product)
    np.copyto(buf.sharp_u8, product, casting="unsafe")
    return buf.sharp_u8


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-pass SR sharpening and ringing guard")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--sharpen-mode", choices=("off", "fixed", "adaptive"), default="fixed")
    parser.add_argument("--static", type=float, default=0.35)
    parser.add_argument("--motion", type=float, default=0.18)
    parser.add_argument("--guard-strength", type=float, default=0.0,
                        help="suppress XeSS vertical edge ringing; 0 disables it")
    parser.add_argument("--video", default="", help="source video for the guard guide frames")
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--in-w", type=int, default=0)
    parser.add_argument("--in-h", type=int, default=0)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.frames <= 0:
        raise SystemExit("[post] invalid dimensions/frame count")
    if not 0.0 <= args.motion <= 1.0 or not 0.0 <= args.static <= 1.0:
        raise SystemExit("[post] strengths must be in 0..1")
    guard = args.guard_strength > 0
    if guard:
        if not args.video or not args.ffmpeg or min(args.in_w, args.in_h) <= 0:
            parser.error("guard requires --video/--ffmpeg/--in-w/--in-h")
        if not 0.0 <= args.guard_strength <= 1.0:
            parser.error("--guard-strength must be in 0..1")

    decoder = None
    if guard:
        command = [
            args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", args.video, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{args.in_w}x{args.in_h}", "-vframes", str(args.frames), "-",
        ]
        decoder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_bytes = args.width * args.height * 3
    source_bytes = args.in_w * args.in_h * 3
    timer = StageTimer()
    buf = _SharpenBuffers(args.height, args.width)
    inbuf = bytearray(frame_bytes)

    guide_pool = queue.Queue()
    guide_ready = queue.Queue(maxsize=2)
    if guard:
        for _ in range(3):
            guide_pool.put(_GuideBuffers(args.height, args.width))
        sourcebuf = bytearray(source_bytes)

        def produce_guides() -> None:
            try:
                for _ in range(args.frames):
                    gb = guide_pool.get()
                    if not read_exact(decoder.stdout, source_bytes, sourcebuf):
                        raise RuntimeError(f"source decoder ended early")
                    source = np.frombuffer(sourcebuf, np.uint8).reshape(
                        args.in_h, args.in_w, 3)
                    compute_blend(gb, source, args.width, args.height,
                                  args.guard_strength)
                    guide_ready.put(gb)
                guide_ready.put(None)
            except BaseException as exc:
                guide_ready.put(exc)

        producer = threading.Thread(target=produce_guides, daemon=True)
        producer.start()

    def next_guide():
        gb = guide_ready.get()
        if isinstance(gb, BaseException):
            raise gb
        return gb

    try:
        for index in range(args.frames):
            with timer.span("upstream_read"):
                if not read_exact(sys.stdin.buffer, frame_bytes, inbuf):
                    raise SystemExit(f"[post] input ended at frame {index}")
                frame_u8 = np.frombuffer(inbuf, np.uint8).reshape(
                    args.height, args.width, 3)
            if guard:
                gb = next_guide()
            with timer.span("sharpen"):
                if args.sharpen_mode == "off":
                    np.copyto(buf.sharp_u8, frame_u8)
                else:
                    sharpen_frame(buf, frame_u8, args.sharpen_mode,
                                  args.static, args.motion)
            with timer.span("guard_blend"):
                if guard:
                    blend, inv_blend = gb.blend, gb.inv_blend
                    result = buf.result
                    np.copyto(result, buf.sharp_u8, casting="unsafe")
                    np.multiply(result, inv_blend[:, :, None], out=result)
                    np.multiply(gb.guide_f, blend[:, :, None], out=buf.guide_scaled)
                    np.add(result, buf.guide_scaled, out=result)
                    np.clip(result, 0.0, 255.0, out=result)
                    np.copyto(buf.out_u8, result, casting="unsafe")
                    guide_pool.put(gb)
                else:
                    np.copyto(buf.out_u8, buf.sharp_u8)
                sys.stdout.buffer.write(buf.out_u8.data)
            if index and index % 60 == 0:
                print(f"[post] {index}/{args.frames}", file=sys.stderr, flush=True)
        sys.stdout.buffer.flush()
        if decoder is not None and decoder.wait() != 0:
            raise RuntimeError("source decoder failed")
    except BaseException:
        if decoder is not None and decoder.poll() is None:
            decoder.terminate()
        if decoder is not None:
            decoder.wait()
        raise
    finally:
        if decoder is not None:
            decoder.stdout.close()
    timer.report("sr-post")


if __name__ == "__main__":
    main()
