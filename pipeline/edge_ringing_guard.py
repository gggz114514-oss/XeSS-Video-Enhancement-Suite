from __future__ import annotations

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

from stage_timer import StageTimer


def read_exact(stream, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError(f"stream ended at {offset}/{size} bytes")
        offset += count
    return data


def suppress_vertical_ringing(source: np.ndarray, output: np.ndarray,
                              strength: float) -> np.ndarray:
    height, width = output.shape[:2]
    guide = cv2.resize(source, (width, height), interpolation=cv2.INTER_CUBIC)
    guide_f = guide.astype(np.float32)
    guide_y = cv2.cvtColor(guide_f, cv2.COLOR_RGB2GRAY)
    vertical_edge = np.abs(cv2.Sobel(guide_y, cv2.CV_32F, 1, 0, ksize=3)) / 8.0
    edge_mask = np.clip((vertical_edge - 0.5) / 4.0, 0.0, 1.0)
    edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), 2.5)
    blend = np.clip(edge_mask * strength, 0.0, 0.90)[..., None]
    output_f = output.astype(np.float32)
    return np.clip(output_f * (1.0 - blend) + guide_f * blend,
                   0.0, 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming XeSS vertical ringing guard")
    parser.add_argument("--video", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--in-w", type=int, required=True)
    parser.add_argument("--in-h", type=int, required=True)
    parser.add_argument("--out-w", type=int, required=True)
    parser.add_argument("--out-h", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--strength", type=float, default=0.75)
    args = parser.parse_args()
    if not os.path.isfile(args.video) or not os.path.isfile(args.ffmpeg):
        parser.error("video or ffmpeg not found")
    if min(args.in_w, args.in_h, args.out_w, args.out_h, args.frames) <= 0:
        parser.error("dimensions and frame count must be positive")
    if not 0.0 <= args.strength <= 1.0:
        parser.error("--strength must be in 0..1")

    command = [
        args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", args.video, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.in_w}x{args.in_h}", "-vframes", str(args.frames), "-",
    ]
    decoder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert decoder.stdout is not None
    input_size = args.in_w * args.in_h * 3
    output_size = args.out_w * args.out_h * 3
    timer = StageTimer()
    try:
        for _ in range(args.frames):
            with timer.span("source_decode"):
                source = np.frombuffer(read_exact(decoder.stdout, input_size), np.uint8)
                source = source.reshape(args.in_h, args.in_w, 3)
            with timer.span("upstream_read"):
                output = np.frombuffer(read_exact(sys.stdin.buffer, output_size), np.uint8)
                output = output.reshape(args.out_h, args.out_w, 3)
            with timer.span("edge_guard"):
                guarded = suppress_vertical_ringing(source, output, args.strength)
            with timer.span("encoder_write_wait"):
                sys.stdout.buffer.write(guarded.tobytes())
        sys.stdout.buffer.flush()
        if decoder.wait() != 0:
            raise RuntimeError("source decoder failed")
    except BaseException:
        if decoder.poll() is None:
            decoder.terminate()
        decoder.wait()
        raise
    finally:
        decoder.stdout.close()
    timer.report("edge_guard")


if __name__ == "__main__":
    main()
