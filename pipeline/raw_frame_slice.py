#!/usr/bin/env python3
"""Select a bounded frame range from a raw stream while draining all input.

Chunked five-frame processing needs future context that must reach the motion
analyser but must not appear in the encoded segment.  Keeping stdout open while
discarding context frames prevents upstream workers from failing with a broken
pipe and does not create an intermediate raw file.
"""

from __future__ import annotations

import argparse
import sys


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError(f"raw stream ended early: wanted {size}, got {size - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="draining raw-frame range selector")
    parser.add_argument("--frame-bytes", type=int, required=True)
    parser.add_argument("--total-frames", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if (args.frame_bytes <= 0 or args.total_frames <= 0 or args.start < 0 or
            args.count <= 0 or args.start + args.count > args.total_frames):
        raise SystemExit("[raw-slice] invalid frame range")
    source, output = sys.stdin.buffer, sys.stdout.buffer
    end = args.start + args.count
    written = 0
    for index in range(args.total_frames):
        frame = read_exact(source, args.frame_bytes)
        if args.start <= index < end:
            output.write(frame)
            written += 1
    output.flush()
    if written != args.count:
        raise RuntimeError(f"selected {written} frames, expected {args.count}")


if __name__ == "__main__":
    main()
