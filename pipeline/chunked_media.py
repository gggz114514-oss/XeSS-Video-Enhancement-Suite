#!/usr/bin/env python3
"""Exact-frame compressed chunk helpers; never create whole-video RGB raw files."""

from __future__ import annotations

import os
from pathlib import Path


def extract_lossless_command(ffmpeg: str, source: str, destination: str,
                             start_frame: int, frame_count: int,
                             fps: float) -> list[str]:
    """Extract an exact CFR FFV1/AVI chunk with reliable frame metadata.

    FFV1/Matroska does not expose a stored frame count and OpenCV derives N-1
    for short chunks from the final timestamp. AVI stores the exact frame
    count; an explicit CFR also prevents short tail chunks from acquiring a
    rounded, filename-changing frame rate.
    """
    end_frame = start_frame + frame_count
    video_filter = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    return [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", source,
        "-map", "0:v:0", "-an", "-vf", video_filter,
        "-frames:v", str(frame_count), "-r", f"{fps:.12g}", "-fps_mode", "cfr",
        "-c:v", "ffv1", "-level", "3", "-g", "1", destination,
    ]


def write_concat_list(path: str | os.PathLike[str], segments: list[str | os.PathLike[str]]) -> None:
    lines = []
    for segment in segments:
        value = Path(segment).resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{value}'")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_command(ffmpeg: str, list_path: str, audio_source: str,
                   destination: str, expected_frames: int, output_fps: float) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "concat", "-safe", "0", "-i", list_path, "-i", audio_source,
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy",
        "-t", f"{expected_frames / output_fps:.9f}",
        "-frames:v", str(expected_frames), destination,
    ]
