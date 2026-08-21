#!/usr/bin/env python3
"""Cheap release-gate validation for encoded XeSS media outputs."""

from __future__ import annotations

import json
import os
import subprocess


class MediaValidationError(RuntimeError):
    pass


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, errors="replace")


def probe_video(python: str, flow_script: str, path: str) -> dict[str, int | float]:
    result = _run([python, flow_script, path, "--probe-only"])
    if result.returncode:
        raise MediaValidationError(
            f"unable to probe encoded output {path}: {result.stderr[-1200:]}"
        )
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaValidationError(f"invalid probe response for {path}: {result.stdout!r}") from exc
    return metadata


def has_audio(ffmpeg: str, path: str) -> bool:
    result = _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path,
        "-map", "0:a:0", "-frames:a", "1", "-f", "null", "-",
    ])
    return result.returncode == 0


def decodes_first_frame(ffmpeg: str, path: str) -> None:
    result = _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path,
        "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-",
    ])
    if result.returncode:
        raise MediaValidationError(
            f"encoded output cannot decode its first video frame: {result.stderr[-1200:]}"
        )


def validate_output(
    *, python: str, flow_script: str, ffmpeg: str, output: str,
    source: str, expected_width: int, expected_height: int,
    expected_fps: float, expected_frames: int,
) -> dict[str, int | float | bool]:
    if not os.path.isfile(output) or os.path.getsize(output) == 0:
        raise MediaValidationError(f"encoded output is missing or empty: {output}")
    decodes_first_frame(ffmpeg, output)
    metadata = probe_video(python, flow_script, output)
    actual = (int(metadata["width"]), int(metadata["height"]))
    expected = (expected_width, expected_height)
    if actual != expected:
        raise MediaValidationError(f"output resolution mismatch: expected {expected}, got {actual}")
    actual_frames = int(metadata["frames"])
    if actual_frames != expected_frames:
        raise MediaValidationError(
            f"output frame-count mismatch: expected {expected_frames}, got {actual_frames}"
        )
    actual_fps = float(metadata["fps"])
    fps_tolerance = max(0.01, expected_fps * 0.001)
    if abs(actual_fps - expected_fps) > fps_tolerance:
        raise MediaValidationError(
            f"output fps mismatch: expected {expected_fps:g}, got {actual_fps:g}"
        )
    source_audio = has_audio(ffmpeg, source)
    output_audio = has_audio(ffmpeg, output)
    if source_audio and not output_audio:
        raise MediaValidationError("source has audio but encoded output does not")
    return {"width": actual[0], "height": actual[1], "fps": actual_fps,
            "frames": actual_frames, "audio": output_audio}
