#!/usr/bin/env python3
"""Product Fast Pro oneVPL AI VPP media path.

The executable in :mod:`src/realtime/vpl_ai_vpp.cpp` consumes a bounded NV12
stdio stream.  This module owns the media contract around it: ffmpeg decodes
directly to NV12, one VPL process runs the requested AI VPP, and a single
    explicitly selected encoder writes the result while source audio/metadata are remuxed. No
full-size raw file or RGB24 intermediate is created.

The wrapper intentionally refuses unsupported HDR/P010 and non-CFR input.
A normal scaler or frame duplicator must never be silently
substituted for an Intel AI VPP backend.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from fastpro_geometry import plan_sr_route
from media_contract import (expected_output_frames as contract_output_frames,
                            expected_output_rate as contract_output_rate)


class FastProError(RuntimeError):
    """A closed, user-facing Fast Pro failure."""


SUPPORTED_RATES = {
    "23.976": Fraction(24000, 1001),
    "24": Fraction(24, 1),
    "29.97": Fraction(30000, 1001),
    "30": Fraction(30, 1),
    "59.94": Fraction(60000, 1001),
    "60": Fraction(60, 1),
}

# These are the measured B580 AI-SR boundaries for this independent Fast Pro
# candidate.  Keep them separate from FI: the oneVPL FI matrix does not inherit
# SR's input-height or scale restrictions without its own evidence.
FAST_PRO_SR_MIN_SCALE = Fraction(7, 5)
FAST_PRO_SR_MAX_ALIGNED_INPUT_HEIGHT = 1440
ENCODERS = ("h264_qsv", "hevc_qsv", "libx264", "libx265", "ffv1")

def canonical_mode(mode: str) -> str:
    mode = {"fg": "fi", "sr-fi": "sr-fg"}.get(mode, mode)
    if mode not in {"sr", "fi", "sr-fg"}:
        raise FastProError(f"Intel 视频接口不支持操作: {mode}")
    return mode

def parse_rate(value: str | int | float | None) -> Fraction:
    """Parse ffprobe's rational or decimal rate without binary rounding."""
    if value is None:
        return Fraction(0, 1)
    text = str(value).strip()
    if not text or text in {"0/0", "N/A"}:
        return Fraction(0, 1)
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            if int(denominator) == 0:
                return Fraction(0, 1)
            return Fraction(int(numerator), int(denominator))
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return Fraction(0, 1)


def rate_text(rate: Fraction) -> str:
    if rate.denominator == 1:
        return str(rate.numerator)
    return f"{rate.numerator}/{rate.denominator}"


def closest_common_rate(rate: Fraction) -> str | None:
    for label, candidate in SUPPORTED_RATES.items():
        if abs(float(rate - candidate)) <= 0.002:
            return label
    return None


@dataclass(frozen=True)
class MediaInfo:
    path: str
    width: int
    height: int
    frames: int
    fps_num: int
    fps_den: int
    duration_s: float
    is_vfr: bool
    audio_streams: int
    pix_fmt: str
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    rotation: int | None
    sample_aspect_ratio: str | None = None
    display_aspect_ratio: str | None = None
    time_base: str | None = None

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den)


def _run(command: Sequence[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FastProError(f"command failed to run: {command[0]}: {exc}") from exc


def _ffmpeg_probe_json(ffmpeg: str, path: str, *, frames: bool = False) -> dict[str, Any]:
    """Small ffmpeg-only probe for portable bundles that omit ffprobe.

    The fallback intentionally uses ffmpeg's decoder and ``showinfo`` rather
    than guessing from file size.  ffprobe remains preferred because it keeps
    stream colour/rotation metadata richer, but a portable runtime can still
    perform the same fail-closed frame/PTS checks without a second executable.
    """
    inspect = _run([ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0",
                    "-f", "null", "-"], timeout=600.0)
    text = inspect.stderr
    video_line = next((line for line in text.splitlines() if "Video:" in line), "")
    size = re.search(r",\s*(\d+)x(\d+)(?:\s|,|$)", video_line)
    pix = re.search(r"Video:.*?,\s*([a-zA-Z0-9_]+)(?:\([^)]*\))?,\s*\d+x\d+", video_line)
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s+fps\b", video_line)
    aspect_match = re.search(
        r"\[SAR\s+(\d+:\d+)\s+DAR\s+(\d+:\d+)\]", video_line)
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    duration = 0.0
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    frame_matches = re.findall(r"frame=\s*(\d+)", text)
    frame_count = max((int(value) for value in frame_matches), default=0)
    if not size or not fps_match or not frame_count:
        raise FastProError(f"ffmpeg could not probe media {path}: {text[-1200:]}")
    audio_count = sum(1 for line in text.splitlines()
                      if "Stream #" in line and "Audio:" in line)
    stream: dict[str, Any] = {
        "codec_type": "video", "width": int(size.group(1)),
        "height": int(size.group(2)), "nb_frames": str(frame_count),
        "nb_read_frames": str(frame_count),
        "avg_frame_rate": fps_match.group(1), "r_frame_rate": fps_match.group(1),
        "duration": str(duration), "pix_fmt": pix.group(1).lower() if pix else "unknown",
    }
    # Header Duration may include an audio tail. It is not video duration.
    # Reconstruct the exact supported rational rate from FFmpeg's rounded
    # display label before deriving CFR video duration from decoded count.
    displayed_rate = parse_rate(fps_match.group(1))
    common_rate = closest_common_rate(displayed_rate)
    if common_rate:
        exact_rate = SUPPORTED_RATES[common_rate]
        stream["avg_frame_rate"] = stream["r_frame_rate"] = rate_text(exact_rate)
        stream["duration"] = str(float(Fraction(frame_count, 1) / exact_rate))
    if aspect_match:
        stream["sample_aspect_ratio"] = aspect_match.group(1)
        stream["display_aspect_ratio"] = aspect_match.group(2)
    for field, token in (("color_range", "tv"), ("color_space", "bt709"),
                         ("color_transfer", "bt709"), ("color_primaries", "bt709")):
        if token in video_line:
            stream[field] = token
    if re.search(r"\(pc(?:,|\))", video_line):
        stream["color_range"] = "pc"
    # The compact FFmpeg fallback is only validated for untagged/BT.709 SDR.
    # Rich metadata must never disappear merely because ffprobe was omitted.
    if re.search(r"smpte2084|arib-std-b67|bt2020|smpte170m|bt470bg|rotation of", text):
        raise FastProError("该媒体包含复杂色彩/旋转元数据，需要完整 ffprobe 依赖进行精确校验")
    result: dict[str, Any] = {"streams": [stream] + [
        {"codec_type": "audio"} for _ in range(audio_count)],
        "format": {"duration": str(duration)}}
    if frames:
        shown = _run([ffmpeg, "-hide_banner", "-loglevel", "info", "-i", path,
                      "-map", "0:v:0", "-vf", "showinfo", "-an", "-f", "null", "-"],
                     timeout=600.0)
        result["frames"] = [
            {"codec_type": "video", "best_effort_timestamp_time": value}
            for value in re.findall(r"pts_time:([-+]?\d+(?:\.\d+)?)", shown.stderr)
        ]
    return result


def _probe_json(ffprobe: str, path: str, *, frames: bool = False,
                ffmpeg: str | None = None) -> dict[str, Any]:
    ffprobe = shutil.which(ffprobe) or ffprobe if ffprobe else ffprobe
    if not ffprobe or not os.path.isfile(ffprobe):
        if ffmpeg:
            return _ffmpeg_probe_json(ffmpeg, path, frames=frames)
        raise FastProError("ffprobe is required for media validation (or pass a portable ffmpeg fallback)")
    command = [ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format"]
    if frames:
        command += ["-show_frames"]
    command += ["-of", "json", path]
    result = _run(command, timeout=300.0 if frames else 60.0)
    if result.returncode:
        raise FastProError(f"ffprobe failed for {path}: {result.stderr[-1200:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FastProError(f"invalid ffprobe JSON for {path}") from exc


def _rotation(stream: dict[str, Any]) -> int | None:
    tags = stream.get("tags") or {}
    value = tags.get("rotate")
    if value is not None:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            pass
    for side_data in stream.get("side_data_list") or []:
        value = side_data.get("rotation")
        if value is not None:
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                pass
    return None


def _stream_duration(stream: dict[str, Any], format_data: dict[str, Any]) -> float:
    for value in (stream.get("duration"), format_data.get("duration")):
        try:
            if value is not None and value != "N/A":
                duration = float(value)
                if duration > 0:
                    return duration
        except (TypeError, ValueError):
            continue
    return 0.0


def probe_media(ffprobe: str, path: str, *, ffmpeg: str | None = None) -> MediaInfo:
    data = _probe_json(ffprobe, path, ffmpeg=ffmpeg)
    streams = data.get("streams") or []
    videos = [item for item in streams if item.get("codec_type") == "video"]
    if not videos:
        raise FastProError(f"input has no video stream: {path}")
    video = videos[0]
    rate = parse_rate(video.get("avg_frame_rate")) or parse_rate(video.get("r_frame_rate"))
    frame_count = video.get("nb_read_frames", video.get("nb_frames"))
    try:
        frames = int(frame_count)
    except (TypeError, ValueError):
        frames = 0
    duration = _stream_duration(video, data.get("format") or {})
    pix_fmt = str(video.get("pix_fmt") or "unknown").lower()
    # avg/r mismatch is the least expensive reliable VFR signal.  The
    # duration-derived check also catches files whose stream flags are stale.
    r_rate = parse_rate(video.get("r_frame_rate"))
    is_vfr = bool(rate and r_rate and abs(float(rate - r_rate)) > 0.002)
    if video.get("duration") in {None, "N/A"} and frames > 0 and rate:
        # Matroska commonly has only container duration, which may be the
        # longer audio stream. Per-frame source_timeline still rejects VFR.
        duration = float(Fraction(frames, 1) / rate)
    if frames > 1 and duration > 0 and rate:
        expected_duration = float(Fraction(frames, 1) / rate)
        # Container durations are commonly rounded to centiseconds.  Treat
        # that normal <= half-frame error as CFR; otherwise a 24 fps clip
        # with duration 10.13 would be mislabelled VFR.
        is_vfr = is_vfr or abs(duration - expected_duration) > max(
            0.5 / float(rate), 0.02)
    audios = sum(1 for item in streams if item.get("codec_type") == "audio")
    return MediaInfo(
        path=os.fspath(path), width=int(video.get("width") or 0),
        height=int(video.get("height") or 0), frames=frames,
        fps_num=rate.numerator, fps_den=rate.denominator,
        duration_s=duration, is_vfr=is_vfr, audio_streams=audios,
        pix_fmt=pix_fmt,
        color_range=video.get("color_range"),
        color_space=video.get("color_space"),
        color_transfer=video.get("color_transfer"),
        color_primaries=video.get("color_primaries"),
        rotation=_rotation(video),
        sample_aspect_ratio=video.get("sample_aspect_ratio"),
        display_aspect_ratio=video.get("display_aspect_ratio"),
        time_base=video.get("time_base"),
    )


def reject_unsupported_media(info: MediaInfo) -> None:
    reasons: list[str] = []
    if info.pix_fmt not in {"nv12", "nv21", "yuv420p", "yuvj420p", "yuv422p", "yuvj422p",
                            "yuv444p", "yuvj444p", "rgb24", "bgr24", "rgba", "bgra", "gray"}:
        reasons.append(f"only validated 8-bit decode formats are accepted ({info.pix_fmt})")
    if info.pix_fmt.startswith(("p010", "yuv420p10", "yuv422p10", "yuv444p10")):
        reasons.append(f"P010/10-bit pixel format is not validated ({info.pix_fmt})")
    if info.color_transfer in {"smpte2084", "arib-std-b67", "smpte428"}:
        reasons.append(f"HDR transfer is not validated ({info.color_transfer})")
    if info.color_space in {"bt2020nc", "bt2020ncl", "bt2020c"}:
        reasons.append(f"BT.2020 HDR/color path is not validated ({info.color_space})")
    if info.color_space not in {None, "unknown", "unspecified", "bt709", "bt470bg", "smpte170m"}:
        reasons.append(f"unsupported VPP transfer matrix ({info.color_space})")
    if info.width <= 0 or info.height <= 0 or info.frames <= 0 or not info.fps:
        reasons.append("video metadata is incomplete")
    if info.width & 1 or info.height & 1:
        reasons.append("NV12 requires even dimensions")
    if info.is_vfr:
        reasons.append("VFR PTS cannot be preserved by the NV12 CFR interface; explicitly convert input to CFR first")
    common = closest_common_rate(info.fps)
    if not common:
        reasons.append(f"frame rate {rate_text(info.fps)} is outside the validated set")
    if reasons:
        raise FastProError("Fast Pro input is unsupported: " + "; ".join(reasons))


def reject_unsupported_sr_geometry(info: MediaInfo, out_width: int,
                                   out_height: int) -> None:
    """Reject SR geometry outside the measured B580 Fast Pro contract.

    The native VPP surface is aligned to 16 rows.  Portrait/vertical
    workarounds are intentionally not part of this import, so a route that
    does not satisfy the measured landscape contract must be selected by a
    later backend/router rather than silently coerced here.
    """
    aligned_height = (info.height + 15) & ~15
    if aligned_height > FAST_PRO_SR_MAX_ALIGNED_INPUT_HEIGHT:
        raise FastProError(
            "Fast Pro SR requires aligned input height <= 1440; "
            f"got {aligned_height} for {info.width}x{info.height}")
    if out_width * FAST_PRO_SR_MIN_SCALE.denominator < info.width * FAST_PRO_SR_MIN_SCALE.numerator:
        raise FastProError(
            "Fast Pro SR requires output width scale >= 1.4x; "
            f"got {out_width / info.width:.4g}x")
    if out_height * FAST_PRO_SR_MIN_SCALE.denominator < info.height * FAST_PRO_SR_MIN_SCALE.numerator:
        raise FastProError(
            "Fast Pro SR requires output height scale >= 1.4x; "
            f"got {out_height / info.height:.4g}x")


def expected_output_frames(mode: str, input_frames: int) -> int:
    operation = "sr" if canonical_mode(mode) == "sr" else "fg"
    try:
        frames, _ = contract_output_frames(
            operation, "intel-vpl-ai", input_frames)
    except ValueError as exc:
        raise FastProError(f"unknown Fast Pro mode: {mode}") from exc
    return frames


def output_rate(mode: str, input_rate: Fraction) -> Fraction:
    operation = "sr" if canonical_mode(mode) == "sr" else "fg"
    try:
        return contract_output_rate(operation, input_rate)
    except ValueError as exc:
        raise FastProError(f"unknown Fast Pro mode: {mode}") from exc


def _vpl_command(mode: str, info: MediaInfo, in_w: int, in_h: int,
                 out_w: int, out_h: int,
                 vpl: str, surface_policy: str) -> list[str]:
    common = [vpl, "sr-fi" if mode == "sr-fg" else mode]
    if mode in {"sr", "sr-fg"}:
        common += ["--in-width", str(in_w), "--in-height", str(in_h),
                   "--out-width", str(out_w), "--out-height", str(out_h),
                   "--in-fps-num", str(info.fps.numerator),
                   "--in-fps-den", str(info.fps.denominator)]
        if mode == "sr-fg":
            out = output_rate(mode, info.fps)
            common += ["--out-fps-num", str(out.numerator), "--out-fps-den", str(out.denominator)]
    else:
        out = output_rate(mode, info.fps)
        common += ["--width", str(info.width), "--height", str(info.height),
                   "--in-fps-num", str(info.fps.numerator),
                   "--in-fps-den", str(info.fps.denominator),
                   "--out-fps-num", str(out.numerator),
                   "--out-fps-den", str(out.denominator)]
    if mode == "fi" and info.color_space in {"bt709", "bt470bg", "smpte170m"}:
        common += ["--matrix", "bt709" if info.color_space == "bt709" else "bt601"]
    if mode == "fi" and info.color_range in {"tv", "limited", "pc", "full"}:
        common += ["--range", "limited" if info.color_range in {"tv", "limited"} else "full"]
    return common + ["--preset", "default", "--surface-policy", surface_policy,
                     "--input", "-", "--output", "-"]


_COLOR_RANGE_FLAGS = {"tv": "tv", "pc": "pc", "full": "pc", "limited": "tv"}


def color_flags(info: MediaInfo) -> list[str]:
    """Re-assert source colour metadata lost at the rawvideo boundary."""
    flags: list[str] = []
    if info.color_range in _COLOR_RANGE_FLAGS:
        flags += ["-color_range", _COLOR_RANGE_FLAGS[info.color_range]]
    for value, flag in ((info.color_space, "-colorspace"),
                        (info.color_transfer, "-color_trc"),
                        (info.color_primaries, "-color_primaries")):
        if value:
            flags += [flag, str(value)]
    return flags


def _sar_filter(info: MediaInfo) -> str | None:
    """Return a safe setsar expression for an anamorphic source, if present."""
    value = str(info.sample_aspect_ratio or "").strip()
    if not value or value in {"1:1", "1/1", "N/A"}:
        return None
    if ":" in value:
        numerator, denominator = value.split(":", 1)
    elif "/" in value:
        numerator, denominator = value.split("/", 1)
    else:
        return None
    try:
        if int(numerator) <= 0 or int(denominator) <= 0:
            return None
    except ValueError:
        return None
    return f"setsar={int(numerator)}/{int(denominator)}"


def build_commands(mode: str, info: MediaInfo, out_w: int, out_h: int, *,
                   ffmpeg: str, vpl: str, source: str, partial: str,
                   encoder: str = "h264_qsv", surface_policy: str = "auto",
                   plan: Any | None = None) -> tuple[list[str], list[str], list[str]]:
    mode = canonical_mode(mode)
    has_sr = mode in {"sr", "sr-fg"}
    if encoder not in ENCODERS:
        raise FastProError(f"Intel 视频接口不支持编码器: {encoder}")
    if encoder == "ffv1" and Path(partial).suffix.lower() not in {".mkv", ".avi"}:
        raise FastProError("FFV1 需要 MKV 或 AVI 输出容器")
    if surface_policy not in {"auto", "conservative"}:
        raise FastProError(f"invalid surface policy: {surface_policy}")
    if has_sr and plan is None:
        plan = plan_sr_route(info.width, info.height, out_w, out_h)
    route = getattr(plan, "route", "direct") if has_sr else "direct"
    if has_sr and route not in {"direct", "rotate"}:
        raise FastProError(
            "Fast Pro SR planner returned no supported route; use the XeSS "
            "quality path explicitly")
    out_rate = output_rate(mode, info.fps)
    out_frames = expected_output_frames(mode, info.frames)

    decode_vf = "format=nv12"
    decode_size = (info.width, info.height)
    vpp_in = (info.width, info.height)
    vpp_out = (out_w, out_h)
    encode_in = (out_w, out_h)
    encode_vf: str | None = None
    if has_sr and route == "rotate":
        detail = plan.detail
        decode_vf = "transpose=1,format=nv12"
        decode_size = (int(detail["rotate_in"][0]), int(detail["rotate_in"][1]))
        vpp_in = decode_size
        vpp_out = (int(detail["ai_out"][0]), int(detail["ai_out"][1]))
        encode_in = vpp_out
        encode_vf = (
            f"transpose=2,scale={out_w}:{out_h}:flags=lanczos,format=nv12"
        )
    elif has_sr and route == "preshrink":
        pre_w, pre_h = (int(plan.detail["pre_dims"][0]),
                        int(plan.detail["pre_dims"][1]))
        decode_vf = f"scale={pre_w}:{pre_h}:flags=lanczos,format=nv12"
        decode_size = (pre_w, pre_h)
        vpp_in = decode_size
        vpp_out = (out_w, out_h)
    sar_filter = _sar_filter(info)
    if info.color_range in {"pc", "full"}:
        decode_vf = decode_vf.replace("format=nv12", "scale=in_range=pc:out_range=pc,format=nv12")
    if sar_filter:
        encode_vf = f"{encode_vf},{sar_filter}" if encode_vf else sar_filter

    # The rawvideo pipe is bounded by the OS pipe and VPL's surface pool.  It
    # never becomes an on-disk raw intermediate and remains NV12 end-to-end.
    decode = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin",
              "-noautorotate", "-i", source, "-map", "0:v:0", "-an", "-vf", decode_vf,
              "-f", "rawvideo", "-pix_fmt", "nv12", "-s",
              f"{decode_size[0]}x{decode_size[1]}", "-frames:v", str(info.frames), "-"]
    decode[-1:-1] = ["-fps_mode", "passthrough"]
    vpp = _vpl_command(mode, info, vpp_in[0], vpp_in[1], vpp_out[0], vpp_out[1],
                       vpl, surface_policy)
    encode = [ffmpeg, "-y", "-hide_banner", "-loglevel", "warning", "-nostdin",
              "-f", "rawvideo", "-pix_fmt", "nv12", "-video_size",
              f"{encode_in[0]}x{encode_in[1]}", "-framerate", rate_text(out_rate), "-i", "-",
              "-i", source, "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
              "-map_chapters", "1"]
    if encode_vf:
        encode += ["-vf", encode_vf]
    encode += ["-fps_mode", "passthrough",
               "-c:v", encoder, "-pix_fmt", "nv12" if encoder.endswith("_qsv") else "yuv420p", "-c:a", "copy",
               *color_flags(info), "-avoid_negative_ts", "disabled"]
    if encoder in {"libx264", "libx265"}:
        encode += ["-preset", "fast", "-crf", "16"]
    elif encoder == "ffv1":
        encode += ["-level", "3"]
    if info.display_aspect_ratio and info.display_aspect_ratio not in {"N/A", "0:1"}:
        encode += ["-aspect", str(info.display_aspect_ratio)]
    if info.rotation is not None:
        encode += ["-metadata:s:v:0", f"rotate={int(info.rotation)}"]
    if Path(partial).suffix.lower() in {".mp4", ".mov", ".m4v"}:
        encode += ["-movflags", "+faststart"]
    encode += [partial]
    return decode, vpp, encode


class _PipePump:
    def __init__(self, source: Any, destination: Any) -> None:
        self.source = source
        self.destination = destination
        self.bytes = 0
        self.first_chunk_s: float | None = None
        self.error: str | None = None
        self._start = time.monotonic()
        self.thread = threading.Thread(target=self._run, name="fast-pro-vpp-pump",
                                       daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                block = self.source.read(1024 * 1024)
                if not block:
                    break
                if self.first_chunk_s is None:
                    self.first_chunk_s = time.monotonic() - self._start
                self.destination.write(block)
                self.destination.flush()
                self.bytes += len(block)
        except (BrokenPipeError, OSError) as exc:
            self.error = str(exc)
        finally:
            try:
                self.source.close()
            except OSError:
                pass
            try:
                self.destination.close()
            except OSError:
                pass


class _StderrDrain:
    def __init__(self, stream: Any, path: Path) -> None:
        self.stream = stream
        self.path = path
        self.tail = bytearray()
        self.thread = threading.Thread(target=self._run, name=f"fast-pro-stderr-{path.stem}",
                                       daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("wb") as output:
                while True:
                    block = self.stream.read(64 * 1024)
                    if not block:
                        break
                    output.write(block)
                    self.tail.extend(block)
                    if len(self.tail) > 64 * 1024:
                        del self.tail[:-64 * 1024]
        finally:
            try:
                self.stream.close()
            except OSError:
                pass


def _terminate(processes: Iterable[subprocess.Popen[Any]], timeout_s: float = 5.0) -> None:
    active = [process for process in processes if process.poll() is None]
    for process in reversed(active):
        try:
            process.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + timeout_s
    for process in reversed(active):
        remaining = max(0.05, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass


def run_stream_chain(decode: Sequence[str], vpp: Sequence[str], encode: Sequence[str], *,
                     log_dir: str | os.PathLike[str], cancel_file: str | None = None,
                     cancel_event: threading.Event | None = None,
                     timeout_s: float | None = None, require_backend: bool = True) -> dict[str, Any]:
    """Run one bounded three-process chain with continuously drained stderr."""
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[Any]] = []
    drains: list[_StderrDrain] = []
    pump: _PipePump | None = None
    start = time.monotonic()
    cancelled = False
    try:
        dec = subprocess.Popen(list(decode), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(dec)
        vpp = subprocess.Popen(list(vpp), stdin=dec.stdout, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        if dec.stdout is not None:
            dec.stdout.close()
        processes.append(vpp)
        enc = subprocess.Popen(list(encode), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(enc)
        stderr_drains = [(dec, "decode"), (vpp, "vpp"), (enc, "encode")]
        for process, name in stderr_drains:
            drain = _StderrDrain(process.stderr, log_root / f"{name}.stderr.log")
            drain.start()
            drains.append(drain)
        pump = _PipePump(vpp.stdout, enc.stdin)
        pump.start()
        for process in processes:
            while process.poll() is None:
                marker = cancel_file and os.path.isfile(cancel_file)
                if (cancel_event is not None and cancel_event.is_set()) or marker:
                    cancelled = True
                    _terminate(processes)
                    break
                if timeout_s is not None and time.monotonic() - start > timeout_s:
                    _terminate(processes)
                    raise FastProError(f"Fast Pro chain exceeded timeout {timeout_s:g}s")
                # A failed upstream process can leave the encoder waiting on a
                # pipe forever.  Fail the whole bounded chain immediately.
                failed = [(p, p.returncode) for p in processes
                          if p.poll() is not None and p.returncode not in (None, 0)]
                if failed:
                    _terminate(processes)
                    break
                if pump is not None and pump.error:
                    _terminate(processes)
                    break
                time.sleep(0.02)
            if cancelled:
                break
        for process in processes:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                _terminate(processes)
        if pump is not None:
            pump.thread.join(timeout=5.0)
        for drain in drains:
            drain.thread.join(timeout=5.0)
        if cancelled:
            raise FastProError("Fast Pro cancellation requested")
        codes = [process.returncode for process in processes]
        tails = {name: bytes(drain.tail).decode("utf-8", "replace")
                 for (_, name), drain in zip(stderr_drains, drains)}
        if any(code != 0 for code in codes):
            detail = "; ".join(f"{name}: {tails.get(name, '')[-800:]}"
                                for name in tails if tails[name])
            raise FastProError(f"Fast Pro chain failed (exit={codes}); {detail}")
        backend_marker = "backend=intel-vpl-ai" in tails.get("vpp", "")
        if require_backend and not backend_marker:
            raise FastProError("VPL process did not report backend=intel-vpl-ai; refusing fallback")
        if pump is not None and pump.error:
            raise FastProError(f"Fast Pro surface stream failed: {pump.error}")
        return {"backend": "intel-vpl-ai", "backend_marker": backend_marker,
                "elapsed_s": time.monotonic() - start,
                "first_chunk_s": pump.first_chunk_s if pump else None,
                "output_bytes": pump.bytes if pump else 0,
                "exit_codes": codes, "stderr": tails,
                "stderr_paths": {name: str(log_root / f"{name}.stderr.log")
                                 for _, name in stderr_drains}}
    finally:
        _terminate(processes)


def _first_frame_check(ffmpeg: str, path: str) -> None:
    # Full decode is required before publishing: damaged later frames cannot
    # pass merely because the first frame and container index are readable.
    result = _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-i", path,
                   "-map", "0:v:0", "-f", "null", "-"], timeout=600)
    if result.returncode:
        raise FastProError(f"encoded video failed complete decode: {result.stderr[-1200:]}")


def _rewrite_color_metadata(ffmpeg: str, path: str, info: MediaInfo) -> None:
    """Materialize color tags when the hardware encoder drops them.

    QSV preserves the pixels but some FFmpeg/QSV combinations omit BT.709
    transfer/primaries from the output stream even when ``-color_*`` options
    were supplied during rawvideo encoding.  A stream-copy metadata pass keeps
    PTS, encoded frames, and the original audio bit-for-bit while making the
    declared SDR signal explicit for the fail-closed validator.
    """
    flags = color_flags(info)
    if not flags:
        return
    source = Path(path)
    # Keep a media suffix so FFmpeg can infer the MP4 muxer.
    repaired = source.with_name(source.stem + ".metadata.partial" + source.suffix)
    repaired.unlink(missing_ok=True)
    # MKV can start with negative AAC priming PTS. Without -copyts this
    # metadata-only remux shifts the whole video by that audio lead-in.
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-copyts",
               "-i", os.fspath(source), "-map", "0", "-c", "copy",
               *flags, "-avoid_negative_ts", "disabled", os.fspath(repaired)]
    try:
        result = _run(command)
        if result.returncode:
            raise FastProError(f"color metadata remux failed: {result.stderr[-1200:]}")
        os.replace(repaired, source)
    except BaseException:
        repaired.unlink(missing_ok=True)
        raise


def _frame_pts(ffprobe: str, path: str, *, ffmpeg: str | None = None) -> list[float]:
    data = _probe_json(ffprobe, path, frames=True, ffmpeg=ffmpeg)
    pts: list[float] = []
    for frame in data.get("frames") or []:
        # ffprobe's frame records use ``media_type`` (while some wrappers use
        # ``codec_type``).  Both are explicit type fields; a record with
        # neither is not evidence of a video frame.  Previously accepting a
        # missing codec_type counted AAC timestamps as video PTS.
        frame_type = frame.get("codec_type") or frame.get("media_type")
        if frame_type != "video":
            continue
        value = frame.get("best_effort_timestamp_time", frame.get("pkt_dts_time"))
        try:
            pts.append(float(value))
        except (TypeError, ValueError):
            raise FastProError("encoded video contains a frame without a timestamp")
    return pts


def source_timeline(ffprobe: str, ffmpeg: str, info: MediaInfo) -> dict[str, Any]:
    pts = _frame_pts(ffprobe, info.path, ffmpeg=ffmpeg)
    tick = parse_rate(info.time_base)
    tolerance = max(0.000001, 1.1 * float(tick)) if tick else 0.0011
    if len(pts) != info.frames or not pts:
        raise FastProError("输入 PTS 数量与解码帧数不一致")
    if abs(pts[0]) > tolerance:
        raise FastProError("Intel 视频接口当前要求视频首 PTS 为零；非零起点需显式预处理以保持音画偏移")
    if any(abs(value - float(Fraction(index, 1) / info.fps)) > tolerance
           for index, value in enumerate(pts)):
        raise FastProError("检测到 VFR/不连续 PTS；Intel NV12 接口不能保留该时序，请先显式转换为 CFR")
    return {"fps_num": info.fps_num, "fps_den": info.fps_den,
            "frames": info.frames, "pts_first": pts[0], "pts_last": pts[-1],
            "time_base": info.time_base,
            "pts_origin": "preserved-zero", "cadence": "CFR", "tolerance_s": tolerance}


def file_identity(path: str) -> dict[str, Any]:
    resolved = Path(shutil.which(path) or path).resolve()
    if not resolved.is_file():
        raise FastProError(f"缺少运行依赖: {path}")
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest}


def audio_fingerprint(ffmpeg: str, path: str, streams: int) -> list[str]:
    values = []
    for index in range(streams):
        result = _run([ffmpeg, "-v", "error", "-i", path, "-map", f"0:a:{index}",
                       "-c", "copy", "-f", "hash", "-hash", "sha256", "-"], timeout=600)
        if result.returncode:
            raise FastProError(f"音轨校验失败: {result.stderr[-1000:]}")
        values.append(result.stdout.strip())
    return values


def validate_output(ffprobe: str, ffmpeg: str, output: str, source: MediaInfo, *,
                    expected_width: int, expected_height: int,
                    expected_frames: int, expected_fps: Fraction,
                    expected_audio: bool | None = None) -> dict[str, Any]:
    """Validate frames, PTS, duration, audio and colour metadata fail-closed."""
    if not os.path.isfile(output) or os.path.getsize(output) <= 0:
        raise FastProError(f"encoded output is missing or empty: {output}")
    _first_frame_check(ffmpeg, output)
    data = _probe_json(ffprobe, output, frames=False, ffmpeg=ffmpeg)
    streams = data.get("streams") or []
    videos = [item for item in streams if item.get("codec_type") == "video"]
    if not videos:
        raise FastProError("encoded output has no video stream")
    video = videos[0]
    got_size = (int(video.get("width") or 0), int(video.get("height") or 0))
    if got_size != (expected_width, expected_height):
        raise FastProError(f"output resolution mismatch: expected {(expected_width, expected_height)}, got {got_size}")
    try:
        got_frames = int(video.get("nb_read_frames", video.get("nb_frames")))
    except (TypeError, ValueError):
        got_frames = 0
    if not got_frames:
        raise FastProError("ffprobe could not count output frames")
    if got_frames != expected_frames:
        raise FastProError(f"output frame-count mismatch: expected {expected_frames}, got {got_frames}")
    pts = _frame_pts(ffprobe, output, ffmpeg=ffmpeg)
    if len(pts) != got_frames:
        raise FastProError(f"output PTS count mismatch: {len(pts)} != {got_frames}")
    if any(curr <= prev for prev, curr in zip(pts, pts[1:])):
        raise FastProError("output PTS is not strictly monotonic")
    tick = parse_rate(video.get("time_base"))
    tolerance = max(0.000001, 1.1 * float(tick)) if tick else 0.0011
    if any(abs(value - float(Fraction(index, 1) / expected_fps)) > tolerance
           for index, value in enumerate(pts)):
        worst = max(range(len(pts)), key=lambda i: abs(pts[i]-float(Fraction(i)/expected_fps)))
        raise FastProError("output PTS cadence or first timestamp differs from the rational output contract: "
                           f"frame={worst},actual={pts[worst]},expected={float(Fraction(worst)/expected_fps)},"
                           f"time_base={video.get('time_base')},tolerance={tolerance},first={pts[:4]}")
    duration = _stream_duration(video, data.get("format") or {})
    if video.get("duration") in {None, "N/A"}:
        duration = pts[-1] - pts[0] + float(1 / expected_fps)
    if source.duration_s and duration:
        tolerance = 1.0 / max(float(expected_fps), 1.0)
        if abs(duration - source.duration_s) > tolerance:
            raise FastProError(f"output duration error exceeds one frame: source={source.duration_s:.6f}s output={duration:.6f}s")
    audio = sum(1 for item in streams if item.get("codec_type") == "audio")
    wanted_audio = source.audio_streams > 0 if expected_audio is None else expected_audio
    if bool(audio) != bool(wanted_audio):
        raise FastProError(f"audio presence mismatch: expected={bool(wanted_audio)} got={bool(audio)}")
    if wanted_audio and audio != source.audio_streams:
        raise FastProError(f"audio stream-count mismatch: {audio} != {source.audio_streams}")
    for key in ("color_range", "color_space", "color_transfer", "color_primaries"):
        expected = getattr(source, key)
        if expected is not None and video.get(key) != expected:
            raise FastProError(f"colour metadata mismatch for {key}: expected={expected!r} got={video.get(key)!r}")
    if source.rotation is not None and _rotation(video) != source.rotation:
        raise FastProError(f"rotation metadata mismatch: expected={source.rotation} got={_rotation(video)}")
    for key in ("sample_aspect_ratio", "display_aspect_ratio"):
        expected = getattr(source, key)
        if expected is not None and video.get(key) != expected:
            raise FastProError(
                f"aspect metadata mismatch for {key}: expected={expected!r} "
                f"got={video.get(key)!r}")
    pix_fmt = str(video.get("pix_fmt") or "").lower()
    if pix_fmt.startswith(("p010", "yuv420p10", "yuv422p10", "yuv444p10")):
        raise FastProError(f"output unexpectedly uses unvalidated 10-bit format: {pix_fmt}")
    return {"width": got_size[0], "height": got_size[1], "frames": got_frames,
            "fps": float(expected_fps), "fps_num": expected_fps.numerator,
            "fps_den": expected_fps.denominator, "duration_s": duration, "audio": bool(audio),
            "audio_streams": audio, "complete_decode": True,
            "pts_first": pts[0], "pts_last": pts[-1], "pix_fmt": pix_fmt,
            "sample_aspect_ratio": video.get("sample_aspect_ratio"),
            "display_aspect_ratio": video.get("display_aspect_ratio"),
            "rotation": _rotation(video),
            "color": {key: video.get(key) for key in
                       ("color_range", "color_space", "color_transfer", "color_primaries")}}


def run_fast_pro(*, mode: str, source_path: str, output_path: str, ffprobe: str,
                 ffmpeg: str, vpl: str, work_dir: str, out_width: int | None = None,
                 out_height: int | None = None, encoder: str = "h264_qsv",
                 surface_policy: str = "auto", cancel_file: str | None = None,
                 cancel_event: threading.Event | None = None,
                 timeout_s: float = 1200) -> dict[str, Any]:
    total_start = time.monotonic()
    mode = canonical_mode(mode)
    info = probe_media(ffprobe, source_path, ffmpeg=ffmpeg)
    reject_unsupported_media(info)
    if mode in {"sr", "sr-fg"}:
        if info.color_range in {"pc", "full"} or info.color_space in {"bt470bg", "smpte170m"}:
            raise FastProError("Intel AI SR 当前不接受显式颜色信号扩展；全范围/BT.601 SR未验证，需显式转换为BT.709有限范围输入")
        out_width = out_width or info.width
        out_height = out_height or info.height
        if out_width < info.width or out_height < info.height:
            raise FastProError("Fast Pro SR cannot downscale")
        route_plan = plan_sr_route(info.width, info.height, out_width, out_height)
        if not route_plan.supported or route_plan.route == "preshrink":
            raise FastProError(
                "Fast Pro SR has no supported route for "
                f"{info.width}x{info.height} -> {out_width}x{out_height}: "
                f"{route_plan.reason}. Use the XeSS quality path for this "
                "geometry.")
    else:
        out_width, out_height = info.width, info.height
        route_plan = None
    if out_width & 1 or out_height & 1:
        raise FastProError("Fast Pro output dimensions must be even")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    destination = Path(output_path).resolve()
    if destination == Path(source_path).resolve():
        raise FastProError("输出不能覆盖输入原片")
    if destination.exists():
        raise FastProError(f"输出已存在，请选择新文件名: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + ".intel-vpl.partial" + destination.suffix)
    if partial.exists():
        raise FastProError(f"发现已有未完成文件，请使用新输出名: {partial}")
    log_dir = work / "logs" / (Path(output_path).stem + "_fast_pro")
    decode, vpp, encode = build_commands(
        mode, info, out_width, out_height, ffmpeg=ffmpeg, vpl=vpl,
        source=source_path, partial=os.fspath(partial), encoder=encoder,
        surface_policy=surface_policy, plan=route_plan)
    try:
        if cancel_file and Path(cancel_file).exists() or cancel_event and cancel_event.is_set():
            raise FastProError("Intel 视频接口任务已取消")
        identities = {name: file_identity(path) for name, path in
                      (("ffmpeg", ffmpeg), ("vpl", vpl))}
        timeline = source_timeline(ffprobe, ffmpeg, info)
        if shutil.disk_usage(destination.parent).free < 64 * 1024 * 1024:
            raise FastProError("输出磁盘剩余空间不足64MiB")
        encoders = _run([ffmpeg, "-v", "error", "-encoders"])
        if encoders.returncode or not re.search(r"\b" + re.escape(encoder) + r"\b", encoders.stdout):
            raise FastProError(f"FFmpeg 缺少所选编码器: {encoder}")
        query = list(vpp)
        query[query.index("--input") + 1] = os.devnull
        query[query.index("--output") + 1] = os.devnull
        preflight = _run(query, timeout=30)
        if preflight.returncode or "backend=intel-vpl-ai" not in preflight.stderr:
            raise FastProError(f"Intel AI 参数预检失败: {preflight.stderr[-1800:]}")
        result = run_stream_chain(decode, vpp, encode, log_dir=log_dir,
                                  cancel_file=cancel_file, cancel_event=cancel_event,
                                  timeout_s=timeout_s)
        _rewrite_color_metadata(ffmpeg, os.fspath(partial), info)
        result["validation"] = validate_output(
            ffprobe, ffmpeg, os.fspath(partial), info,
            expected_width=out_width, expected_height=out_height,
            expected_frames=expected_output_frames(mode, info.frames),
            expected_fps=output_rate(mode, info.fps))
        before_audio = audio_fingerprint(ffmpeg, source_path, info.audio_streams)
        after_audio = audio_fingerprint(ffmpeg, str(partial), info.audio_streams)
        if before_audio != after_audio:
            raise FastProError("音轨压缩载荷与原片不一致；拒绝发布截断音频")
        result["audio_payload_sha256"] = after_audio
        if (cancel_file and Path(cancel_file).exists()) or (cancel_event and cancel_event.is_set()):
            raise FastProError("Intel 视频接口任务已取消；输出未发布")
        os.replace(partial, output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    result.update({"mode": mode, "source": asdict(info), "output": os.fspath(output_path),
                   "output_width": out_width, "output_height": out_height,
                   "route_plan": (asdict(route_plan) if route_plan is not None else None),
                   "requested_geometry": {
                       "mode": mode,
                       "input": [info.width, info.height],
                       "output": [out_width, out_height],
                   },
                   "planned_geometry": {
                       "route": route_plan.route if route_plan is not None else "direct",
                       "backend": "intel-vpl-ai",
                       "rotate_workaround": bool(
                           route_plan is not None and route_plan.route == "rotate"),
                   },
                   "applied_geometry": {
                       "route": route_plan.route if route_plan is not None else "direct",
                       "backend": result.get("backend", "intel-vpl-ai"),
                       "rotate_workaround": bool(
                           route_plan is not None and route_plan.route == "rotate"),
                   },
                   # Short aliases keep the route state machine easy for
                   # capability consumers to inspect without unpacking the
                   # richer geometry records above.
                   "requested": {
                       "mode": mode,
                       "input": [info.width, info.height],
                       "output": [out_width, out_height],
                   },
                   "planned": {
                       "route": route_plan.route if route_plan is not None else "direct",
                       "rotate_workaround": bool(
                           route_plan is not None and route_plan.route == "rotate"),
                   },
                   "applied": {
                       "route": route_plan.route if route_plan is not None else "direct",
                       "rotate_workaround": bool(
                           route_plan is not None and route_plan.route == "rotate"),
                   },
                   "rotate_workaround_used": bool(
                       route_plan is not None and route_plan.route == "rotate"),
                   "surface_policy": surface_policy, "encoder": encoder,
                   "timing_mode": "cfr-preserved",
                   "raw_intermediate": False, "rgb24_intermediate": False,
                   "expected_output_frames": expected_output_frames(mode, info.frames),
                   "fi_frame_semantics": "2N" if mode != "sr" else None,
                   "backend": "intel-vpl-ai"})
    native_log = result["stderr"]["vpp"]
    counts = re.search(r"Processed (\d+) input frames -> (\d+) output frames", native_log)
    cpu_bytes = re.search(r"cpu_nv12_upload_bytes=(\d+); cpu_nv12_readback_bytes=(\d+)", native_log)
    result.update({"schema": "intel-vpl-job-v2", "status": "success",
        "operation": "fg" if mode == "fi" else mode, "display_name": "Intel 视频接口",
        "commands": {"decode": decode, "vpp": vpp, "encode": encode, "preflight": query},
        "preflight": {"stderr": preflight.stderr, "returncode": preflight.returncode},
        "input_timeline": timeline, "dependencies": identities,
        "native_input_frames": int(counts[1]) if counts else None,
        "native_output_frames": int(counts[2]) if counts else None,
        "cpu_full_frame_upload_bytes": int(cpu_bytes[1]) if cpu_bytes else None,
        "cpu_full_frame_readback_bytes": int(cpu_bytes[2]) if cpu_bytes else None,
        "gpu_only": False, "vpp_sessions": 2, "processing_vpp_sessions": 1,
        "preflight_vpp_sessions": 1, "ai_extensions": 2 if mode == "sr-fg" else 1,
        "source_pair_analysis_count": None, "directional_dispatch_count": None,
        "sr_consume_count": None, "fg_consume_count": None,
        "internal_motion_observability": "Intel runtime does not expose motion or shared-analysis counters",
        "vpp_color_signal": {"matrix": info.color_space, "range": info.color_range,
            "signal_extension_applied": mode == "fi" and bool(info.color_space in {"bt709", "bt470bg", "smpte170m"} or info.color_range in {"tv", "limited", "pc", "full"}),
            "sr_extension_constraint": "AI SR rejects VideoSignalInfo; only existing limited-BT709 or untagged NV12 SR path is validated",
            "unknown_source_policy": "preserve unknown; no invented color metadata"},
        "runtime_modules": re.findall(r"runtime_module=([^\r\n]+)", native_log),
        "total_wall_s": time.monotonic() - total_start,
        "first_complete_playable_s": time.monotonic() - total_start,
        "audio_policy": "copy-all-streams-verified-payload-sha256",
        "final_encode_count": 1, "intermediate_encoded_files": 0})
    for state in ("requested", "planned", "applied"):
        result[state].update({"backend": "intel-vpl-ai", "operation": result["operation"],
                              "encoder": encoder, "output": [out_width, out_height],
                              "fps_num": output_rate(mode, info.fps).numerator,
                              "fps_den": output_rate(mode, info.fps).denominator})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fast Pro Intel oneVPL AI SR/FI media path")
    parser.add_argument("mode", choices=("sr", "fi", "fg", "sr-fg", "sr-fi"))
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--vpl", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out-width", type=int)
    parser.add_argument("--out-height", type=int)
    parser.add_argument("--encoder", choices=ENCODERS, default="h264_qsv")
    parser.add_argument("--surface-policy", choices=("auto", "conservative"), default="auto")
    parser.add_argument("--cancel-file")
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    args = parser.parse_args(argv)
    try:
        result = run_fast_pro(mode=args.mode, source_path=args.source, output_path=args.output,
                              ffprobe=args.ffprobe, ffmpeg=args.ffmpeg, vpl=args.vpl,
                              work_dir=args.work_dir, out_width=args.out_width,
                              out_height=args.out_height, encoder=args.encoder,
                              surface_policy=args.surface_policy, cancel_file=args.cancel_file,
                              timeout_s=args.timeout_seconds)
    except (FastProError, OSError) as exc:
        report = Path(args.work_dir).resolve() / "reports" / "fast_pro_last.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"schema": "intel-vpl-job-v2", "backend": "intel-vpl-ai",
            "status": "cancelled" if "cancel" in str(exc).lower() or "取消" in str(exc) else "failed",
            "error": str(exc), "requested": vars(args), "applied": None}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[fast-pro] error: {exc}", file=sys.stderr)
        return 1
    report = Path(args.work_dir).resolve() / "reports" / "fast_pro_last.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
