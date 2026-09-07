"""Read compressed packet PTS without decoding a second copy of the video."""
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess


def rate_text(value):
    return str(Fraction(value).limit_denominator(1_000_000))


def probe_cfr(video, ffmpeg, *, max_frames=0):
    candidates = (os.environ.get("XESS_FFPROBE"),
                  str(Path(ffmpeg).with_name("ffprobe.exe")), shutil.which("ffprobe"))
    executable = next((path for path in candidates if path and Path(path).is_file()), None)
    if executable is None:
        raise RuntimeError("ffprobe is required to verify source PTS; set XESS_FFPROBE to its complete runtime installation")
    command = [executable, "-v", "error", "-select_streams", "v:0", "-show_streams",
               "-show_packets", "-show_entries",
               "stream=width,height,avg_frame_rate,r_frame_rate,time_base:packet=pts,duration",
               "-of", "json", os.fspath(video)]
    data = json.loads(subprocess.check_output(command, timeout=60))
    stream = data["streams"][0]
    time_base = Fraction(stream["time_base"])
    rate = Fraction(stream["avg_frame_rate"])
    if rate <= 0:
        rate = Fraction(stream["r_frame_rate"])
    if rate <= 0:
        raise RuntimeError("source has no valid rational frame rate")
    packets = data.get("packets", [])
    if not packets or any("pts" not in packet for packet in packets):
        raise RuntimeError("source packet PTS is unavailable; fixed-rate raw transport cannot preserve unknown timing")
    pts = sorted(int(packet["pts"]) for packet in packets)
    if len(set(pts)) != len(pts):
        raise RuntimeError("duplicate source PTS is unsupported by the fixed-rate raw transport")
    origin = pts[0]
    # Container average rates can include a shortened final packet duration
    # (e.g. selecting even frames). Validate the actual PTS grid before choosing
    # either reported rational rate; never silently prefer an inaccurate mean.
    candidates = [rate, Fraction(stream["r_frame_rate"])]
    valid_rates = [candidate for candidate in candidates if candidate > 0 and
                   all(abs((point - origin) * time_base - Fraction(index, 1) / candidate) <= time_base
                       for index, point in enumerate(pts))]
    if not valid_rates:
        raise RuntimeError("VFR source is unsupported by the fixed-rate SR/FG raw transport; refusing silent CFR retiming")
    rate = valid_rates[0]
    if max_frames > 0:
        pts = pts[:max_frames]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "fps": float(rate), "fps_rational": str(rate), "frames": len(pts),
            "source_pts_ticks": pts, "source_time_base": str(time_base),
            "source_pts_origin": origin, "timing_policy": "CFR verified from compressed packet PTS; output starts at zero",
            "probe_command": command}
