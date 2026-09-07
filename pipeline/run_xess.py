#!/usr/bin/env python3
"""XeSS SR 1.2 portable video upscaler."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from chunked_media import concat_command, extract_lossless_command, write_concat_list
from media_validation import MediaValidationError, validate_output
from media_timeline import probe_cfr, rate_text
from shm_ring import RingOwner, packet_slot_size
from stage_timer import StageTimer, timing_requested
from source_fanout import SourceFanout
from offline_encoders import video_options
import runtime_hooks
from workdir_guard import (WorkdirError, create_workspace, estimate_sr_bytes,
                           finalize_output, partial_output_path)


ROOT = os.path.dirname(os.path.abspath(__file__))


def first_existing(*paths):
    return next((os.path.abspath(path) for path in paths if path and os.path.isfile(path)), "")


PY = first_existing(os.environ.get("XESS_PYTHON"), os.path.join(ROOT, "python", "python.exe"), sys.executable)
FFMPEG = first_existing(os.environ.get("XESS_FFMPEG"), os.path.join(ROOT, "ffmpeg.exe"), shutil.which("ffmpeg"))
XESS = first_existing(os.environ.get("XESS_VSR"), os.path.join(ROOT, "xess-vsr.exe"), os.path.join(ROOT, "build", "xess-vsr.exe"))
FLOW = os.path.join(ROOT, "flow.py")
PREPARE = os.path.join(ROOT, "prepare_sr.py")
POST = os.path.join(ROOT, "sr_postprocess.py")
FUSION = os.path.join(ROOT, "five_frame_fusion.py")
MFSR = os.path.join(ROOT, "five_frame_mfsr.py")
DEPTH_MODEL = os.path.join(ROOT, "models", "depth-anything-v2-small", "depth_anything_v2_small.xml")

# Mainline motion estimation is native Fast DIS only.  The retired SEA-RAFT
# choices stay accepted so old workflows and scripts keep running on DIS.
FLOW_MODES = {
    "dis-fast": ("dis", False),
    "dis-occlusion": ("dis", True),
    "sea-raft-single": ("dis", False),
    "sea-raft": ("dis", False),
}
LEGACY_FLOW_MODES = ("sea-raft-single", "sea-raft")
_MIGRATION_PRINTED = False


def resolve_flow_mode(flow_mode):
    """Map a retired SEA-RAFT flow mode onto native Fast DIS once per process."""
    global _MIGRATION_PRINTED
    if flow_mode in LEGACY_FLOW_MODES and not _MIGRATION_PRINTED:
        print(f"[run_xess] SEA-RAFT has been retired from the mainline; "
              f"flow mode '{flow_mode}' now runs native Fast DIS", flush=True)
        _MIGRATION_PRINTED = True
    return "dis-fast" if flow_mode in LEGACY_FLOW_MODES else flow_mode


def die(message):
    print(f"[run_xess] error: {message}", file=sys.stderr)
    raise SystemExit(1)


def command_text(command):
    return subprocess.list2cmdline([os.fspath(item) for item in command])


def run(command, *, environment=None, **kwargs):
    print(f"[run_xess] $ {command_text(command)}", flush=True)
    result = subprocess.run(command, env=environment, **kwargs)
    if result.returncode:
        detail = result.stderr[-3000:] if getattr(result, "stderr", None) else ""
        die(f"command failed ({result.returncode}): {detail}")
    return result


def quality_for(scale):
    if scale <= 1.05:
        return 6
    if scale <= 1.6:
        return 4
    if scale <= 2.2:
        return 3
    if scale <= 2.8:
        return 2
    if scale <= 3.5:
        return 1
    return 0


def resolve_settings(args):
    defaults = {
        "fast": {"bidirectional": False, "mv_path": "highres",
                 "responsive": True, "sharpen": 0.25, "static": 0.30, "motion": 0.16},
        "balanced": {"bidirectional": False, "mv_path": "highres",
                     "responsive": True, "sharpen": 0.30, "static": 0.34, "motion": 0.18},
        "quality": {"bidirectional": False, "mv_path": "highres",
                    "responsive": True, "sharpen": 0.35, "static": 0.38, "motion": 0.20},
    }[args.preset]
    flow = "dis"
    bidirectional = defaults["bidirectional"]
    if args.flow_mode != "auto":
        flow, bidirectional = FLOW_MODES[resolve_flow_mode(args.flow_mode)]
    mv_path = args.mv_path if args.mv_path != "auto" else defaults["mv_path"]
    responsive = defaults["responsive"] if args.responsive_mask == "auto" else args.responsive_mask == "on"
    sharpen = defaults["sharpen"] if args.sharpen is None else args.sharpen
    sharpen_mode = args.sharpen_mode
    if sharpen_mode == "auto":
        sharpen_mode = "fixed" if args.preset == "fast" else "adaptive"
    io_mode = args.io_mode
    static = args.sharpen_static if args.sharpen_static is not None else defaults["static"]
    motion = args.sharpen_motion if args.sharpen_motion is not None else defaults["motion"]
    return {"flow": flow, "bidirectional": bidirectional, "mv_path": mv_path,
            "responsive": responsive, "sharpen": sharpen,
            "sharpen_mode": sharpen_mode, "static": static,
            "motion": motion, "io_mode": io_mode}


def resolve_io_mode(requested: str, input_height: int, output_height: int) -> str:
    """Choose the default transport for a resolution without hidden thresholds."""
    if requested == "auto":
        return "shared" if max(input_height, output_height) >= 720 else "stream"
    return requested


def prep_command(args, settings, width, height, frames, *, stream,
                 raw="", mv_dir="", depth_dir="", mask_dir="", debug_dir="", ring=None):
    command = [PY, PREPARE, "--in-w", str(width), "--in-h", str(height),
               "--frames", str(frames), "--engine", settings["flow"],
               "--temporal", str(args.depth_temporal), "--consistency", str(args.flow_consistency),
               "--dilate", str(args.mv_dilate), "--depth-edge", str(args.depth_edge),
               "--mv-path", settings["mv_path"], "--responsive-max", str(args.responsive_max)]
    if stream:
        command.append("--stream")
        if ring is not None:
            command.extend(ring.arguments())
    else:
        command.extend(("--raw", raw, "--mv-out", mv_dir))
    if settings["bidirectional"]:
        command.append("--bidirectional")
    needs_depth = settings["mv_path"] == "lowres-depth" or args.force_depth
    if needs_depth:
        command.extend(("--depth-model", args.depth_model, "--depth-device", args.depth_device))
        if not stream and depth_dir:
            command.extend(("--depth-out", depth_dir))
    if settings["responsive"]:
        command.append("--responsive-mask")
        if not stream and mask_dir:
            command.extend(("--mask-out", mask_dir))
    if debug_dir:
        command.extend(("--debug-dir", debug_dir))
    shared_motion_dir = getattr(args, "shared_motion_dir", "")
    if shared_motion_dir:
        command.extend(("--shared-motion-dir", os.fspath(shared_motion_dir),
                        "--shared-motion-slots",
                        str(getattr(args, "shared_motion_slots", 4)),
                        "--pts-num", str(getattr(args, "pts_num", 1)),
                        "--pts-den", str(getattr(args, "pts_den", 1))))
        if getattr(args, "source_pts_file", ""):
            command.extend(("--source-pts-file", os.fspath(args.source_pts_file)))
    return command, needs_depth


def xess_command(args, settings, width, height, out_w, out_h, frames, quality, *,
                 stream, raw="", mv_dir="", depth_dir="", mask_dir="", out_raw="", reset="",
                 ring=None, out_ring=None):
    command = [XESS, "--in-w", str(width), "--in-h", str(height), "--out-w", str(out_w),
               "--out-h", str(out_h), "--frames-count", str(frames), "--quality", str(quality),
               "--mv-path", settings["mv_path"], "--responsive-max", str(args.responsive_max)]
    if stream:
        command.append("--stream")
        if ring is not None:
            command.extend(ring.arguments())
        if out_ring is not None:
            command.extend(["--out-shm-name", out_ring.name,
                            "--out-shm-slots", str(out_ring.slots),
                            "--out-shm-slot-size", str(out_ring.slot_size)])
        if settings["responsive"]:
            command.extend(("--mask", "stream"))
    else:
        command.extend(("--frames", raw, "--mv", mv_dir, "--out", out_raw))
        if settings["mv_path"] == "lowres-depth":
            command.extend(("--depth", depth_dir))
        if settings["responsive"]:
            command.extend(("--mask", mask_dir))
        if reset:
            command.extend(("--reset-frames", reset))
    if settings["mv_path"] == "highres":
        command.extend(("--mv-upsample", "bilinear"))
    if args.device >= 0:
        command.extend(("--device", str(args.device)))
    if args.verbose:
        command.append("--verbose")
    return command


def encoder_command(args, settings, out_w, out_h, fps, frames, partial):
    command = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", rate_text(fps), "-i", "-",
               "-i", args.video, "-map", "0:v", "-map", "1:a?"] + video_options(args)
    if (settings["sharpen_mode"] == "fixed" and settings["sharpen"] > 0 and
            args.edge_guard_strength <= 0):
        command.extend(("-vf", f"cas=strength={settings['sharpen']:.4f}"))
    command.extend(("-c:a", "copy", "-t", f"{frames / fps:.9f}",
                    "-frames:v", str(frames), os.fspath(partial)))
    return command


def post_command(args, settings, width, height, out_w, out_h, frames, *, ring=None):
    """Fused final-stage processor: sharpening + vertical ringing guard.

    Returns ``None`` when neither effect needs its own stage -- fixed
    sharpening without a guard keeps living in the encoder's cas filter.
    """
    guard = args.edge_guard_strength > 0
    sharpen_process = settings["sharpen"] > 0 and (
        settings["sharpen_mode"] == "adaptive" or guard)
    if not sharpen_process and not guard:
        return None
    command = [PY, POST, "--width", str(out_w), "--height", str(out_h),
               "--frames", str(frames),
               "--sharpen-mode", settings["sharpen_mode"] if sharpen_process else "off"]
    command.extend(("--threads", str(getattr(args, "post_threads", 4))))
    if ring is not None:
        command.extend(ring.arguments())
    if sharpen_process:
        if settings["sharpen_mode"] == "adaptive":
            command.extend(("--static", str(settings["static"]),
                            "--motion", str(settings["motion"])))
        else:
            command.extend(("--static", str(settings["sharpen"]),
                            "--motion", str(settings["sharpen"])))
    if guard:
        command.extend(("--guard-strength", str(args.edge_guard_strength),
                        "--video", args.video, "--ffmpeg", FFMPEG,
                        "--in-w", str(width), "--in-h", str(height)))
    return command


def terminate_all(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def run_stream(args, settings, environment, driver_environment, width, height, out_w,
               out_h, fps, frames, quality, partial):
    decoder_command = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-nostdin", "-i", args.video,
                       "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                       "-vframes", str(frames), "-"]
    needs_depth = settings["mv_path"] == "lowres-depth" or args.force_depth
    ring = None
    output_ring = None
    guide_rings = []
    guard_ring = None
    mfsr_ring = None
    fanout = None
    post_args = post_command(args, settings, width, height, out_w, out_h, frames)
    if settings["io_mode"] == "shared":
        slots = max(2, int(getattr(settings, "get", lambda k, d=None: None)( "_slots", 0) or 0) or 4)
        ring = RingOwner(slots=slots, slot_size=packet_slot_size(
            width, height, depth=needs_depth, mask=settings["responsive"]), prefix="xess-sr")
        # The post stage used to receive xess-vsr stdout through a small
        # anonymous pipe, which serialized the two CPU-heavy processes.  Use
        # a second raw-frame ring when there is a post stage.  MFSR still
        # requires stdin, so keep its legacy pipe path for that expert mode.
        if post_args is not None and not args.five_frame_mfsr:
            output_ring = RingOwner(slots=6, slot_size=out_w * out_h * 3,
                                    prefix="xess-sr-out")
            post_args = post_command(args, settings, width, height, out_w, out_h,
                                     frames, ring=output_ring)
    if args.edge_guard_strength > 0:
        guard_ring = RingOwner(slots=8, slot_size=width * height * 3, prefix="xess-sr-guide")
        guide_rings.append(guard_ring)
        post_args.extend([token.replace("--shm-", "--guide-shm-") for token in guard_ring.arguments()])
    if args.five_frame_mfsr:
        mfsr_ring = RingOwner(slots=8, slot_size=width * height * 3, prefix="xess-mfsr-guide")
        guide_rings.append(mfsr_ring)
    prepare_command, _ = prep_command(args, settings, width, height, frames,
                                      stream=True, ring=ring)
    worker_command = xess_command(args, settings, width, height, out_w, out_h, frames, quality,
                                  stream=True, ring=ring, out_ring=output_ring)
    for command in (decoder_command, prepare_command, worker_command):
        print(f"[run_xess] $ {command_text(command)}")
    processes = []
    try:
        decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, env=environment)
        processes.append(decoder)
        source_stream = decoder.stdout
        if guide_rings:
            fanout = SourceFanout(decoder.stdout, width * height * 3, frames, guide_rings)
            source_stream = fanout.reader
        if args.five_frame_fusion:
            fusion_command = [PY, FUSION, "--width", str(width), "--height", str(height),
                              "--frames", str(frames), "--strength", str(args.fusion_strength)]
            print(f"[run_xess] $ {command_text(fusion_command)}")
            fusion = subprocess.Popen(fusion_command, stdin=source_stream,
                                      stdout=subprocess.PIPE, env=environment)
            source_stream.close()
            source_stream = fusion.stdout
            processes.append(fusion)
        prepare = subprocess.Popen(prepare_command, stdin=source_stream,
                                   stdout=(subprocess.PIPE if ring is None else subprocess.DEVNULL),
                                   env=driver_environment)
        source_stream.close()
        processes.append(prepare)
        worker = subprocess.Popen(worker_command, stdin=(prepare.stdout if ring is None else subprocess.DEVNULL),
                                  stdout=(subprocess.PIPE if output_ring is None else subprocess.DEVNULL),
                                  env=driver_environment)
        if prepare.stdout is not None:
            prepare.stdout.close()
        processes.append(worker)
        video_stream = worker.stdout
        if args.five_frame_mfsr:
            mfsr_command = [PY, MFSR, "--video", args.video, "--ffmpeg", FFMPEG,
                            "--in-w", str(width), "--in-h", str(height),
                            "--out-w", str(out_w), "--out-h", str(out_h),
                            "--frames", str(frames), "--strength", str(args.mfsr_strength),
                            "--detail-boost", str(args.mfsr_detail_boost),
                            "--max-injection", str(args.mfsr_max_injection)]
            mfsr_command.extend([token.replace("--shm-", "--guide-shm-") for token in mfsr_ring.arguments()])
            print(f"[run_xess] $ {command_text(mfsr_command)}")
            mfsr = subprocess.Popen(mfsr_command, stdin=video_stream,
                                    stdout=subprocess.PIPE, env=environment)
            video_stream.close()
            video_stream = mfsr.stdout
            processes.append(mfsr)
        if post_args is not None:
            print(f"[run_xess] $ {command_text(post_args)}")
            post = subprocess.Popen(post_args,
                                    stdin=(subprocess.DEVNULL if output_ring is not None else video_stream),
                                    stdout=subprocess.PIPE, env=environment)
            if video_stream is not None:
                video_stream.close()
            video_stream = post.stdout
            processes.append(post)
        if getattr(args, "preview_dir", ""):
            tap_command = runtime_hooks.preview_tap_command(
                PY, out_w, out_h, fps, frames,
                os.path.abspath(args.preview_dir), FFMPEG, backend="xess",
                preview_scale=getattr(settings, "get", lambda k, d=None: d)(
                    "_preview_scale", 1.0))
            print(f"[run_xess] $ preview-tap -> {args.preview_dir}")
            tap = subprocess.Popen(tap_command, stdin=video_stream,
                                   stdout=subprocess.PIPE, env=environment)
            video_stream.close()
            video_stream = tap.stdout
            processes.append(tap)
        encode_command = encoder_command(args, settings, out_w, out_h, fps, frames, partial)
        print(f"[run_xess] $ {command_text(encode_command)}")
        encoder = subprocess.Popen(encode_command, stdin=video_stream, env=environment)
        video_stream.close()
        processes.append(encoder)
        if fanout is not None:
            fanout.start()
        codes = [process.wait() for process in reversed(processes)]
        if fanout is not None:
            fanout.join()
            print("[source-fanout] " + json.dumps({"decoded_source_frames": fanout.frames_read,
                  "source_decode_count": 1, "guide_consumers": len(guide_rings),
                  "guide_rgb_copy_bytes": frames * width * height * 3 * len(guide_rings),
                  "guide_capacity_bytes": sum(ring.slots * ring.slot_size for ring in guide_rings)}))
        if any(codes):
            die(f"stream pipeline failed; exit codes (reverse order): {codes}")
    except BaseException:
        terminate_all(processes)
        raise
    finally:
        if ring is not None:
            ring.close()
        if output_ring is not None:
            output_ring.close()
        if fanout is not None and fanout.thread.is_alive():
            fanout.thread.join(timeout=35)
        for guide_ring in guide_rings:
            guide_ring.close()


def run_chunked(args, settings, environment, driver_environment, workspace,
                width, height, out_w, out_h, fps, frames, quality, partial):
    segments_dir = workspace.mkdir("segments")
    chunk_source = workspace.path("chunk_source.avi")
    concat_list = workspace.path("segments.ffconcat")
    segments = []
    segment_settings = dict(settings)
    segment_settings["io_mode"] = "stream"
    for chunk_index, start in enumerate(range(0, frames, args.chunk_frames)):
        count = min(args.chunk_frames, frames - start)
        segment = workspace.path(f"segments/sr_{chunk_index:06d}.mp4")
        print(f"[run_xess] chunk {chunk_index + 1}: source frames {start}..{start + count - 1}")
        run(extract_lossless_command(FFMPEG, args.video, os.fspath(chunk_source),
                                     start, count, fps),
            environment=environment)
        segment_args = copy.copy(args)
        segment_args.video = os.fspath(chunk_source)
        run_stream(segment_args, segment_settings, environment, driver_environment,
                   width, height, out_w, out_h, fps, count, quality, segment)
        segments.append(segment)
        Path(chunk_source).unlink(missing_ok=True)
    write_concat_list(concat_list, segments)
    run(concat_command(FFMPEG, os.fspath(concat_list), args.video, os.fspath(partial),
                       frames, fps),
        environment=environment)


def run_file(args, settings, environment, driver_environment, workspace, width, height,
             out_w, out_h, fps, frames, quality, partial):
    raw = workspace.path("frames.raw")
    mv_dir = workspace.mkdir("mvs")
    depth_dir = workspace.mkdir("depth")
    mask_dir = workspace.mkdir("mask")
    out_raw = workspace.path("out.raw")
    debug_dir = workspace.mkdir("debug") if args.debug_prep else ""
    run([FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-i", args.video, "-an",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-vframes", str(frames), raw], environment=environment)
    prepare_command, _ = prep_command(args, settings, width, height, frames,
                                      stream=False, raw=os.fspath(raw), mv_dir=os.fspath(mv_dir),
                                      depth_dir=os.fspath(depth_dir), mask_dir=os.fspath(mask_dir),
                                      debug_dir=os.fspath(debug_dir) if debug_dir else "")
    run(prepare_command, environment=driver_environment)
    reset = os.path.join(mv_dir, "reset_frames.txt")
    run(xess_command(args, settings, width, height, out_w, out_h, frames, quality, stream=False,
                     raw=os.fspath(raw), mv_dir=os.fspath(mv_dir), depth_dir=os.fspath(depth_dir),
                     mask_dir=os.fspath(mask_dir), out_raw=os.fspath(out_raw), reset=reset),
        environment=driver_environment)
    processes = []
    try:
        with open(out_raw, "rb") as source:
            video_stream = source
            command = post_command(args, settings, width, height, out_w, out_h, frames)
            if command is not None:
                print(f"[run_xess] $ {command_text(command)}")
                process = subprocess.Popen(command, stdin=video_stream, stdout=subprocess.PIPE,
                                           env=environment)
                video_stream = process.stdout
                processes.append(process)
            command = encoder_command(args, settings, out_w, out_h, fps, frames, partial)
            print(f"[run_xess] $ {command_text(command)}")
            encoder = subprocess.Popen(command, stdin=video_stream, env=environment)
            if video_stream is not source:
                video_stream.close()
            processes.append(encoder)
            codes = [process.wait() for process in reversed(processes)]
            if any(codes):
                die(f"file pipeline postprocess failed; exit codes (reverse order): {codes}")
    except BaseException:
        terminate_all(processes)
        raise


def main():
    parser = argparse.ArgumentParser(description="XeSS SR 1.2 video upscaler")
    parser.add_argument("video")
    parser.add_argument("scale", nargs="?", type=float, default=1.0)
    parser.add_argument("--preset", choices=("fast", "balanced", "quality"), default="fast")
    parser.add_argument("--quality", type=int, default=-1)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--flow-mode", choices=("auto", "dis-fast", "dis-occlusion", "sea-raft-single", "sea-raft"), default="auto")
    parser.add_argument("--mv-path", choices=("auto", "highres", "lowres-depth"), default="auto")
    parser.add_argument("--responsive-mask", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--responsive-max", type=float, default=0.8)
    parser.add_argument("--depth-model", default=DEPTH_MODEL)
    parser.add_argument("--depth-device", default="GPU")
    parser.add_argument("--depth-temporal", type=float, default=0.25)
    parser.add_argument("--flow-consistency", type=float, default=1.5)
    parser.add_argument("--mv-dilate", type=int, default=1)
    parser.add_argument("--depth-edge", type=float, default=0.04)
    parser.add_argument("--force-depth", action="store_true")
    parser.add_argument("--sharpen", type=float, default=None)
    parser.add_argument("--sharpen-mode", choices=("auto", "off", "fixed", "adaptive"), default="auto")
    parser.add_argument("--sharpen-static", type=float, default=None)
    parser.add_argument("--sharpen-motion", type=float, default=None)
    parser.add_argument("--five-frame-fusion", action="store_true",
                        help="motion-compensated 5-frame source fusion (stream/shared only)")
    parser.add_argument("--fusion-strength", type=float, default=0.35)
    parser.add_argument("--five-frame-mfsr", action="store_true",
                        help="project 5 source frames onto the output grid and inject reliable detail")
    parser.add_argument("--mfsr-strength", type=float, default=1.80)
    parser.add_argument("--mfsr-detail-boost", type=float, default=0.90)
    parser.add_argument("--mfsr-max-injection", type=float, default=22.0)
    parser.add_argument("--edge-guard-strength", type=float, default=0.75,
                        help="suppress XeSS vertical edge ringing; 0 disables it")
    parser.add_argument("--tier", default="",
                        help="public tier: fast | quality (preset registry)")
    parser.add_argument("--preview-dir", default="",
                        help="streaming fMP4 preview session directory")
    parser.add_argument("--memory-strategy", choices=("auto", "conservative"),
                        default="auto")
    parser.add_argument("--post-threads", type=int, default=4,
                        help="parallel SR postprocess workers (default: 4)")
    parser.add_argument("--io-mode", choices=("auto", "stream", "chunked", "file"), default="auto")
    parser.add_argument("--chunk-frames", type=int, default=48)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--allow-system-drive-temp", action="store_true")
    parser.add_argument("--reserve-free-gb", type=float, default=None)
    parser.add_argument("--max-temp-gb", type=float, default=None)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--debug-prep", action="store_true")
    parser.add_argument("--encoder-preset", default="slow")
    parser.add_argument("--encoder", choices=("auto", "h264_qsv", "hevc_qsv", "libx264", "libx265", "ffv1"), default=None)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stage-timing", action="store_true",
                        help="emit one [timing] JSON line per pipeline component "
                             "(also enabled by XESS_STAGE_TIMING=1)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from offline_encoders import configure_cli_encoder
    try:
        configure_cli_encoder(args)
    except ValueError as exc:
        parser.error(str(exc))

    for path, label in ((PY, "Python"), (FFMPEG, "ffmpeg.exe"), (XESS, "xess-vsr.exe"),
                        (FLOW, "flow.py"), (PREPARE, "prepare_sr.py"), (POST, "sr_postprocess.py")):
        if not path or not os.path.isfile(path):
            die(f"missing {label}: {path}")
    if not os.path.isfile(args.video):
        die(f"video not found: {args.video}")
    if args.scale <= 0 or not 0 <= args.crf <= 51 or not 0 <= args.responsive_max <= 1:
        die("invalid scale/CRF/responsive maximum")
    if args.sharpen is not None and not 0 <= args.sharpen <= 1:
        die("--sharpen must be in 0..1")
    if not 0 <= args.fusion_strength <= 1:
        die("--fusion-strength must be in 0..1")
    if not 0 <= args.mfsr_strength <= 8 or not 0 <= args.mfsr_detail_boost <= 4:
        die("invalid MFSR strength/detail boost")
    if not 0 <= args.mfsr_max_injection <= 128:
        die("--mfsr-max-injection must be in 0..128")
    if not 0 <= args.edge_guard_strength <= 1:
        die("--edge-guard-strength must be in 0..1")
    if not 1 <= args.post_threads <= 16:
        die("--post-threads must be in 1..16")
    if args.five_frame_fusion and args.five_frame_mfsr:
        die("choose either --five-frame-fusion or --five-frame-mfsr, not both")
    if args.five_frame_fusion and not os.path.isfile(FUSION):
        die(f"missing five_frame_fusion.py: {FUSION}")
    if args.five_frame_mfsr and not os.path.isfile(MFSR):
        die(f"missing five_frame_mfsr.py: {MFSR}")
    if not 8 <= args.chunk_frames <= 600:
        die("--chunk-frames must be in 8..600")

    try:
        metadata = probe_cfr(args.video, FFMPEG, max_frames=args.frames)
    except RuntimeError as exc:
        die(str(exc))
    width, height, fps = int(metadata["width"]), int(metadata["height"]), float(metadata["fps"])
    frames = min(int(metadata["frames"]), args.frames) if args.frames > 0 else int(metadata["frames"])
    if width <= 0 or height <= 0 or fps <= 0 or frames < 2:
        die(f"invalid media metadata: {width}x{height}, {fps}, {frames}")
    out_w = int(round(width * args.scale / 16) * 16)
    out_h = int(round(height * args.scale / 16) * 16)
    _tier = runtime_hooks.tier_settings_sr(args.tier, args.scale)
    if _tier and args.quality < 0:
        quality = int(_tier["_quality_hint"])
    else:
        quality = args.quality if args.quality >= 0 else quality_for(args.scale)
    if not 0 <= quality <= 6:
        die("--quality must be in 0..6")
    settings = resolve_settings(args)
    tier_override = _tier
    if tier_override and hasattr(args, "edge_guard_strength")             and tier_override.get("guard_strength") is not None:
        # the post chain reads args.edge_guard_strength; make the tier stick
        args.edge_guard_strength = float(tier_override["guard_strength"])
    if tier_override:
        for key in ("mv_path", "sharpen_mode", "static", "motion"):
            if tier_override.get(key) is not None:
                settings[key] = tier_override[key]
        settings["responsive"] = bool(tier_override.get("responsive", True))
        settings["guard_strength"] = tier_override.get("guard_strength",
                                                       settings.get("guard_strength", 0.75))
        if tier_override.get("post_threads"):
            args.post_threads = max(1, int(tier_override["post_threads"]))
        print(f"[run_xess] tier {args.tier}: guard={settings['guard_strength']}, "
              f"post-threads={args.post_threads}")
    ok_budget, budget_report = runtime_hooks.preflight_budget(
        os.environ.get("XESS_RT_WORK",
                       os.path.join(ROOT, "..", "..", "..", "work")),
        "保守" if getattr(args, "memory_strategy", "auto") == "conservative" else "自动",
        width, height, out_w, out_h, frames, fps,
        depth_on=settings["mv_path"] == "lowres-depth",
        post_threads=args.post_threads,
        extra_fixed={"output_ring_slots": 6})
    print(f"[run_xess] memory: {budget_report['message']} "
          f"(queue={budget_report['queue_slots']})")
    print("[run_xess] budget: "
      + json.dumps(budget_report.get("items", {}), ensure_ascii=False))
    if not ok_budget:
        die(f"资源预算不足，已拒绝启动：{budget_report['message']}")
    settings["_slots"] = int(budget_report["queue_slots"])
    settings["_preview_scale"] = float(budget_report.get("preview_scale", 1.0))
    settings["io_mode"] = resolve_io_mode(settings["io_mode"], height, out_h)
    if (args.five_frame_fusion or args.five_frame_mfsr) and settings["io_mode"] not in ("stream", "shared"):
        die("five-frame processing currently requires --io-mode stream/shared")
    needs_depth = settings["mv_path"] == "lowres-depth" or args.force_depth
    if needs_depth and not os.path.isfile(args.depth_model):
        die(f"depth model missing: {args.depth_model}")

    output_dir = os.path.abspath(args.out_dir or os.path.dirname(os.path.abspath(args.video)))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    fusion_suffix = "_mfsr5" if args.five_frame_mfsr else ("_5f" if args.five_frame_fusion else "")
    output = os.path.join(output_dir, f"{base}_xess_sr12_{args.preset}{fusion_suffix}_{args.scale:g}x_{out_w}x{out_h}.mp4")
    if os.environ.get("XESS_OUTPUT_CONTAINER") == ".mkv":
        output = str(Path(output).with_suffix(".mkv"))
    partial = partial_output_path(output)
    estimate = estimate_sr_bytes(width, height, out_w, out_h, frames,
                                 io_mode=settings["io_mode"], include_depth=needs_depth,
                                 include_mask=settings["responsive"],
                                 chunk_frames=args.chunk_frames)
    try:
        workspace = create_workspace(kind="sr12", explicit_work_dir=args.work_dir,
                                     output_dir=output_dir, package_dir=ROOT,
                                     input_path=args.video,
                                     allow_system_drive=args.allow_system_drive_temp,
                                     reserve_free_gb=args.reserve_free_gb,
                                     max_temp_gb=args.max_temp_gb,
                                     estimated_bytes=estimate, keep=args.keep,
                                     label="run_xess")
    except WorkdirError as exc:
        die(str(exc))
    environment = workspace.child_environment()
    driver_environment = workspace.driver_environment(environment)
    stage_timing = timing_requested(args.stage_timing, environment=environment)
    if stage_timing:
        for child_env in (environment, driver_environment):
            child_env["XESS_STAGE_TIMING"] = "1"
        print("[run_xess] stage timing enabled (--stage-timing / XESS_STAGE_TIMING)")
    timer = StageTimer(stage_timing)
    print(f"[run_xess] {width}x{height} {fps:g}fps {frames} frames -> {out_w}x{out_h}; "
          f"preset={args.preset}, flow={settings['flow']}, mv={settings['mv_path']}, "
          f"mask={settings['responsive']}, io={settings['io_mode']}, sharpen={settings['sharpen_mode']}")
    if args.five_frame_fusion:
        print(f"[run_xess] five-frame fusion: on, strength={args.fusion_strength:g}")
    if args.five_frame_mfsr:
        print(f"[run_xess] five-frame MFSR: on, strength={args.mfsr_strength:g}, "
              f"detail={args.mfsr_detail_boost:g}, clamp={args.mfsr_max_injection:g}")
    if args.edge_guard_strength > 0:
        print(f"[run_xess] vertical ringing guard: on, strength={args.edge_guard_strength:g}")
    succeeded = False
    try:
        if args.dry_run:
            print("[run_xess] dry-run complete; no media processing started")
            succeeded = True
            return
        if settings["io_mode"] in ("stream", "shared"):
            with timer.span("total"):
                run_stream(args, settings, environment, driver_environment, width, height,
                           out_w, out_h, fps, frames, quality, partial)
        elif settings["io_mode"] == "chunked":
            with timer.span("total"):
                run_chunked(args, settings, environment, driver_environment, workspace,
                            width, height, out_w, out_h, fps, frames, quality, partial)
        else:
            with timer.span("total"):
                run_file(args, settings, environment, driver_environment, workspace, width,
                         height, out_w, out_h, fps, frames, quality, partial)
        if stage_timing:
            print(f"[timing] component=run-xess {{\"total_s\":"
                  f"{round(timer.totals.get('total', 0.0), 3)}, \"preset\": \"{args.preset}\", "
                  f"\"io\": \"{settings['io_mode']}\", \"frames\": {frames}}}",
                  file=sys.stderr, flush=True)
        try:
            report = validate_output(
                python=PY, flow_script=FLOW, ffmpeg=FFMPEG, output=os.fspath(partial),
                source=args.video, expected_width=out_w, expected_height=out_h,
                expected_fps=fps, expected_frames=frames,
            )
        except MediaValidationError as exc:
            die(f"output validation failed: {exc}")
        print(f"[run_xess] validation: {report['width']}x{report['height']}, "
              f"{report['frames']} frames, {report['fps']:g} fps, audio={report['audio']}")
        finalize_output(partial, output)
        workspace.mark_complete()
        succeeded = True
    finally:
        if not succeeded and Path(partial).is_file():
            Path(partial).unlink()
        workspace.cleanup(label="run_xess")
    print(f"[run_xess] complete: {output}")


if __name__ == "__main__":
    main()
