#!/usr/bin/env python3
"""XeSS FG 1.2 portable runner with independent motion/depth analysis."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from chunked_media import concat_command, extract_lossless_command, write_concat_list
from media_validation import MediaValidationError, validate_output
from shm_ring import RingOwner, packet_slot_size
from workdir_guard import (WorkdirError, create_workspace, estimate_fg_bytes,
                           finalize_output, partial_output_path)


ROOT = os.path.dirname(os.path.abspath(__file__))


def first_existing(*paths):
    return next((os.path.abspath(path) for path in paths if path and os.path.isfile(path)), "")


PY = first_existing(os.environ.get("XESS_PYTHON"), os.path.join(ROOT, "python", "python.exe"), sys.executable)
FFMPEG = first_existing(os.environ.get("XESS_FFMPEG"), os.path.join(ROOT, "ffmpeg.exe"), shutil.which("ffmpeg"))
XESS_FG = first_existing(os.environ.get("XESS_FG"), os.path.join(ROOT, "xess-fg.exe"), os.path.join(ROOT, "build", "xess-fg.exe"))
FLOW = os.path.join(ROOT, "flow.py")
PREPARE = os.path.join(ROOT, "prepare_fg.py")
SHARPEN = os.path.join(ROOT, "adaptive_sharpen.py")
RAW_SLICE = os.path.join(ROOT, "raw_frame_slice.py")
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
        print(f"[run_fg] SEA-RAFT has been retired from the mainline; "
              f"flow mode '{flow_mode}' now runs native Fast DIS", flush=True)
        _MIGRATION_PRINTED = True
    return "dis-fast" if flow_mode in LEGACY_FLOW_MODES else flow_mode


def die(message):
    print(f"[run_fg] error: {message}", file=sys.stderr)
    raise SystemExit(1)


def command_text(command):
    return subprocess.list2cmdline([os.fspath(item) for item in command])


def run(command, *, environment=None, **kwargs):
    print(f"[run_fg] $ {command_text(command)}", flush=True)
    result = subprocess.run(command, env=environment, **kwargs)
    if result.returncode:
        detail = result.stderr[-3000:] if getattr(result, "stderr", None) else ""
        die(f"command failed ({result.returncode}): {detail}")
    return result


def resolve_settings(args):
    defaults = {
        "fast": {"bidirectional": False, "mode": "fixed",
                 "sharpen": 0.12, "static": 0.18, "motion": 0.08, "window": 2},
        "balanced": {"bidirectional": False, "mode": "adaptive",
                     "sharpen": 0.16, "static": 0.22, "motion": 0.10, "window": 5},
        "quality": {"bidirectional": False, "mode": "adaptive",
                    "sharpen": 0.18, "static": 0.25, "motion": 0.10, "window": 5},
    }[args.preset]
    flow = "dis"
    bidirectional = defaults["bidirectional"]
    if args.flow_mode != "auto":
        flow, bidirectional = FLOW_MODES[resolve_flow_mode(args.flow_mode)]
    sharpen = defaults["sharpen"] if args.final_sharpen is None else args.final_sharpen
    mode = defaults["mode"] if args.sharpen_mode == "auto" else args.sharpen_mode
    io_mode = args.io_mode
    motion_window = defaults["window"] if args.motion_window == "auto" else int(args.motion_window)
    return {"flow": flow, "bidirectional": bidirectional, "sharpen": sharpen,
            "sharpen_mode": mode,
            "static": defaults["static"] if args.sharpen_static is None else args.sharpen_static,
            "motion": defaults["motion"] if args.sharpen_motion is None else args.sharpen_motion,
            "io_mode": io_mode, "motion_window": motion_window}


def prep_command(args, settings, width, height, frames, *, stream,
                 raw="", mv_dir="", depth_dir="", mask_dir="", debug_dir="", ring=None):
    command = [PY, PREPARE, "--in-w", str(width), "--in-h", str(height),
               "--frames", str(frames), "--engine", settings["flow"],
               "--temporal", str(args.depth_temporal), "--consistency", str(args.flow_consistency),
               "--dilate", str(args.mv_dilate), "--depth-edge", str(args.depth_edge),
               "--motion-window", str(settings["motion_window"]),
               "--temporal-motion-strength", str(args.temporal_motion_strength),
               "--temporal-depth-strength", str(args.temporal_depth_strength),
               "--mv-path", "lowres-depth"]
    if stream:
        command.append("--stream")
        if ring is not None:
            command.extend(ring.arguments())
    else:
        command.extend(("--raw", raw, "--mv-out", mv_dir, "--depth-out", depth_dir))
    if settings["bidirectional"]:
        command.append("--bidirectional")
    if args.depth == "ai":
        command.extend(("--depth-model", args.depth_model, "--depth-device", args.depth_device))
    if args.overlay_mask:
        command.extend(("--overlay-mask", args.overlay_mask))
        if not stream:
            command.extend(("--mask-out", mask_dir))
    if debug_dir:
        command.extend(("--debug-dir", debug_dir))
    return command


def worker_command(args, width, height, fps, frames, *, stream, raw="", mv_dir="",
                   depth_dir="", mask_dir="", generated="", reset="", ring=None):
    command = [XESS_FG, "--width", str(width), "--height", str(height),
               "--frames-count", str(frames), "--fps", str(fps)]
    if stream:
        command.append("--stream")
        if ring is not None:
            command.extend(ring.arguments())
        if args.overlay_mask:
            command.extend(("--ui-mask", "stream"))
    else:
        command.extend(("--frames", raw, "--mv", mv_dir, "--depth", depth_dir,
                        "--out", generated))
        if args.overlay_mask:
            command.extend(("--ui-mask", mask_dir))
        if reset:
            command.extend(("--reset-frames", reset))
    if args.device >= 0:
        command.extend(("--device", str(args.device)))
    if args.verbose:
        command.append("--verbose")
    # Internal callers created before the direct backend was added may not carry
    # this field. Keep direct readback as the safe/default contract everywhere.
    command.extend(("--capture-mode", getattr(args, "capture_mode", "direct")))
    if args.allow_overlay:
        command.append("--allow-overlay")
    return command


def encoder_command(args, settings, width, height, output_fps, output_frames, partial,
                    trim_start=0):
    command = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(output_fps),
               "-i", "-", "-i", args.video, "-map", "0:v", "-map", "1:a?",
               "-c:v", "libx264", "-preset", args.encoder_preset, "-crf", str(args.crf),
               "-pix_fmt", "yuv420p"]
    filters = []
    if trim_start:
        filters.append(f"trim=start_frame={trim_start},setpts=PTS-STARTPTS")
    if settings["sharpen_mode"] == "fixed" and settings["sharpen"] > 0:
        filters.append(f"cas=strength={settings['sharpen']:.4f}")
    if filters:
        command.extend(("-vf", ",".join(filters)))
    command.extend(("-c:a", "copy", "-t", f"{output_frames / output_fps:.9f}",
                    "-frames:v", str(output_frames),
                    os.fspath(partial)))
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


def chunk_layout(start, count, total_frames, motion_window):
    """Return source context and exact raw-output slice for one FG chunk."""
    if start < 0 or count <= 0 or start + count > total_frames:
        raise ValueError("invalid chunk range")
    temporal_context = 2 if motion_window == 5 else 1
    left = min(temporal_context, start)
    right = min(2 if motion_window == 5 else 0,
                total_frames - (start + count))
    actual_start = start - left
    actual_count = count + left + right
    slice_start = 0 if start == 0 else 2 * left - 1
    slice_count = 2 * count - 1 if start == 0 else 2 * count
    return actual_start, actual_count, slice_start, slice_count, left, right


def run_stream(args, settings, environment, driver_environment,
               width, height, fps, frames, partial, trim_start=0,
               output_limit=None):
    raw_output_frames = frames * 2 - 1
    output_frames = (int(output_limit) if output_limit is not None
                     else raw_output_frames - trim_start)
    if trim_start < 0 or output_frames <= 0 or trim_start + output_frames > raw_output_frames:
        die("invalid raw output slice")
    decoder_command = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-nostdin",
                       "-i", args.video, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{width}x{height}", "-vframes", str(frames), "-"]
    ring = None
    if settings["io_mode"] == "shared":
        ring = RingOwner(slots=4, slot_size=packet_slot_size(
            width, height, depth=True, mask=bool(args.overlay_mask)), prefix="xess-fg")
    prepare_command = prep_command(args, settings, width, height, frames,
                                   stream=True, ring=ring)
    fg_command = worker_command(args, width, height, fps, frames, stream=True, ring=ring)
    for command in (decoder_command, prepare_command, fg_command):
        print(f"[run_fg] $ {command_text(command)}")
    processes = []
    try:
        decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, env=environment)
        processes.append(decoder)
        prepare = subprocess.Popen(prepare_command, stdin=decoder.stdout,
                                   stdout=(subprocess.PIPE if ring is None else subprocess.DEVNULL),
                                   env=driver_environment)
        decoder.stdout.close()
        processes.append(prepare)
        fg = subprocess.Popen(fg_command,
                              stdin=(prepare.stdout if ring is None else subprocess.DEVNULL),
                              stdout=subprocess.PIPE,
                              env=driver_environment)
        if prepare.stdout is not None:
            prepare.stdout.close()
        processes.append(fg)
        video_stream = fg.stdout
        if settings["sharpen_mode"] == "adaptive" and settings["sharpen"] > 0:
            sharpen_command = [PY, SHARPEN, "--width", str(width), "--height", str(height),
                               "--frames", str(raw_output_frames), "--static", str(settings["static"]),
                               "--motion", str(settings["motion"])]
            print(f"[run_fg] $ {command_text(sharpen_command)}")
            sharpener = subprocess.Popen(sharpen_command, stdin=video_stream,
                                         stdout=subprocess.PIPE, env=environment)
            video_stream.close()
            video_stream = sharpener.stdout
            processes.append(sharpener)
        if trim_start or output_frames != raw_output_frames:
            slice_command = [PY, RAW_SLICE, "--frame-bytes", str(width * height * 3),
                             "--total-frames", str(raw_output_frames),
                             "--start", str(trim_start), "--count", str(output_frames)]
            print(f"[run_fg] $ {command_text(slice_command)}")
            slicer = subprocess.Popen(slice_command, stdin=video_stream,
                                      stdout=subprocess.PIPE, env=environment)
            video_stream.close()
            video_stream = slicer.stdout
            processes.append(slicer)
        encode_command = encoder_command(args, settings, width, height, fps * 2.0,
                                         output_frames, partial)
        print(f"[run_fg] $ {command_text(encode_command)}")
        encoder = subprocess.Popen(encode_command, stdin=video_stream, env=environment)
        video_stream.close()
        processes.append(encoder)
        codes = [process.wait() for process in reversed(processes)]
        if any(codes):
            die(f"stream pipeline failed; exit codes (reverse order): {codes}")
    except BaseException:
        terminate_all(processes)
        raise
    finally:
        if ring is not None:
            ring.close()


def run_chunked(args, settings, environment, driver_environment, workspace,
                width, height, fps, frames, partial):
    workspace.mkdir("segments")
    chunk_source = workspace.path("chunk_source.avi")
    concat_list = workspace.path("segments.ffconcat")
    segments = []
    segment_settings = dict(settings)
    segment_settings["io_mode"] = "stream"
    for chunk_index, start in enumerate(range(0, frames, args.chunk_frames)):
        count = min(args.chunk_frames, frames - start)
        (actual_start, actual_count, slice_start, slice_count,
         left, right) = chunk_layout(start, count, frames, settings["motion_window"])
        segment = workspace.path(f"segments/fg_{chunk_index:06d}.mp4")
        print(f"[run_fg] chunk {chunk_index + 1}: source frames {actual_start}.."
              f"{actual_start + actual_count - 1}, context={left}+{right}, "
              f"output-slice={slice_start}+{slice_count}")
        run(extract_lossless_command(FFMPEG, args.video, os.fspath(chunk_source),
                                     actual_start, actual_count, fps), environment=environment)
        segment_args = copy.copy(args)
        segment_args.video = os.fspath(chunk_source)
        run_stream(segment_args, segment_settings, environment,
                   driver_environment, width, height, fps, actual_count, segment,
                   trim_start=slice_start, output_limit=slice_count)
        segments.append(segment)
        Path(chunk_source).unlink(missing_ok=True)
    write_concat_list(concat_list, segments)
    run(concat_command(FFMPEG, os.fspath(concat_list), args.video, os.fspath(partial),
                       frames * 2 - 1, fps * 2.0), environment=environment)


def interleave(source_path, generated_path, output_path, frame_bytes, frame_count):
    if os.path.getsize(source_path) != frame_bytes * frame_count:
        die("source raw length mismatch")
    if os.path.getsize(generated_path) != frame_bytes * (frame_count - 1):
        die("generated raw length mismatch")
    with open(source_path, "rb") as source, open(generated_path, "rb") as generated, open(output_path, "wb") as output:
        output.write(source.read(frame_bytes))
        for _ in range(frame_count - 1):
            output.write(generated.read(frame_bytes))
            output.write(source.read(frame_bytes))


def run_file(args, settings, environment, driver_environment,
             workspace, width, height, fps, frames, partial):
    raw = workspace.path("source.rgb")
    mv_dir = workspace.mkdir("mvs")
    depth_dir = workspace.mkdir("depth")
    mask_dir = workspace.mkdir("ui-mask") if args.overlay_mask else ""
    generated = workspace.path("generated.rgb")
    interleaved = workspace.path("interleaved.rgb")
    debug_dir = workspace.mkdir("debug") if args.debug_prep else ""
    run([FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-i", args.video,
         "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-vframes", str(frames), raw], environment=environment)
    run(prep_command(args, settings, width, height, frames, stream=False,
                     raw=os.fspath(raw), mv_dir=os.fspath(mv_dir), depth_dir=os.fspath(depth_dir),
                     mask_dir=os.fspath(mask_dir) if mask_dir else "",
                     debug_dir=os.fspath(debug_dir) if debug_dir else ""),
        environment=driver_environment)
    reset = os.path.join(mv_dir, "reset_frames.txt")
    run(worker_command(args, width, height, fps, frames, stream=False, raw=os.fspath(raw),
                       mv_dir=os.fspath(mv_dir), depth_dir=os.fspath(depth_dir),
                       mask_dir=os.fspath(mask_dir) if mask_dir else "",
                       generated=os.fspath(generated), reset=reset),
        environment=driver_environment)
    interleave(raw, generated, interleaved, width * height * 3, frames)
    with open(interleaved, "rb") as source:
        run(encoder_command(args, settings, width, height, fps * 2.0,
                            frames * 2 - 1, partial), environment=environment, stdin=source)


def main():
    parser = argparse.ArgumentParser(description="XeSS FG 1.2 2x frame generation")
    parser.add_argument("video")
    parser.add_argument("--preset", choices=("fast", "balanced", "quality"), default="fast")
    parser.add_argument("--flow-mode", choices=("auto", "dis-fast", "dis-occlusion", "sea-raft-single", "sea-raft"), default="auto")
    parser.add_argument("--depth", choices=("ai", "constant"), default="ai")
    parser.add_argument("--depth-model", default=DEPTH_MODEL)
    parser.add_argument("--depth-device", default="GPU")
    parser.add_argument("--depth-temporal", type=float, default=0.25)
    parser.add_argument("--flow-consistency", type=float, default=1.5)
    parser.add_argument("--mv-dilate", type=int, default=1)
    parser.add_argument("--depth-edge", type=float, default=0.04)
    parser.add_argument("--motion-window", choices=("auto", "2", "5"), default="auto",
                        help="motion/depth observation window; auto=2 for Fast, 5 otherwise")
    parser.add_argument("--temporal-motion-strength", type=float, default=0.65)
    parser.add_argument("--temporal-depth-strength", type=float, default=0.18)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--overlay-mask", default="", help="reserved optional UI/subtitle mask input")
    parser.add_argument("--final-sharpen", type=float, default=None)
    parser.add_argument("--sharpen-mode", choices=("auto", "off", "fixed", "adaptive"), default="auto")
    parser.add_argument("--sharpen-static", type=float, default=None)
    parser.add_argument("--sharpen-motion", type=float, default=None)
    parser.add_argument("--io-mode", choices=("auto", "stream", "chunked", "file"), default="auto")
    parser.add_argument("--chunk-frames", type=int, default=48)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--allow-system-drive-temp", action="store_true")
    parser.add_argument("--reserve-free-gb", type=float, default=None)
    parser.add_argument("--max-temp-gb", type=float, default=None)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--debug-prep", action="store_true")
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--encoder-preset", default="slow")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--capture-mode", choices=("direct", "window"), default="direct",
                        help="generated-frame capture backend; direct avoids WGC/DPI/overlay issues")
    parser.add_argument("--allow-overlay", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path, label in ((PY, "Python"), (FFMPEG, "ffmpeg.exe"), (XESS_FG, "xess-fg.exe"),
                        (FLOW, "flow.py"), (PREPARE, "prepare_fg.py")):
        if not path or not os.path.isfile(path):
            die(f"missing {label}: {path}")
    if not os.path.isfile(args.video):
        die(f"video not found: {args.video}")
    if args.depth == "ai" and not os.path.isfile(args.depth_model):
        die(f"depth model missing: {args.depth_model}")
    if args.overlay_mask and not os.path.isfile(args.overlay_mask):
        die(f"overlay mask not found: {args.overlay_mask}")
    if args.final_sharpen is not None and not 0 <= args.final_sharpen <= 1:
        die("--final-sharpen must be in 0..1")
    if not 0 <= args.temporal_motion_strength <= 1:
        die("--temporal-motion-strength must be in 0..1")
    if not 0 <= args.temporal_depth_strength <= 0.5:
        die("--temporal-depth-strength must be in 0..0.5")
    if not 0 <= args.crf <= 51:
        die("--crf must be in 0..51")
    if not 8 <= args.chunk_frames <= 600:
        die("--chunk-frames must be in 8..600")

    metadata = json.loads(run([PY, FLOW, args.video, "--probe-only"],
                              capture_output=True, text=True).stdout)
    width, height, fps = int(metadata["width"]), int(metadata["height"]), float(metadata["fps"])
    frames = min(int(metadata["frames"]), args.frames) if args.frames > 0 else int(metadata["frames"])
    if width <= 0 or height <= 0 or fps <= 0 or frames < 2:
        die(f"invalid media metadata: {width}x{height}, {fps}, {frames}")
    settings = resolve_settings(args)
    if settings["io_mode"] == "auto":
        settings["io_mode"] = "shared" if height > 720 else "stream"
    output_dir = os.path.abspath(args.out_dir or os.path.dirname(os.path.abspath(args.video)))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    output = os.path.join(output_dir, f"{base}_xess_fg12_{args.preset}_2x_{fps * 2:g}fps.mp4")
    partial = partial_output_path(output)
    estimate = estimate_fg_bytes(width, height, frames, io_mode=settings["io_mode"],
                                 include_depth=True, chunk_frames=args.chunk_frames)
    try:
        workspace = create_workspace(kind="fg12", explicit_work_dir=args.work_dir,
                                     output_dir=output_dir, package_dir=ROOT, input_path=args.video,
                                     allow_system_drive=args.allow_system_drive_temp,
                                     reserve_free_gb=args.reserve_free_gb, max_temp_gb=args.max_temp_gb,
                                     estimated_bytes=estimate, keep=args.keep, label="run_fg")
    except WorkdirError as exc:
        die(str(exc))
    environment = workspace.child_environment()
    driver_environment = workspace.driver_environment(environment)
    print(f"[run_fg] {width}x{height} {fps:g}fps {frames} frames -> {fps * 2:g}fps; "
          f"preset={args.preset}, flow={settings['flow']}, depth={args.depth}, "
          f"motion-window={settings['motion_window']}, io={settings['io_mode']}, "
          f"sharpen={settings['sharpen_mode']}")
    succeeded = False
    try:
        if args.dry_run:
            print("[run_fg] dry-run complete; no media processing started")
            succeeded = True
            return
        if settings["io_mode"] in ("stream", "shared"):
            run_stream(args, settings, environment, driver_environment,
                       width, height, fps, frames, partial)
        elif settings["io_mode"] == "chunked":
            run_chunked(args, settings, environment, driver_environment,
                        workspace, width, height, fps, frames, partial)
        else:
            run_file(args, settings, environment, driver_environment,
                     workspace, width, height, fps, frames, partial)
        try:
            report = validate_output(
                python=PY, flow_script=FLOW, ffmpeg=FFMPEG, output=os.fspath(partial),
                source=args.video, expected_width=width, expected_height=height,
                expected_fps=fps * 2.0, expected_frames=frames * 2 - 1,
            )
        except MediaValidationError as exc:
            die(f"output validation failed: {exc}")
        print(f"[run_fg] validation: {report['width']}x{report['height']}, "
              f"{report['frames']} frames, {report['fps']:g} fps, audio={report['audio']}")
        finalize_output(partial, output)
        workspace.mark_complete()
        succeeded = True
    finally:
        if not succeeded and Path(partial).is_file():
            Path(partial).unlink()
        workspace.cleanup(label="run_fg")
    print(f"[run_fg] complete: {output}")


if __name__ == "__main__":
    main()
