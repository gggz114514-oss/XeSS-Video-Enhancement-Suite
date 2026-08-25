#!/usr/bin/env python3
"""One-pass streamed XeSS SR 1.2 -> independent FG 1.2 -> encode pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import run_fg as fg
import run_xess as sr
from chunked_media import concat_command, extract_lossless_command, write_concat_list
from media_validation import MediaValidationError, validate_output
from shm_ring import RingOwner, packet_slot_size
from workdir_guard import (WorkdirError, create_workspace, estimate_fg_bytes,
                           estimate_sr_bytes, finalize_output, partial_output_path)


ROOT = os.path.dirname(os.path.abspath(__file__))


def die(message):
    print(f"[pipeline] error: {message}", file=sys.stderr)
    raise SystemExit(1)


def command_text(command):
    return subprocess.list2cmdline([os.fspath(item) for item in command])


def terminate_all(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def run_chunked_pipeline(args, workspace, environment, width, height, out_w, out_h,
                         fps, frames, quality, partial):
    segments_dir = workspace.mkdir("segments")
    chunk_source = workspace.path("chunk_source.avi")
    concat_list = workspace.path("segments.ffconcat")
    segments = []
    for chunk_index, start in enumerate(range(0, frames, args.chunk_frames)):
        count = min(args.chunk_frames, frames - start)
        window_five = (args.fg_motion_window == "5" or
                       (args.fg_motion_window == "auto" and args.fg_preset != "fast"))
        motion_window = 5 if window_five else 2
        (actual_start, actual_count, slice_start, slice_count,
         left, right) = fg.chunk_layout(start, count, frames, motion_window)
        sr.run(extract_lossless_command(sr.FFMPEG, args.video, os.fspath(chunk_source),
                                        actual_start, actual_count, fps),
               environment=environment)
        command = [
            sr.PY, os.path.abspath(__file__), os.fspath(chunk_source),
            "--scale", str(args.scale), "--quality", str(quality),
            "--sr-preset", args.sr_preset, "--fg-preset", args.fg_preset,
            "--sr-flow-mode", args.sr_flow_mode, "--fg-flow-mode", args.fg_flow_mode,
            "--mv-path", args.mv_path, "--responsive-mask", args.responsive_mask,
            "--depth", args.depth, "--depth-model", args.depth_model,
            "--depth-device", args.depth_device, "--depth-temporal", str(args.depth_temporal),
            "--flow-consistency", str(args.flow_consistency), "--mv-dilate", str(args.mv_dilate),
            "--depth-edge", str(args.depth_edge), "--responsive-max", str(args.responsive_max),
            "--fg-motion-window", args.fg_motion_window,
            "--temporal-motion-strength", str(args.temporal_motion_strength),
            "--temporal-depth-strength", str(args.temporal_depth_strength),
            "--final-sharpen", args.final_sharpen, "--sharpen-mode", args.sharpen_mode,
            "--frames", str(actual_count), "--out-dir", os.fspath(segments_dir),
            "--device", str(args.device), "--io-mode", "stream",
            "--work-dir", os.fspath(workspace.root),
            "--crf", str(args.crf), "--encoder-preset", args.encoder_preset,
            "--slice-start", str(slice_start), "--slice-count", str(slice_count),
        ]
        if args.allow_system_drive_temp:
            command.append("--allow-system-drive-temp")
        if args.reserve_free_gb is not None:
            command.extend(("--reserve-free-gb", str(args.reserve_free_gb)))
        if args.max_temp_gb is not None:
            command.extend(("--max-temp-gb", str(args.max_temp_gb)))
        if args.allow_overlay:
            command.append("--allow-overlay")
        if args.verbose:
            command.append("--verbose")
        print(f"[pipeline] chunk {chunk_index + 1}: source frames {actual_start}.."
              f"{actual_start + actual_count - 1}, context={left}+{right}, "
              f"output-slice={slice_start}+{slice_count}")
        sr.run(command, environment=environment)
        child_name = (f"chunk_source_xess_pipeline_sr12-{args.sr_preset}_"
                      f"fg12-{args.fg_preset}_{out_w}x{out_h}_{fps * 2:g}fps.mp4")
        child_output = Path(segments_dir, child_name)
        if not child_output.is_file():
            die(f"chunk output is missing: {child_output}")
        segment = workspace.path(f"segments/pipeline_{chunk_index:06d}.mp4")
        os.replace(child_output, segment)
        segments.append(segment)
        Path(chunk_source).unlink(missing_ok=True)
    write_concat_list(concat_list, segments)
    sr.run(concat_command(sr.FFMPEG, os.fspath(concat_list), args.video,
                          os.fspath(partial), frames * 2 - 1, fps * 2.0),
           environment=environment)


def main():
    parser = argparse.ArgumentParser(description="Stream XeSS SR 1.2 into independent FG 1.2")
    parser.add_argument("video")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--quality", type=int, default=-1)
    parser.add_argument("--sr-preset", choices=("fast", "balanced", "quality"), default="fast")
    parser.add_argument("--fg-preset", choices=("fast", "balanced", "quality"), default="fast")
    parser.add_argument("--sr-flow-mode", choices=("auto", "dis-fast", "dis-occlusion", "sea-raft-single", "sea-raft"), default="auto")
    parser.add_argument("--fg-flow-mode", choices=("auto", "dis-fast", "dis-occlusion", "sea-raft-single", "sea-raft"), default="auto")
    parser.add_argument("--mv-path", choices=("auto", "highres", "lowres-depth"), default="auto")
    parser.add_argument("--responsive-mask", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--depth", choices=("ai", "constant"), default="ai")
    parser.add_argument("--depth-model", default=sr.DEPTH_MODEL)
    parser.add_argument("--depth-device", default="GPU")
    parser.add_argument("--depth-temporal", type=float, default=0.25)
    parser.add_argument("--flow-consistency", type=float, default=1.5)
    parser.add_argument("--mv-dilate", type=int, default=1)
    parser.add_argument("--depth-edge", type=float, default=0.04)
    parser.add_argument("--fg-motion-window", choices=("auto", "2", "5"), default="auto")
    parser.add_argument("--temporal-motion-strength", type=float, default=0.65)
    parser.add_argument("--temporal-depth-strength", type=float, default=0.18)
    parser.add_argument("--responsive-max", type=float, default=0.8)
    parser.add_argument("--final-sharpen", default="auto", help="auto or 0..1")
    parser.add_argument("--sharpen-mode", choices=("auto", "off", "fixed", "adaptive"), default="auto")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--io-mode", choices=("auto", "stream", "chunked"), default="auto")
    parser.add_argument("--chunk-frames", type=int, default=48)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--allow-system-drive-temp", action="store_true")
    parser.add_argument("--reserve-free-gb", type=float, default=None)
    parser.add_argument("--max-temp-gb", type=float, default=None)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--encoder-preset", default="slow")
    parser.add_argument("--allow-overlay", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drop-first-output", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--slice-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--slice-count", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    for path, label in ((sr.PY, "Python"), (sr.FFMPEG, "ffmpeg.exe"),
                        (sr.XESS, "xess-vsr.exe"), (fg.XESS_FG, "xess-fg.exe")):
        if not path or not os.path.isfile(path):
            die(f"missing {label}: {path}")
    if not os.path.isfile(args.video) or args.scale <= 0 or not 0 <= args.crf <= 51:
        die("invalid input/scale/CRF")
    if not 8 <= args.chunk_frames <= 600:
        die("--chunk-frames must be in 8..600")
    if args.depth == "ai" and not os.path.isfile(args.depth_model):
        die(f"depth model missing: {args.depth_model}")

    metadata = json.loads(sr.run([sr.PY, sr.FLOW, args.video, "--probe-only"],
                                 capture_output=True, text=True).stdout)
    width, height, fps = int(metadata["width"]), int(metadata["height"]), float(metadata["fps"])
    frames = min(int(metadata["frames"]), args.frames) if args.frames > 0 else int(metadata["frames"])
    if width <= 0 or height <= 0 or fps <= 0 or frames < 2:
        die(f"invalid media metadata: {width}x{height}, {fps}, {frames}")
    out_w = int(round(width * args.scale / 16) * 16)
    out_h = int(round(height * args.scale / 16) * 16)
    quality = args.quality if args.quality >= 0 else sr.quality_for(args.scale)
    if not 0 <= quality <= 6:
        die("--quality must be in 0..6")
    if args.io_mode == "chunked":
        transport = "chunked"
    elif args.io_mode == "auto" and max(height, out_h) > 720:
        transport = "shared"
    else:
        transport = "stream"

    sr_args = SimpleNamespace(
        preset=args.sr_preset, flow_mode=args.sr_flow_mode, mv_path=args.mv_path,
        responsive_mask=args.responsive_mask, sharpen=0.0, sharpen_mode="off",
        sharpen_static=None, sharpen_motion=None, force_depth=False,
        depth_temporal=args.depth_temporal, flow_consistency=args.flow_consistency,
        mv_dilate=args.mv_dilate, depth_edge=args.depth_edge,
        responsive_max=args.responsive_max, depth_model=args.depth_model,
        depth_device=args.depth_device, device=args.device, verbose=args.verbose,
        io_mode=transport,
    )
    fg_args = SimpleNamespace(
        preset=args.fg_preset, flow_mode=args.fg_flow_mode, final_sharpen=None,
        sharpen_mode=args.sharpen_mode, sharpen_static=None, sharpen_motion=None,
        depth=args.depth, depth_model=args.depth_model, depth_device=args.depth_device,
        depth_temporal=args.depth_temporal, flow_consistency=args.flow_consistency,
        mv_dilate=args.mv_dilate, depth_edge=args.depth_edge, device=args.device,
        motion_window=args.fg_motion_window,
        temporal_motion_strength=args.temporal_motion_strength,
        temporal_depth_strength=args.temporal_depth_strength,
        verbose=args.verbose, allow_overlay=args.allow_overlay, video=args.video,
        encoder_preset=args.encoder_preset, crf=args.crf,
        io_mode=transport, capture_mode="direct",
        overlay_mask="",
    )
    sr_settings = sr.resolve_settings(sr_args)
    fg_settings = fg.resolve_settings(fg_args)
    if args.final_sharpen != "auto":
        try:
            final_strength = float(args.final_sharpen)
        except ValueError:
            die("--final-sharpen must be auto or a number in 0..1")
        if not 0 <= final_strength <= 1:
            die("--final-sharpen must be in 0..1")
        fg_settings["sharpen"] = final_strength
        if final_strength == 0:
            fg_settings["sharpen_mode"] = "off"

    output_dir = os.path.abspath(args.out_dir or os.path.dirname(os.path.abspath(args.video)))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    output = os.path.join(output_dir,
                          f"{base}_xess_pipeline_sr12-{args.sr_preset}_fg12-{args.fg_preset}_{out_w}x{out_h}_{fps * 2:g}fps.mp4")
    partial = partial_output_path(output)
    sr_depth = sr_settings["mv_path"] == "lowres-depth"
    estimate = estimate_sr_bytes(width, height, out_w, out_h, frames, io_mode=transport,
                                 include_depth=sr_depth, include_mask=sr_settings["responsive"],
                                 chunk_frames=args.chunk_frames)
    estimate += estimate_fg_bytes(out_w, out_h, frames, io_mode=transport,
                                  include_depth=True, chunk_frames=args.chunk_frames)
    try:
        workspace = create_workspace(kind="pipeline", explicit_work_dir=args.work_dir,
                                     output_dir=output_dir, package_dir=ROOT, input_path=args.video,
                                     allow_system_drive=args.allow_system_drive_temp,
                                     reserve_free_gb=args.reserve_free_gb, max_temp_gb=args.max_temp_gb,
                                     estimated_bytes=estimate, keep=args.keep, label="pipeline")
    except WorkdirError as exc:
        die(str(exc))
    environment = workspace.child_environment()
    driver_environment = workspace.driver_environment(environment)
    print(f"[pipeline] {width}x{height}@{fps:g} -> {out_w}x{out_h}@{fps * 2:g}; "
          f"SR={args.sr_preset}/{sr_settings['flow']}, FG={args.fg_preset}/{fg_settings['flow']}, "
          "independent analysis, one final encode")
    succeeded = False
    processes = []
    sr_ring = None
    fg_ring = None
    try:
        if args.dry_run:
            print("[pipeline] dry-run complete; no media processing started")
            succeeded = True
            return
        if args.io_mode == "chunked":
            run_chunked_pipeline(args, workspace, environment, width, height, out_w, out_h,
                                 fps, frames, quality, partial)
            try:
                report = validate_output(
                    python=sr.PY, flow_script=sr.FLOW, ffmpeg=sr.FFMPEG,
                    output=os.fspath(partial), source=args.video,
                    expected_width=out_w, expected_height=out_h,
                    expected_fps=fps * 2.0, expected_frames=frames * 2 - 1,
                )
            except MediaValidationError as exc:
                die(f"chunked output validation failed: {exc}")
            print(f"[pipeline] validation: {report['width']}x{report['height']}, "
                  f"{report['frames']} frames, {report['fps']:g} fps, audio={report['audio']}")
            finalize_output(partial, output)
            workspace.mark_complete()
            succeeded = True
            print(f"[pipeline] complete: {output}")
            return
        decoder_command = [sr.FFMPEG, "-hide_banner", "-loglevel", "warning", "-nostdin",
                           "-i", args.video, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{width}x{height}", "-vframes", str(frames), "-"]
        if transport == "shared":
            sr_ring = RingOwner(slots=4, slot_size=packet_slot_size(
                width, height, depth=sr_depth, mask=sr_settings["responsive"]),
                prefix="xess-pipeline-sr")
            fg_ring = RingOwner(slots=4, slot_size=packet_slot_size(
                out_w, out_h, depth=True, mask=False), prefix="xess-pipeline-fg")
        sr_prepare, _ = sr.prep_command(sr_args, sr_settings, width, height,
                                        frames, stream=True, ring=sr_ring)
        sr_worker = sr.xess_command(sr_args, sr_settings, width, height, out_w, out_h,
                                    frames, quality, stream=True, ring=sr_ring)
        fg_prepare = fg.prep_command(fg_args, fg_settings, out_w, out_h,
                                     frames, stream=True, ring=fg_ring)
        fg_worker = fg.worker_command(fg_args, out_w, out_h, fps, frames,
                                      stream=True, ring=fg_ring)
        for command in (decoder_command, sr_prepare, sr_worker, fg_prepare, fg_worker):
            print(f"[pipeline] $ {command_text(command)}")

        decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, env=environment)
        processes.append(decoder)
        sr_prep_process = subprocess.Popen(
            sr_prepare, stdin=decoder.stdout,
            stdout=(subprocess.PIPE if sr_ring is None else subprocess.DEVNULL),
            env=driver_environment)
        decoder.stdout.close(); processes.append(sr_prep_process)
        sr_process = subprocess.Popen(
            sr_worker,
            stdin=(sr_prep_process.stdout if sr_ring is None else subprocess.DEVNULL),
            stdout=subprocess.PIPE, env=driver_environment)
        if sr_prep_process.stdout is not None:
            sr_prep_process.stdout.close()
        processes.append(sr_process)
        fg_prep_process = subprocess.Popen(
            fg_prepare, stdin=sr_process.stdout,
            stdout=(subprocess.PIPE if fg_ring is None else subprocess.DEVNULL),
            env=driver_environment)
        sr_process.stdout.close(); processes.append(fg_prep_process)
        fg_process = subprocess.Popen(
            fg_worker,
            stdin=(fg_prep_process.stdout if fg_ring is None else subprocess.DEVNULL),
            stdout=subprocess.PIPE, env=driver_environment)
        if fg_prep_process.stdout is not None:
            fg_prep_process.stdout.close()
        processes.append(fg_process)
        video_stream = fg_process.stdout
        raw_output_frames = frames * 2 - 1
        trim_start = args.slice_start if args.slice_start else (1 if args.drop_first_output else 0)
        output_frames = args.slice_count if args.slice_count > 0 else raw_output_frames - trim_start
        if trim_start < 0 or output_frames <= 0 or trim_start + output_frames > raw_output_frames:
            die("invalid pipeline raw output slice")
        if fg_settings["sharpen_mode"] == "adaptive" and fg_settings["sharpen"] > 0:
            sharpen_command = [sr.PY, sr.SHARPEN, "--width", str(out_w), "--height", str(out_h),
                               "--frames", str(raw_output_frames), "--static", str(fg_settings["static"]),
                               "--motion", str(fg_settings["motion"])]
            print(f"[pipeline] $ {command_text(sharpen_command)}")
            sharpener = subprocess.Popen(sharpen_command, stdin=video_stream,
                                         stdout=subprocess.PIPE, env=environment)
            video_stream.close(); video_stream = sharpener.stdout; processes.append(sharpener)
        if trim_start or output_frames != raw_output_frames:
            slice_command = [sr.PY, fg.RAW_SLICE,
                             "--frame-bytes", str(out_w * out_h * 3),
                             "--total-frames", str(raw_output_frames),
                             "--start", str(trim_start), "--count", str(output_frames)]
            print(f"[pipeline] $ {command_text(slice_command)}")
            slicer = subprocess.Popen(slice_command, stdin=video_stream,
                                      stdout=subprocess.PIPE, env=environment)
            video_stream.close(); video_stream = slicer.stdout; processes.append(slicer)
        encode_command = fg.encoder_command(fg_args, fg_settings, out_w, out_h,
                                            fps * 2.0, output_frames, partial)
        print(f"[pipeline] $ {command_text(encode_command)}")
        encoder = subprocess.Popen(encode_command, stdin=video_stream, env=environment)
        video_stream.close(); processes.append(encoder)
        codes = [process.wait() for process in reversed(processes)]
        if any(codes):
            die(f"pipeline failed; exit codes (reverse order): {codes}")
        try:
            report = validate_output(
                python=sr.PY, flow_script=sr.FLOW, ffmpeg=sr.FFMPEG,
                output=os.fspath(partial), source=args.video,
                expected_width=out_w, expected_height=out_h,
                expected_fps=fps * 2.0, expected_frames=output_frames,
            )
        except MediaValidationError as exc:
            die(f"output validation failed: {exc}")
        print(f"[pipeline] validation: {report['width']}x{report['height']}, "
              f"{report['frames']} frames, {report['fps']:g} fps, audio={report['audio']}")
        finalize_output(partial, output)
        workspace.mark_complete()
        succeeded = True
    except BaseException:
        terminate_all(processes)
        raise
    finally:
        if sr_ring is not None:
            sr_ring.close()
        if fg_ring is not None:
            fg_ring.close()
        if not succeeded and Path(partial).is_file():
            Path(partial).unlink()
        workspace.cleanup(label="pipeline")
    print(f"[pipeline] complete: {output}")


if __name__ == "__main__":
    main()
