"""Real runtime integration matrix. Invoke under the shared GPU lease."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--encoders-only", action="store_true")
    parser.add_argument("--encoder-matrix", action="store_true")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--use-source", action="store_true", help="Use the original complete video; its frame count must match --frames")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--five-frame", action="store_true")
    parser.add_argument("--anti-stripe", action="store_true")
    parser.add_argument("--routes", nargs="+", default=["gpu-block", "cpu-dis", "intel-vpl-ai", "gpu-dis", "amd-of"])
    parser.add_argument("--encoders", nargs="+", default=["h264_qsv", "hevc_qsv", "libx264", "libx265", "ffv1"])
    parser.add_argument("--modes", nargs="+", default=["sr", "fg", "sr-fg"])
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    fixture = args.source if args.use_source else args.work / f"input{args.frames}.mp4"
    ffmpeg = args.runtime / "media/ffmpeg.exe"
    if not args.use_source and not fixture.exists():
        # Re-encoded test fixture, not a visual-quality reference. Cutting an
        # arbitrary B-frame packet range by stream-copy can leave a PTS hole.
        subprocess.run([str(ffmpeg), "-v", "error", "-i", str(args.source), "-frames:v", str(args.frames),
                        "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "fast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-bf", "0", "-c:a", "copy",
                        str(fixture)], check=True)
    # A shorter audio tail must not silently truncate our fixed-frame fixture.
    count = subprocess.check_output([str(args.runtime / "probe/ffprobe.exe"), "-v", "error",
        "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(fixture)], timeout=60).decode().strip()
    if int(count) != args.frames:
        raise RuntimeError(f"测试素材帧数错误：请求 {args.frames}，实际 {count}；不启动测试矩阵")
    commands = ([(backend, mode, "auto") for backend in
                 ("gpu-block", "cpu-dis", "intel-vpl-ai", "gpu-dis", "amd-of") for mode in ("sr", "fg", "sr-fg")]
                if not args.encoders_only else
                [("cpu-dis", "sr", enc) for enc in ("hevc_qsv", "libx264", "libx265", "ffv1")])
    if args.encoder_matrix:
        commands = [(route, mode, enc) for route in args.routes for mode in args.modes for enc in args.encoders]
    results = []
    for backend, mode, encoder in commands:
        case = args.work / (backend + "-" + mode + "-" + encoder)
        case.mkdir()
        request = dict(schema_version="xess.offline.request.v1", input=str(fixture.resolve()),
                       output=str((case / ("output.mkv" if encoder == "ffv1" else "output.mp4")).resolve()),
                       backend=backend, mode=mode, scale=args.scale, encoder=encoder, depth="ai",
                       sharpen=args.sharpen, five_frame=args.five_frame, anti_stripe=args.anti_stripe)
        request_path = case / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        command = [str(args.runtime / "python/python.exe"), str(args.tree / "pipeline/offline_toolbox.py"),
                   "run", "--request", str(request_path), "--runtime-root", str(args.runtime),
                   "--work-root", str(case / "work"), "--status-file", str(case / "status.json")]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", TEMP=str(case), TMP=str(case))
        start = time.monotonic()
        with (case / "wrapper.log").open("wb") as stream:
            try:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        env=env, cwd=args.tree, timeout=180)
                rc = result.returncode
            except subprocess.TimeoutExpired:
                # Outer GPU lease will kill its owned process tree at deadline.
                # Do not start another GPU job while a failed worker may remain.
                raise RuntimeError("wrapper超时；立即停止矩阵，交给外层lease收尾")
        status = json.loads((case / "status.json").read_text(encoding="utf-8")) if (case / "status.json").exists() else {}
        item = dict(backend=backend, mode=mode, encoder=encoder, rc=rc,
                    seconds=time.monotonic()-start, status=status.get("status"), error=status.get("error"),
                    output=request["output"], output_exists=Path(request["output"]).is_file(), command=command)
        reports = list((case / "work").glob("job-*/media-result.json"))
        if reports:
            media = json.loads(reports[0].read_text(encoding="utf-8"))
            item.update(actual_frames=media["applied"]["output_frames"], pts_gate=media["pts_gate"],
                        lossless_gate=media.get("ffv1_rgb_lossless_gate"), boundaries=media["boundaries"])
        if item["status"] == "complete" and item["output_exists"]:
            actual = json.loads(subprocess.check_output([str(args.runtime / "probe/ffprobe.exe"),
                "-v", "error", "-count_frames", "-show_entries", "stream=codec_type,codec_name,pix_fmt,nb_read_frames",
                "-of", "json", request["output"]], timeout=60))
            video = next(s for s in actual["streams"] if s["codec_type"] == "video")
            expected_codec = {"auto": "h264", "h264_qsv": "h264", "hevc_qsv": "hevc", "libx264": "h264", "libx265": "hevc", "ffv1": "ffv1"}[encoder]
            expected_frames = args.frames if mode == "sr" else args.frames*2-(backend != "intel-vpl-ai")
            item.update(actual_frames=int(video["nb_read_frames"]), actual_codec=video["codec_name"],
                        pixel_format=video.get("pix_fmt"), audio=any(s["codec_type"]=="audio" for s in actual["streams"]))
            if item["actual_frames"] != expected_frames or item["actual_codec"] != expected_codec:
                item["rc"] = 1
                item["error"] = "independent codec/frame-count gate failed"
            if backend in ("gpu-block", "gpu-dis", "amd-of"):
                native_files = list((case / "work").glob("job-*/native.json"))
                native = json.loads(native_files[0].read_text()) if native_files else {}
                effects, post = native.get("sr_effects", {}), native.get("terminal_post", {})
                item["sr_effects"] = effects
                item["terminal_post"] = post
                wanted_frames = args.frames if args.five_frame or args.anti_stripe else 0
                if (effects.get("five_frame") != args.five_frame or effects.get("anti_stripe") != args.anti_stripe
                        or effects.get("frames") != wanted_frames
                        or post.get("requested") != ("on" if args.sharpen else "off")
                        or post.get("dispatch_count") != (expected_frames if args.sharpen else 0)):
                    item["rc"] = 1
                    item["error"] = "native effect application/count gate failed"
        results.append(item)
        (args.work / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in item.items() if k != "command"}, ensure_ascii=False), flush=True)
    return 0 if all(r["rc"] == 0 and r["status"] == "complete" and r["output_exists"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
