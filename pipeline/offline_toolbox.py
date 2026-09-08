"""Five-route offline control plane. No video pixels cross this interface.

Native GPU workers own their frame graphs. CPU DIS keeps its independently
tested pipeline. A missing capability is an error, never a silent fallback.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
MODES = ("sr", "fg", "sr-fg")
ALL_ENCODERS = ["h264_qsv", "hevc_qsv", "libx264", "libx265", "ffv1"]
WORKERS = {"gpu-block": "gpu-block-native.exe", "gpu-dis": "gpu-dis-native.exe",
           "amd-of": "amd-of-native.exe", "intel-vpl-ai": "vpl-ai-vpp.exe",
           "cpu-dis": "xess-vsr.exe"}
LABELS = {"gpu-block": "GPU Block", "cpu-dis": "CPU DIS", "intel-vpl-ai": "Intel 视频接口",
          "gpu-dis": "GPU DIS（实验）", "amd-of": "AMD 光流（实验）"}


class OfflineError(RuntimeError):
    pass


class Cancelled(OfflineError):
    pass


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capability(backend):
    cpu, intel = backend == "cpu-dis", backend == "intel-vpl-ai"
    gpu_effects = backend in ("gpu-block", "gpu-dis", "amd-of")
    return {"id": backend, "display_name": LABELS[backend],
            "status": "experimental" if backend in ("gpu-dis", "amd-of") else "stable",
            "modes": list(MODES), "encoders": list(ALL_ENCODERS),
            "effects": {"sharpen": [] if intel else list(MODES),
                        "five_frame": ["sr", "sr-fg"] if gpu_effects else ["sr"] if cpu else [],
                        "anti_stripe": ["sr", "sr-fg"] if cpu or gpu_effects else []},
            "depth": [] if intel else ["ai", "constant"] if cpu else ["ai"],
            "arc_a_compat": backend == "gpu-block",
            "frame_semantics": "2N" if intel else "2N-1"}


def runtime_paths(runtime):
    runtime = Path(runtime).resolve()
    return {"runtime": runtime, "bin": runtime / "bin", "ffmpeg": runtime / "media" / "ffmpeg.exe",
            "ffprobe": runtime / "probe" / "ffprobe.exe",
            "model": runtime / "models" / "depth-anything-v2-small" / "depth_anything_v2_small.xml",
            "shaders": runtime / "shaders" / "common", "dis_shaders": runtime / "shaders" / "dis"}


def missing_runtime(paths, backend, mode="sr", depth="ai"):
    files = [paths["ffmpeg"], paths["ffprobe"], paths["bin"] / WORKERS[backend]]
    if backend == "cpu-dis" and mode != "sr":
        files.append(paths["bin"] / "xess-fg.exe")
    if backend != "intel-vpl-ai":
        files.append(paths["bin"] / "libxess.dll")
        if mode != "sr":
            files += [paths["bin"] / "libxess_fg.dll", paths["bin"] / "libxell.dll"]
        if depth == "ai":
            files += [paths["model"], paths["model"].with_suffix(".bin")]
    if backend in ("gpu-block", "gpu-dis", "amd-of"):
        files += [paths["shaders"] / "native_depth_rgb.cso", paths["shaders"] / "native_provider_adapter.dxil",
                  paths["bin"] / "libvpl.dll", paths["bin"] / "openvino_intel_gpu_plugin.dll"]
    if backend == "gpu-dis":
        files.append(paths["dis_shaders"] / "dis_native_gray.dxil")
    if backend == "amd-of":
        files.append(paths["shaders"] / "native_amd_dense.dxil")
    return [str(p) for p in files if not p.is_file()]


def capabilities(runtime=None):
    routes = []
    for backend in LABELS:
        route = capability(backend)
        if runtime:
            route["missing_files"] = missing_runtime(runtime_paths(runtime), backend)
            route["available"] = not route["missing_files"]
            route["availability_scope"] = "files_only; hardware checked on launch"
        routes.append(route)
    return {"ok": True, "schema_version": "xess.offline.capabilities.v1", "backends": routes}


def validate_request(request):
    r = dict(request)
    if r.get("schema_version", "xess.offline.request.v1") != "xess.offline.request.v1":
        raise OfflineError("请求格式版本不受支持。")
    r.setdefault("backend", "gpu-block")
    r.setdefault("mode", "sr")
    r.setdefault("scale", 1.5)
    r.setdefault("depth", "ai")
    r.setdefault("encoder", "auto")
    if r["backend"] not in LABELS or r["mode"] not in MODES:
        raise OfflineError("未知算法或处理类型。")
    cap = capability(r["backend"])
    r.setdefault("arc_a_compat", False)
    if not isinstance(r["arc_a_compat"], bool):
        raise OfflineError("Arc A 系列兼容模式必须是布尔开关。")
    if r["arc_a_compat"] and not cap["arc_a_compat"]:
        raise OfflineError("Arc A 系列兼容模式目前仅支持 GPU Block，请关闭该项或切换后端。")
    if isinstance(r["scale"], bool) or not isinstance(r["scale"], (int, float)) or not math.isfinite(r["scale"]) or not 1 <= r["scale"] <= 4:
        raise OfflineError("超分倍率必须为 1 到 4 之间的有限数值。")
    if r["depth"] not in ("ai", "constant"):
        raise OfflineError("未知深度选项。")
    if r["backend"] in ("gpu-block", "gpu-dis", "amd-of") and r["depth"] != "ai":
        raise OfflineError("此 GPU 路线必须使用 AI 深度，请改为「AI 深度」；不会静默用常量替代。")
    for key in ("sharpen", "five_frame", "anti_stripe"):
        r.setdefault(key, False)
        if not isinstance(r[key], bool):
            raise OfflineError(key + " 必须是布尔开关。")
        if r[key] and r["mode"] not in cap["effects"][key]:
            raise OfflineError(f"{cap['display_name']} 的当前模式尚不支持 {key}；请关闭该项，不会静默忽略。")
    if r["encoder"] == "auto":
        r["encoder"] = "h264_qsv"
    if r["encoder"] not in cap["encoders"]:
        raise OfflineError(f"{cap['display_name']} 当前不支持 {r['encoder']}；请选择该路线支持的编码器。")
    for key in ("input", "output"):
        if not r.get(key) or not Path(r[key]).is_absolute():
            raise OfflineError(key + " 必须为完整文件路径。")
        r[key] = str(Path(r[key]).resolve())
    if r["input"] == r["output"] or (Path(r["output"]).exists() and Path(r["input"]).samefile(r["output"])):
        raise OfflineError("输出不能覆盖输入原片。")
    if not Path(r["input"]).is_file():
        raise OfflineError("输入视频不存在。")
    if Path(r["output"]).exists():
        raise OfflineError("输出文件已存在，请更换名称。")
    suffix = Path(r["output"]).suffix.lower()
    if suffix not in (".mp4", ".mkv") or (r["encoder"] == "ffv1" and suffix != ".mkv"):
        raise OfflineError("请使用 MP4/MKV 输出；FFV1 无损编码必须使用 MKV。")
    return r


def plan(request, runtime):
    r = validate_request(request)
    paths = runtime_paths(runtime)
    missing = missing_runtime(paths, r["backend"], r["mode"], r["depth"])
    if r["arc_a_compat"]:
        for name in ("native_rgba_share.cso", "native_rgba_ingress.dxil"):
            if not (paths["shaders"] / name).is_file():
                missing.append(str(paths["shaders"] / name))
    if r["backend"] in ("gpu-block", "gpu-dis", "amd-of") and (r["five_frame"] or r["anti_stripe"]):
        effect_shader = paths["shaders"] / "native_sr_effects.dxil"
        if not effect_shader.is_file():
            missing.append(str(effect_shader))
    if missing:
        raise OfflineError("运行时缺少文件，请修复工具箱运行时：\n" + "\n".join(missing))
    from gpu_media_finalize import source_contract
    data, video, pts, fps = source_contract(paths["ffprobe"], r["input"])
    width, height = int(video["width"]), int(video["height"])
    if width % 2 or height % 2:
        raise OfflineError("当前入口要求偶数宽高；不会静默裁剪原片。")
    if video.get("sample_aspect_ratio", "1:1") not in ("1:1", "0:1", None):
        raise OfflineError("当前版本尚不支持非方形像素视频。")
    if any(int(s.get("rotation", 0)) % 360 for s in video.get("side_data_list", [])):
        raise OfflineError("视频带旋转元数据，请先明确旋转为实际竖屏像素；不会静默猜测方向。")
    out_w, out_h = (width, height) if r["mode"] == "fg" else (
        round(width * r["scale"] / 16) * 16, round(height * r["scale"] / 16) * 16)
    if min(out_w, out_h) < 16 or max(out_w, out_h) > 8192:
        raise OfflineError("输出尺寸超出本版本已支持的范围。")
    native = r["backend"] in ("gpu-block", "gpu-dis", "amd-of")
    if native:
        if video.get("codec_name") not in ("h264", "hevc") or video.get("pix_fmt") not in ("yuv420p", "nv12"):
            raise OfflineError("当前 GPU 视频入口支持 8-bit H.264/HEVC 4:2:0；其他输入请使用 CPU DIS，不会自动转码降质。")
        if video.get("color_range") in ("pc", "full") or video.get("color_transfer") in ("smpte2084", "arib-std-b67"):
            raise OfflineError("当前 GPU 路线尚未定标 HDR/全范围输入，请勿按 SDR 处理。")
    if r["backend"] == "intel-vpl-ai":
        from fast_pro import probe_media, reject_unsupported_media, plan_sr_route
        info = probe_media(str(paths["ffprobe"]), r["input"], ffmpeg=str(paths["ffmpeg"]))
        reject_unsupported_media(info)
        if r["mode"] != "fg":
            if info.color_range in ("pc", "full") or info.color_space in ("bt470bg", "smpte170m"):
                raise OfflineError("Intel AI 超分当前只接受已定标的有限范围输入，不支持显式 BT.601。")
            p = plan_sr_route(width, height, out_w, out_h)
            if not p.supported or p.route == "preshrink":
                raise OfflineError("Intel 视频接口不支持该尺寸组合：" + p.reason)
    frames = len(pts)
    if frames < 2:
        raise OfflineError("请提供至少两帧视频。")
    expected = frames if r["mode"] == "sr" else (2 * frames if r["backend"] == "intel-vpl-ai" else 2 * frames - 1)
    rate = fps if r["mode"] == "sr" else fps * 2
    return {"requested": request, "applied": r, "backend": r["backend"], "mode": r["mode"],
            "source_width": width, "source_height": height, "source_frames": frames,
            "source_fps": str(fps), "output_width": out_w, "output_height": out_h,
            "output_frames": expected, "fps_rational": str(rate), "fps": float(rate),
            "scale_x": out_w / width, "scale_y": out_h / height,
            "frame_semantics": "N" if r["mode"] == "sr" else capability(r["backend"])["frame_semantics"],
            "codec": video["codec_name"], "color_space": video.get("color_space", "unknown"),
            "native_gpu": native,
            "encoder_boundary": ("terminal_rgb_readback_to_software" if native and r["encoder"] in ("libx264", "libx265", "ffv1")
                                 else "gpu_qsv" if native else "existing_video_runner"),
            "output": r["output"]}


def run_process(command, *, cwd, environment, log, cancelled, timeout=86400):
    if cancelled():
        raise Cancelled("任务已取消。")
    started = time.monotonic()
    with Path(log).open("wb") as stream:
        process = subprocess.Popen([str(s) for s in command], cwd=cwd, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if cancelled():
                    raise Cancelled("任务已取消，未发布成片。")
                if time.monotonic() - started > timeout:
                    raise OfflineError("任务超过时间限制，请查看本次日志。")
                time.sleep(.1)
            if process.returncode:
                with Path(log).open("rb") as tail:
                    tail.seek(max(0, Path(log).stat().st_size - 6000))
                    detail = tail.read().decode("utf-8", errors="replace")
                raise OfflineError(f"后端退出码 {process.returncode}：\n{detail}")
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/pid", str(process.pid), "/t", "/f"],
                                   capture_output=True, timeout=15)
                else:
                    process.kill()
                process.wait(timeout=10)


def native_encoded_path(p, job):
    encoder = p["applied"]["encoder"]
    return job / ("output.h264" if encoder == "h264_qsv" else "output.hevc" if encoder == "hevc_qsv" else "output.encoded.mkv")


def native_command(p, paths, job, cancel_file):
    r = p["applied"]
    backend = r["backend"]
    command = [paths["bin"] / WORKERS[backend], "--input", job / ("input." + p["codec"]),
               "--codec", p["codec"], "--max-frames", str(p["source_frames"]),
               "--fps", format(float(Fraction(p["source_fps"])), ".12g"),
               "--shader-dir", paths["shaders"], "--qsv-out", native_encoded_path(p, job),
               "--terminal-encoder", r["encoder"],
               "--report", job / "native.json", "--gpu-mode", r["mode"],
               "--output-width", str(p["output_width"]), "--output-height", str(p["output_height"]),
               "--motion-backend", backend, "--motion-scale", ".5" if backend == "gpu-block" else "1",
               "--motion-repair", "refine" if backend == "gpu-block" else "off", "--slots", "5" if r.get("five_frame", False) else "4",
               "--xess-quality", "ultra-quality" if r["scale"] <= 1.6 else "quality",
               "--gpu-post", "on" if r["sharpen"] else "off", "--cancel-file", cancel_file]
    if r.get("arc_a_compat", False):
        command += ["--decode-share", "rgba-d3d11"]
    if backend in ("gpu-block", "gpu-dis", "amd-of"):
        command += ["--gpu-five-frame", "on" if r.get("five_frame", False) else "off",
                    "--gpu-anti-stripe", "on" if r.get("anti_stripe", False) else "off"]
    if r["depth"] == "ai":
        command += ["--depth-model", paths["model"], "--gpu-depth-input", "ffmpeg-rgb"]
    if backend == "gpu-dis":
        command += ["--dis-shader-dir", paths["dis_shaders"], "--dis-provider-report", job / "provider.json"]
    return command


def cpu_command(p, paths, job):
    r, mode = p["applied"], p["mode"]
    script = {"sr": "run_xess.py", "fg": "run_fg.py", "sr-fg": "run_pipeline.py"}[mode]
    command = [sys.executable, ROOT / script, r["input"]]
    if mode == "sr":
        command += [str(r["scale"]), "--preset", "fast", "--flow-mode", "dis-occlusion",
                    "--sharpen", ".25" if r["sharpen"] else "0",
                    "--edge-guard-strength", ".90" if r["anti_stripe"] else "0"]
        if r["five_frame"]:
            command += ["--five-frame-fusion"]
        if r["depth"] == "ai":
            command += ["--force-depth"]
    elif mode == "fg":
        command += ["--preset", "fast", "--flow-mode", "dis-occlusion",
                    "--depth", r["depth"], "--motion-window", "2",
                    "--final-sharpen", ".25" if r["sharpen"] else "0"]
    else:
        command += ["--scale", str(r["scale"]), "--sr-preset", "fast", "--fg-preset", "fast",
                    "--sr-flow-mode", "dis-occlusion", "--fg-flow-mode", "dis-occlusion",
                    "--motion-sharing", "shared", "--depth", r["depth"], "--fg-motion-window", "2",
                    "--final-sharpen", ".25" if r["sharpen"] else "0",
                    "--anti-stripe", "on" if r["anti_stripe"] else "off"]
    command += ["--sharpen-mode", "fixed", "--depth-model", paths["model"], "--io-mode", "stream",
                "--out-dir", job / "cpu-output", "--work-dir", job / "cpu-work"]
    return command


def run_job(request, runtime, work_root, cancel_file=None, status_file=None):
    job = Path(work_root).resolve() / ("job-" + uuid.uuid4().hex)
    job.mkdir(parents=True)
    cancel_file = Path(cancel_file).resolve() if cancel_file else job / "cancel"
    status_file = Path(status_file).resolve() if status_file else job / "status.json"
    started = time.monotonic()
    def status(stage, **extra):
        result = {"schema_version": "xess.offline.status.v1", "status": stage,
                  "elapsed_s": time.monotonic() - started, "job": str(job), **extra}
        atomic_json(status_file, result)
        return result
    cancelled = lambda: cancel_file.exists()
    try:
        status("planning")
        p = plan(request, runtime)
        atomic_json(job / "plan.json", p)
        r, paths = p["applied"], runtime_paths(runtime)
        if cancelled():
            raise Cancelled("任务已取消。")
        # Limit predictable compressed scratch, not whole decoded video. No raw
        # full-clip cache is created by this control plane.
        needed = max(2 * 1024**3, Path(r["input"]).stat().st_size * 3)
        if shutil.disk_usage(job).free < needed + 1024**3:
            raise OfflineError("工作盘空间不足：请至少预留原片三倍加 1 GiB，或选择其他工作目录。")
        environment = os.environ.copy()
        environment.update({"TEMP": str(job), "TMP": str(job), "PYTHONIOENCODING": "utf-8",
                            "XESS_PYTHON": sys.executable, "XESS_FFMPEG": str(paths["ffmpeg"]),
                            "XESS_VSR": str(paths["bin"] / "xess-vsr.exe"),
                            "XESS_FG": str(paths["bin"] / "xess-fg.exe"),
                            "XESS_OUTPUT_ENCODER": r["encoder"],
                            "XESS_OUTPUT_CONTAINER": Path(r["output"]).suffix,
                            "XESS_FFPROBE": str(paths["ffprobe"]),
                            "OPENVINO_CACHE_DIR": str(Path(work_root).resolve() / "openvino-cache")})
        environment.update({"XESS_TERMINAL_FFMPEG": str(paths["ffmpeg"]),
                            "XESS_TERMINAL_OUTPUT": str(native_encoded_path(p, job)),
                            "XESS_TERMINAL_LOG": str(job / "terminal.log"),
                            "XESS_TERMINAL_CANCEL": str(cancel_file)})
        status("processing", backend=r["backend"], expected_frames=p["output_frames"])
        if r["backend"] == "intel-vpl-ai":
            from fast_pro import run_fast_pro
            # fast_pro owns child lifetime, logs, full decode/PTS validation.
            result = run_fast_pro(mode=r["mode"], source_path=r["input"], output_path=r["output"],
                ffprobe=str(paths["ffprobe"]), ffmpeg=str(paths["ffmpeg"]),
                vpl=str(paths["bin"] / WORKERS[r["backend"]]), work_dir=str(job),
                out_width=p["output_width"], out_height=p["output_height"], encoder=r["encoder"],
                cancel_file=str(cancel_file), timeout_s=86400)
            atomic_json(job / "intel-result.json", result)
        elif p["native_gpu"]:
            raw = job / ("input." + p["codec"])
            filter_name = "h264_mp4toannexb" if p["codec"] == "h264" else "hevc_mp4toannexb"
            run_process([paths["ffmpeg"], "-v", "error", "-i", r["input"], "-map", "0:v:0",
                         "-c:v", "copy", "-bsf:v", filter_name, "-an", "-f", p["codec"], raw],
                        cwd=job, environment=environment, log=job / "demux.log", cancelled=cancelled)
            command = native_command(p, paths, job, cancel_file)
            atomic_json(job / "command.json", [str(x) for x in command])
            try:
                run_process(command, cwd=paths["bin"], environment=environment,
                            log=job / "native.log", cancelled=cancelled)
            except Cancelled:
                raise
            except OfflineError as exc:
                terminal_log = job / "terminal.log"
                if terminal_log.exists():
                    with terminal_log.open("rb") as tail:
                        tail.seek(max(0, terminal_log.stat().st_size - 4000))
                        detail = tail.read().decode("utf-8", errors="replace")
                    raise OfflineError(f"成片编码失败（{r['encoder']}），未发布输出。\n{exc}\n编码器日志：\n{detail}") from exc
                raise
            if cancelled():
                raise Cancelled("任务已取消，未封装成片。")
            status("validating", backend=r["backend"])
            from gpu_media_finalize import finalize
            provider = job / ("native.json.amd.json" if r["backend"] == "amd-of" else "provider.json")
            finalize(source=r["input"], elementary=native_encoded_path(p, job), native_report=job / "native.json",
                     provider_report=provider if provider.exists() else None, backend=r["backend"],
                     output=r["output"], ffmpeg=paths["ffmpeg"], ffprobe=paths["ffprobe"],
                     report=job / "media-result.json", cancelled=cancelled)
            # Known files owned by this job only. Keep reports/logs and final.
            raw.unlink(missing_ok=True)
            native_encoded_path(p, job).unlink(missing_ok=True)
        else:
            command = cpu_command(p, paths, job)
            atomic_json(job / "command.json", [str(x) for x in command])
            run_process(command, cwd=ROOT, environment=environment,
                        log=job / "cpu.log", cancelled=cancelled)
            status("validating", backend=r["backend"])
            outputs = [f for f in (job / "cpu-output").glob("*") if f.suffix in (".mp4", ".mkv") and ".partial" not in f.name]
            if len(outputs) != 1:
                raise OfflineError("CPU 后端未产出唯一且已校验的成片。")
            # CPU workers validate exact count/fps/audio before renaming.
            # The final delivery name is published only after that validation.
            if cancelled():
                raise Cancelled("任务已取消。")
            Path(r["output"]).parent.mkdir(parents=True, exist_ok=True)
            if Path(r["output"]).exists():
                raise OfflineError("输出名称被其他任务占用，保留工作目录中的成片。")
            shutil.move(str(outputs[0]), r["output"])
        if not Path(r["output"]).is_file() or Path(r["output"]).stat().st_size == 0:
            raise OfflineError("没有找到已完成的视频，不能标记成功。")
        return {"ok": True, **status("complete", output=r["output"], backend=r["backend"], plan=p)}
    except Exception as error:
        stage = "cancelled" if isinstance(error, Cancelled) or cancelled() else "failed"
        result = {"ok": False, **status(stage, error=str(error))}
        print(str(error), file=sys.stderr)
        return result


def main():
    parser = argparse.ArgumentParser(description="XeSS 离线工具箱")
    parser.add_argument("command", choices=("capabilities", "plan", "run"))
    parser.add_argument("--runtime-root", default=os.environ.get("XESS_RUNTIME_ROOT"))
    parser.add_argument("--request")
    parser.add_argument("--work-root", default=str(ROOT.parent / "work" / "offline"))
    parser.add_argument("--cancel-file")
    parser.add_argument("--status-file")
    args = parser.parse_args()
    try:
        if args.command == "capabilities":
            result = capabilities(args.runtime_root)
        else:
            if not args.request or not args.runtime_root:
                raise OfflineError("需要请求文件及工具箱运行时目录。")
            request = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
            with redirect_stdout(sys.stderr):
                result = ({"ok": True, "plan": plan(request, args.runtime_root)} if args.command == "plan"
                          else run_job(request, args.runtime_root, args.work_root, args.cancel_file, args.status_file))
    except Exception as error:
        result = {"ok": False, "status": "failed", "error": str(error)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else (130 if result.get("status") == "cancelled" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
