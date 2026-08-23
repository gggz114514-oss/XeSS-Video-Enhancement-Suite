from __future__ import annotations

import collections
import importlib.util
import json
import os
import pathlib
import struct
import subprocess
import sys
import threading
import uuid
from typing import Callable

import cv2
import numpy as np
import torch

try:
    from .runtime_manager import (
        RuntimeManagerError, ensure_runtime, prepare_existing_engine,
    )
except ImportError:  # self_test.py imports this file as a top-level module.
    from runtime_manager import (
        RuntimeManagerError, ensure_runtime, prepare_existing_engine,
    )

try:
    import comfy.model_management as model_management
    import comfy.utils as comfy_utils
except ImportError:  # Allows source-level tests outside a running ComfyUI server.
    model_management = None
    comfy_utils = None


NODE_DIR = pathlib.Path(__file__).resolve().parent
ENGINE_ENV = "COMFYUI_XESS_ENGINE"
CONFIG_PATH = NODE_DIR / "xess_config.json"
_ENGINE_LOCK = threading.Lock()
_MODULE_CACHE: dict[tuple[pathlib.Path, str], object] = {}
_XPU_PROBE: dict[str, str] = {}

_XPU_WORK_KEYS = (
    "TEMP", "TMP", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME",
    "OPENVINO_CACHE_DIR", "PYTHONIOENCODING",
)

SR_PRESETS = {
    "fast": {"flow": "dis", "bidirectional": False, "mv_path": "highres",
             "sharpen_mode": "fixed", "sharpen": 0.25, "static": 0.30, "motion": 0.16},
    "balanced": {"flow": "sea-raft", "bidirectional": False, "mv_path": "lowres-depth",
                 "sharpen_mode": "adaptive", "sharpen": 0.30, "static": 0.34, "motion": 0.18},
    "quality": {"flow": "sea-raft", "bidirectional": True, "mv_path": "lowres-depth",
                "sharpen_mode": "adaptive", "sharpen": 0.35, "static": 0.38, "motion": 0.20},
}

FG_PRESETS = {
    "fast": {"flow": "dis", "bidirectional": False, "window": 2,
             "sharpen_mode": "fixed", "sharpen": 0.12, "static": 0.18, "motion": 0.08},
    "balanced": {"flow": "sea-raft", "bidirectional": False, "window": 5,
                 "sharpen_mode": "adaptive", "sharpen": 0.16, "static": 0.22, "motion": 0.10},
    "quality": {"flow": "sea-raft", "bidirectional": True, "window": 5,
                "sharpen_mode": "adaptive", "sharpen": 0.18, "static": 0.25, "motion": 0.10},
}

PRESET_CHOICES = ("快速（速度优先）", "均衡（质量优先）", "高质量（最慢）")
PRESET_VALUES = {
    "快速（速度优先）": "fast", "均衡（质量优先）": "balanced", "高质量（最慢）": "quality",
}
QUALITY_CHOICES = (
    "自动（按倍率选择）", "0 极致性能", "1 性能", "2 均衡",
    "3 质量", "4 超高质量", "5 超高质量+", "6 原尺寸抗锯齿",
)
QUALITY_VALUES = {"自动（按倍率选择）": "auto"}
FLOW_CHOICES = ("跟随处理档位", "DIS 极速", "DIS 遮挡增强", "SEA-RAFT 单向", "SEA-RAFT 双向")
FLOW_VALUES = {
    "跟随处理档位": "preset", "DIS 极速": "dis-fast", "DIS 遮挡增强": "dis-occlusion",
    "SEA-RAFT 单向": "sea-raft-single", "SEA-RAFT 双向": "sea-raft",
}
FLOW_RESOLUTION_CHOICES = ("自动 720p（推荐）", "原生分辨率（实验 / 极慢）")
FLOW_RESOLUTION_VALUES = {
    "自动 720p（推荐）": "auto720",
    "原生分辨率（实验 / 极慢）": "native",
    "auto720": "auto720", "native": "native",
}
MV_CHOICES = ("跟随处理档位", "高分辨率运动矢量", "低分辨率运动矢量+深度")
MV_VALUES = {
    "跟随处理档位": "preset", "高分辨率运动矢量": "highres",
    "低分辨率运动矢量+深度": "lowres-depth",
}
SHARPEN_CHOICES = ("跟随处理档位", "关闭", "固定锐化", "运动自适应锐化")
SHARPEN_VALUES = {
    "跟随处理档位": "preset", "关闭": "off", "固定锐化": "fixed",
    "运动自适应锐化": "adaptive",
}
TRANSPORT_CHOICES = ("自动", "内存管道", "共享内存")
TRANSPORT_VALUES = {"自动": "auto", "内存管道": "stream", "共享内存": "shared"}
DEPTH_CHOICES = ("AI 深度（推荐）", "固定深度（更快）")
DEPTH_VALUES = {"AI 深度（推荐）": "ai", "固定深度（更快）": "constant"}
WINDOW_CHOICES = ("跟随处理档位", "2 帧（更快）", "5 帧（更稳）")
WINDOW_VALUES = {"跟随处理档位": "preset", "2 帧（更快）": "2", "5 帧（更稳）": "5"}


INPUT_LABELS = {
    "images": "图像批次", "video": "视频", "source_fps": "源视频帧率",
    "preset": "处理档位", "scale": "放大倍率", "quality": "XeSS 画质档位",
    "flow_mode": "光流算法", "flow_resolution": "SEA-RAFT 分析分辨率",
    "mv_path": "运动矢量模式",
    "responsive_mask": "启用响应遮罩", "responsive_strength": "响应遮罩强度",
    "depth_temporal": "深度时序平滑", "flow_consistency": "光流一致性阈值",
    "mv_dilate": "运动边缘扩张", "depth_edge": "深度边缘阈值",
    "temporal_fusion": "五帧时序融合强度", "mfsr_enabled": "启用五帧多帧超分",
    "mfsr_strength": "MFSR 注入强度", "mfsr_detail_boost": "MFSR 细节增强",
    "mfsr_max_injection": "MFSR 最大注入", "sharpen_mode": "锐化模式",
    "sharpen_static": "静态区域锐化", "sharpen_motion": "运动区域锐化",
    "artifact_guard_strength": "竖向边缘振铃保护",
    "depth_mode": "深度模式", "motion_window": "运动分析窗口",
    "temporal_motion_strength": "五帧运动修正强度",
    "temporal_depth_strength": "五帧深度修正强度",
    "allow_overlay": "旧版窗口捕获允许覆盖层", "ui_mask": "字幕 / UI 遮罩",
    "transport": "数据传输方式", "device": "显卡编号",
    "free_vram": "运行前释放模型显存", "max_output_gb": "最大输出内存（GiB）",
    "engine_path": "XeSS 引擎目录", "work_dir": "工作缓存目录",
    "verbose": "输出详细日志",
}

INPUT_TOOLTIPS = {
    "preset": "快速适合预览；均衡适合日常成片；高质量使用双向 SEA-RAFT，耗时最高。",
    "scale": "输出宽高倍率。例如 480p 到 720p 通常填 1.5。",
    "quality": "自动会根据倍率选择 XeSS 档位。倍率 1.0 时可选 6 做原尺寸抗锯齿。",
    "flow_mode": "一般保持“跟随处理档位”。SEA-RAFT 遮挡边缘更稳，但明显更慢。",
    "flow_resolution": "自动 720p 实测速度更快且画质相同或更好；原生高分辨率仅供对照，速度会大幅降低。",
    "mv_path": "一般保持“跟随处理档位”。高质量预设会自动使用深度辅助路径。",
    "responsive_mask": "减少细线、字幕和快速变化区域的时序拖影。",
    "temporal_fusion": "0 为关闭；建议从 0.20～0.35 开始。会增加少量耗时。",
    "mfsr_enabled": "将相邻五帧可信高频注入 XeSS 成品，细节更多，但更慢。",
    "artifact_guard_strength": "抑制鼻梁、下颌和硬阴影旁的细竖线；0 关闭，0.75 为推荐值，过高会轻微软化竖向边缘。",
    "sharpen_mode": "建议跟随档位。运动自适应锐化可减少运动区域锯齿和噪声。",
    "depth_mode": "AI 深度更适合遮挡边缘；固定深度更快，但复杂运动更容易出错。",
    "allow_overlay": "仅供旧版 window 捕获后端兼容；当前 direct 交换链直读无需开启，也不会读取 RTSS OSD。",
    "ui_mask": "白色代表字幕/UI，黑色代表普通画面；可保护静态覆盖元素。",
    "transport": "自动：720p 及以下使用内存管道，更高分辨率使用共享内存。",
    "device": "-1 自动选择 Intel Arc；通常无需修改。",
    "free_vram": "运行 XeSS 前卸载 ComfyUI 模型，降低显存不足概率。",
    "max_output_gb": "预计输出张量超过此内存时提前停止，避免挤爆内存；0 表示不限制。",
    "engine_path": "保持 auto 自动查找；移动便携包后再手动填写。",
    "work_dir": "保持 auto 使用 E 盘缓存，并拒绝写入系统盘。",
}

ADVANCED_INPUTS = {
    "flow_mode", "flow_resolution", "mv_path", "responsive_strength", "depth_temporal", "flow_consistency",
    "mv_dilate", "depth_edge", "mfsr_strength", "mfsr_detail_boost", "mfsr_max_injection",
    "sharpen_static", "sharpen_motion", "motion_window", "temporal_motion_strength",
    "temporal_depth_strength", "allow_overlay", "transport", "device", "free_vram",
    "max_output_gb", "engine_path", "work_dir", "verbose", "artifact_guard_strength",
}


class XeSSNodeError(RuntimeError):
    pass


class _NullProgress:
    def update(self, _count: int) -> None:
        return


def _progress(total: int):
    return comfy_utils.ProgressBar(total) if comfy_utils is not None else _NullProgress()


def _interrupt() -> None:
    if model_management is not None:
        model_management.throw_exception_if_processing_interrupted()


def _free_vram(enabled: bool) -> None:
    if not enabled or model_management is None:
        return
    unload = getattr(model_management, "unload_all_models", None)
    if callable(unload):
        unload()
    model_management.soft_empty_cache()


def _node_config() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise XeSSNodeError(f"无法读取节点配置 {CONFIG_PATH}：{exc}") from exc
    if not isinstance(data, dict):
        raise XeSSNodeError(f"节点配置必须是 JSON 对象：{CONFIG_PATH}")
    return data


def _configured_path(value) -> pathlib.Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    expanded = pathlib.Path(os.path.expandvars(os.path.expanduser(value.strip())))
    if not expanded.is_absolute():
        expanded = NODE_DIR / expanded
    return expanded


def _candidate_engine_roots(requested: str):
    if requested and requested.strip().lower() != "auto":
        yield pathlib.Path(os.path.expandvars(os.path.expanduser(requested.strip())))
    configured = os.environ.get(ENGINE_ENV, "").strip()
    if configured:
        yield pathlib.Path(os.path.expandvars(os.path.expanduser(configured)))
    config_engine = _configured_path(_node_config().get("engine_path"))
    if config_engine is not None:
        yield config_engine
    yield NODE_DIR / ".runtime" / "engine"
    yield NODE_DIR / "xess-portable-pipeline"
    yield NODE_DIR.parent / "xess-portable-pipeline"
    for ancestor in (NODE_DIR, *NODE_DIR.parents):
        yield ancestor / "xess-tools" / "dist" / "release-sr12-fg12" / "xess-portable-pipeline"
        yield ancestor / "xess-tools" / "dist" / "xess-portable-pipeline"


def _engine_root(requested: str) -> pathlib.Path:
    checked: list[str] = []
    for candidate in _candidate_engine_roots(requested):
        candidate = candidate.resolve()
        if (candidate / "xess-portable-pipeline").is_dir():
            candidate = candidate / "xess-portable-pipeline"
        checked.append(os.fspath(candidate))
        if prepare_existing_engine(candidate):
            return candidate
    try:
        return ensure_runtime()
    except RuntimeManagerError as exc:
        raise XeSSNodeError(
            "找不到兼容的 XeSS 固定运行时，自动安装也未成功。"
            "可运行节点目录内的 install_runtime.bat 后重试。"
            f"已检查：{checked}\n安装错误：{exc}"
        ) from exc


def _work_root(engine: pathlib.Path, requested: str) -> pathlib.Path:
    if requested and requested.strip().lower() != "auto":
        root = pathlib.Path(os.path.expandvars(os.path.expanduser(requested.strip()))).resolve()
    else:
        config = _node_config()
        configured_work = _configured_path(config.get("work_dir"))
        if configured_work is not None:
            root = configured_work.resolve()
        else:
            xess_root = next((item for item in (engine, *engine.parents) if item.name == "xess-tools"), None)
            root = ((xess_root / "work" / "comfyui-cache") if xess_root is not None
                    else (engine.parent / "work" / "comfyui-cache"))
    root.mkdir(parents=True, exist_ok=True)
    system_drive = os.environ.get("SystemDrive", "C:").upper()
    allow_system_drive = bool(_node_config().get("allow_system_drive", False))
    if os.path.splitdrive(os.fspath(root))[0].upper() == system_drive and not allow_system_drive:
        raise XeSSNodeError(f"拒绝把 XeSS 工作缓存放在系统盘：{root}")
    return root


def _environment(engine: pathlib.Path, work: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.fspath(engine) + os.pathsep + env.get("PATH", "")
    env["TEMP"] = env["TMP"] = os.fspath(work)
    env["XDG_CACHE_HOME"] = os.fspath(work / "xdg")
    env["HF_HOME"] = os.fspath(work / "huggingface")
    env["TORCH_HOME"] = os.fspath(work / "torch")
    env["OPENVINO_CACHE_DIR"] = os.fspath(work / "openvino")
    env["PYTHONIOENCODING"] = "utf-8"
    for name in ("xdg", "huggingface", "torch", "openvino"):
        (work / name).mkdir(exist_ok=True)
    return env


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _load_engine_module(engine: pathlib.Path, module_name: str):
    key = (engine, module_name)
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    path = engine / f"{module_name}.py"
    unique = f"_comfyui_xess_{module_name}_{abs(hash(os.fspath(engine)))}"
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise XeSSNodeError(f"无法加载引擎模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    return module


def _xpu_probe(python: str, engine: pathlib.Path,
               env: dict[str, str]) -> tuple[bool, str]:
    # Match the real prepare process: OpenCV is imported by prepare_common,
    # then SEA-RAFT initializes torch XPU, and only afterwards does the depth
    # estimator import OpenVINO.  Importing OpenVINO before torch can select a
    # different SYCL/Level Zero runtime on some Arc A-series installations.
    script = (
        "import cv2\n"
        "import torch\n"
        "available = bool(torch.xpu.is_available())\n"
        "count = int(torch.xpu.device_count()) if available else 0\n"
        "print('XPU_PROBE', torch.__version__, available, count)\n"
        "if not available or count < 1: raise SystemExit(2)\n"
        "probe = torch.zeros(1, device='xpu')\n"
        "torch.xpu.synchronize()\n"
        "import openvino\n"
    )
    try:
        probe = subprocess.run(
            [python, "-c", script], env=env, cwd=engine,
            capture_output=True, text=True, errors="replace", timeout=45,
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"probe could not start: {exc}"
    stdout = (probe.stdout or "").strip()
    stderr = (probe.stderr or "").strip()
    detail = f"rc={probe.returncode} stdout={stdout!r}"
    if stderr:
        detail += f" stderr={stderr[-1200:]!r}"
    return probe.returncode == 0, detail


def _inherited_xpu_environment(env: dict[str, str]) -> dict[str, str]:
    """Restore the launcher's proven XPU environment and retain configured caches."""
    inherited = os.environ.copy()
    for key in _XPU_WORK_KEYS:
        if key in env:
            inherited[key] = env[key]
    return inherited


def _xpu_environment_candidates(env: dict[str, str]):
    """Yield conservative XPU environments in recovery order."""
    yield "node", env.copy()

    inherited = _inherited_xpu_environment(env)
    yield "inherited-main", inherited

    # ONEAPI_ROOT is installed as a system variable on Windows and does not
    # prove that setvars.bat completed.  Let the probe decide, then use the
    # current oneAPI selector only as a fallback.
    level_zero = inherited.copy()
    level_zero.pop("SYCL_DEVICE_FILTER", None)
    level_zero["ONEAPI_DEVICE_SELECTOR"] = "level_zero:*"
    yield "level-zero", level_zero

    # Older bundled SYCL runtimes may not understand ONEAPI_DEVICE_SELECTOR.
    legacy = inherited.copy()
    legacy.pop("ONEAPI_DEVICE_SELECTOR", None)
    legacy["SYCL_DEVICE_FILTER"] = "level_zero:gpu"
    yield "legacy-level-zero", legacy


def _xpu_python(engine: pathlib.Path, flow: str, env: dict[str, str]) -> str:
    portable = engine / "python" / "python.exe"
    if flow != "sea-raft":
        return os.fspath(portable)
    current = os.path.abspath(sys.executable)
    candidates = list(_xpu_environment_candidates(env))
    cached = _XPU_PROBE.get(current)
    if cached:
        selected = next((candidate for name, candidate in candidates if name == cached), None)
        if selected is not None:
            env.clear()
            env.update(selected)
            return current

    attempts: list[str] = []
    # Compare the complete environment.  Arc launchers can supply critical
    # Level Zero/SYCL switches without changing PATH or either device selector.
    # Collapsing candidates on only those three values would silently skip the
    # inherited environment that already works in ComfyUI's main process.
    fingerprints: set[tuple[tuple[str, str], ...]] = set()
    for name, candidate in candidates:
        fingerprint = tuple(sorted(candidate.items()))
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        ok, detail = _xpu_probe(current, engine, candidate)
        if ok:
            env.clear()
            env.update(candidate)
            _XPU_PROBE[current] = name
            print(f"[ComfyUI-XeSS] XPU subprocess environment: {name}", flush=True)
            return current
        attempts.append(f"{name}: {detail}")

    raise XeSSNodeError(
        "Balanced/Quality 在当前 ComfyUI Python 的子进程里无法初始化 torch.xpu。\n"
        "已按真实导入顺序测试原节点环境、启动器原始环境和 Level Zero 回退。\n"
        f"python={current}\n" + "\n".join(attempts)
    )


def _resolve_flow(preset: dict, override: str) -> tuple[str, bool]:
    if override == "preset":
        return preset["flow"], preset["bidirectional"]
    mapping = {
        "dis-fast": ("dis", False),
        "dis-occlusion": ("dis", True),
        "sea-raft-single": ("sea-raft", False),
        "sea-raft": ("sea-raft", True),
    }
    return mapping[override]


def _quality_for(scale: float) -> int:
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


def _parse_quality(value: str, scale: float) -> int:
    return _quality_for(scale) if value == "auto" else int(value.split()[0])


def _rgb8(images: torch.Tensor) -> np.ndarray:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
        raise XeSSNodeError("images 必须是 ComfyUI IMAGE 批次 [B,H,W,C]")
    if images.shape[0] < 1 or images.shape[1] < 2 or images.shape[2] < 2:
        raise XeSSNodeError("输入图像批次为空或尺寸无效")
    converted = (images[..., :3].detach().clamp(0.0, 1.0) * 255.0).round()
    return np.ascontiguousarray(converted.to(dtype=torch.uint8, device="cpu").numpy())


def _guard_output(frames: int, width: int, height: int, max_output_gb: float) -> None:
    float_bytes = frames * width * height * 3 * 4
    if max_output_gb > 0 and float_bytes > max_output_gb * 1024**3:
        raise XeSSNodeError(
            f"输出 IMAGE 张量预计 {float_bytes / 1024**3:.2f} GiB，超过限制 {max_output_gb:.2f} GiB；"
            "请减小批次/分辨率，或提高 max_output_gb"
        )


def _transport(value: str, height: int) -> str:
    if value == "auto":
        return "shared" if height > 720 else "stream"
    return value


def _ring(engine: pathlib.Path, width: int, height: int, *, depth: bool, mask: bool):
    module = _load_engine_module(engine, "shm_ring")
    header_size = struct.calcsize("<4sHHIIIIIIIIII")
    pixels = width * height
    slot_size = header_size + pixels * (3 + 8 + (4 if depth else 0) + (1 if mask else 0))
    return module.RingOwner(slots=4, slot_size=slot_size, prefix="comfy-xess")


def _drain_stderr(process: subprocess.Popen, label: str,
                  lines: collections.deque[str], verbose: bool) -> None:
    assert process.stderr is not None
    for raw in iter(process.stderr.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        lines.append(f"[{label}] {line}")
        lower = line.lower()
        if verbose or any(word in lower for word in ("error", "failed", "adapter:", "version", "complete")):
            print(f"[ComfyUI-XeSS/{label}] {line}", flush=True)


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _read_into(stream, array: np.ndarray) -> None:
    view = memoryview(array).cast("B")
    offset = 0
    while offset < len(view):
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError(f"输出流提前结束：收到 {offset}/{len(view)} 字节")
        offset += count


def _run_raw_chain(
    frames_u8: np.ndarray,
    output_frames: int,
    output_width: int,
    output_height: int,
    prepare_command: list[str],
    worker_command: list[str],
    engine: pathlib.Path,
    env: dict[str, str],
    *,
    transport: str,
    ring,
    fusion_command: list[str] | None = None,
    transform: Callable[[int, np.ndarray], np.ndarray] | None = None,
    verbose: bool = False,
) -> np.ndarray:
    processes: list[subprocess.Popen] = []
    stderr_lines: collections.deque[str] = collections.deque(maxlen=160)
    stderr_threads: list[threading.Thread] = []
    writer_errors: list[BaseException] = []
    flags = _creation_flags()

    def launch(label: str, command: list[str], *, stdin, stdout):
        print(f"[ComfyUI-XeSS] {label}: {subprocess.list2cmdline(command)}", flush=True)
        process = subprocess.Popen(
            command, cwd=engine, env=env, stdin=stdin, stdout=stdout,
            stderr=subprocess.PIPE, bufsize=0, creationflags=flags,
        )
        processes.append(process)
        thread = threading.Thread(target=_drain_stderr,
                                  args=(process, label, stderr_lines, verbose), daemon=True)
        thread.start()
        stderr_threads.append(thread)
        return process

    try:
        if fusion_command is not None:
            fusion = launch("fusion", fusion_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            input_pipe = fusion.stdin
            prepare_input = fusion.stdout
        else:
            fusion = None
            input_pipe = None
            prepare_input = subprocess.PIPE

        prepare_stdout = subprocess.DEVNULL if transport == "shared" else subprocess.PIPE
        prepare = launch("prepare", prepare_command, stdin=prepare_input, stdout=prepare_stdout)
        if fusion is not None:
            assert fusion.stdout is not None
            fusion.stdout.close()
        else:
            input_pipe = prepare.stdin

        worker_stdin = subprocess.DEVNULL if transport == "shared" else prepare.stdout
        worker = launch("worker", worker_command, stdin=worker_stdin, stdout=subprocess.PIPE)
        if prepare.stdout is not None:
            prepare.stdout.close()
        assert input_pipe is not None and worker.stdout is not None

        def write_input() -> None:
            try:
                for frame in frames_u8:
                    view = memoryview(frame).cast("B")
                    offset = 0
                    while offset < len(view):
                        written = input_pipe.write(view[offset:])
                        if not written:
                            raise BrokenPipeError(
                                f"输入管道提前关闭：写入 {offset}/{len(view)} 字节"
                            )
                        offset += written
                input_pipe.close()
            except BaseException as exc:
                writer_errors.append(exc)
                try:
                    input_pipe.close()
                except OSError:
                    pass

        writer_thread = threading.Thread(target=write_input, daemon=True)
        writer_thread.start()
        output = np.empty((output_frames, output_height, output_width, 3), np.uint8)
        pbar = _progress(output_frames)
        for index in range(output_frames):
            _read_into(worker.stdout, output[index])
            if transform is not None:
                output[index] = transform(index, output[index])
            pbar.update(1)
            _interrupt()
        extra = worker.stdout.read(1)
        worker.stdout.close()
        writer_thread.join()
        codes = [process.wait() for process in reversed(processes)]
        for thread in stderr_threads:
            thread.join(timeout=1)
        if writer_errors:
            raise XeSSNodeError(f"输入内存流失败：{writer_errors[0]}")
        if extra:
            raise XeSSNodeError("worker 输出多于声明帧数")
        if any(codes):
            detail = "\n".join(stderr_lines)
            raise XeSSNodeError(f"XeSS 内存管线失败，退出码 {list(reversed(codes))}\n{detail[-8000:]}")
        return output
    except BaseException as exc:
        _terminate(processes)
        for thread in stderr_threads:
            thread.join(timeout=1)
        if isinstance(exc, KeyboardInterrupt):
            raise
        detail = "\n".join(stderr_lines)
        raise XeSSNodeError(f"XeSS 内存管线异常：{exc}\n{detail[-8000:]}") from exc
    finally:
        if ring is not None:
            ring.close()


def _sharpen(frames: np.ndarray, mode: str, static: float, motion_strength: float) -> None:
    if mode == "off" or max(static, motion_strength) <= 0:
        return
    previous_luma = None
    fixed = mode == "fixed"
    kernel = np.ones((3, 3), np.uint8)
    for index in range(len(frames)):
        frame_u8 = frames[index]
        frame = frame_u8.astype(np.float32)
        luma = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if previous_luma is None or fixed:
            motion = np.zeros_like(luma)
        else:
            motion = np.clip(np.abs(luma - previous_luma) / 32.0, 0.0, 1.0)
            motion = cv2.GaussianBlur(motion, (5, 5), 0.9)
        contrast = cv2.dilate(luma, kernel) - cv2.erode(luma, kernel)
        noise_guard = np.clip((contrast - 72.0) / 80.0, 0.0, 0.65)
        strength = (static if fixed else static * (1.0 - motion) + motion_strength * motion)
        strength *= 1.0 - noise_guard
        blurred = cv2.GaussianBlur(frame, (0, 0), 0.8)
        detail = np.clip(frame - blurred, -24.0, 24.0)
        frames[index] = np.clip(frame + detail * (strength[..., None] * 1.65), 0, 255).astype(np.uint8)
        previous_luma = luma


def _suppress_vertical_ringing(source: np.ndarray, output: np.ndarray, strength: float) -> None:
    """Blend XeSS-only vertical edge halos back toward a bicubic source guide."""
    if strength <= 0:
        return
    out_height, out_width = output.shape[1:3]
    for index in range(min(len(source), len(output))):
        guide = cv2.resize(source[index], (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        guide_f = guide.astype(np.float32)
        guide_y = cv2.cvtColor(guide_f, cv2.COLOR_RGB2GRAY)
        vertical_edge = np.abs(cv2.Sobel(guide_y, cv2.CV_32F, 1, 0, ksize=3)) / 8.0
        edge_mask = np.clip((vertical_edge - 0.5) / 4.0, 0.0, 1.0)
        edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), 2.5)
        blend = np.clip(edge_mask * strength, 0.0, 0.90)[..., None]
        frame_f = output[index].astype(np.float32)
        output[index] = np.clip(frame_f * (1.0 - blend) + guide_f * blend,
                                0.0, 255.0).astype(np.uint8)


def _to_tensor(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames).to(dtype=torch.float32).div_(255.0)


def _resolve_sharpen(preset: dict, mode: str, static: float, motion: float):
    resolved_mode = preset["sharpen_mode"] if mode == "preset" else mode
    if static < 0:
        static = preset["sharpen"] if resolved_mode == "fixed" else preset["static"]
    if motion < 0:
        motion = static if resolved_mode == "fixed" else preset["motion"]
    return resolved_mode, static, motion


def _canonical(value: str, mapping: dict[str, str]) -> str:
    """Accept both Chinese UI values and old English workflow values."""
    return mapping.get(value, value)


def _localized_schema(schema: dict) -> dict:
    """Add Chinese labels/tooltips without changing stable backend input IDs."""
    localized = {}
    for group, inputs in schema.items():
        localized[group] = {}
        for name, spec in inputs.items():
            options = dict(spec[1]) if len(spec) > 1 else {}
            if name in INPUT_LABELS:
                options["display_name"] = INPUT_LABELS[name]
            if name in INPUT_TOOLTIPS:
                options["tooltip"] = INPUT_TOOLTIPS[name]
            if name in ADVANCED_INPUTS:
                options["advanced"] = True
            localized[group][name] = (spec[0], options)
    return localized


def _common_inputs():
    return {
        "transport": (TRANSPORT_CHOICES, {"default": "自动"}),
        "device": ("INT", {"default": -1, "min": -1, "max": 16, "step": 1}),
        "free_vram": ("BOOLEAN", {"default": True}),
        "max_output_gb": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 128.0, "step": 0.5}),
        "engine_path": ("STRING", {"default": "auto"}),
        "work_dir": ("STRING", {"default": "auto"}),
        "verbose": ("BOOLEAN", {"default": False}),
    }


class XeSSSuperResolution:
    DESCRIPTION = "XeSS SR 1.2：IMAGE 批次内存流超分/AA；不落整段视频或 raw。"
    CATEGORY = "XeSS 视频处理/专家"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("处理后图像", "宽度", "高度", "运行信息")

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "images": ("IMAGE",),
            "preset": (PRESET_CHOICES, {"default": "快速（速度优先）"}),
            "scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
            "quality": (QUALITY_CHOICES, {"default": "自动（按倍率选择）"}),
            "flow_mode": (FLOW_CHOICES, {"default": "跟随处理档位"}),
            "mv_path": (MV_CHOICES, {"default": "跟随处理档位"}),
            "responsive_mask": ("BOOLEAN", {"default": True}),
            "responsive_strength": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.02}),
            "depth_temporal": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.01}),
            "flow_consistency": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),
            "mv_dilate": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),
            "depth_edge": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.005}),
            "temporal_fusion": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "mfsr_enabled": ("BOOLEAN", {"default": False}),
            "mfsr_strength": ("FLOAT", {"default": 1.80, "min": 0.0, "max": 8.0, "step": 0.1}),
            "mfsr_detail_boost": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 4.0, "step": 0.05}),
            "mfsr_max_injection": ("FLOAT", {"default": 22.0, "min": 0.0, "max": 128.0, "step": 1.0}),
            "sharpen_mode": (SHARPEN_CHOICES, {"default": "跟随处理档位"}),
            "sharpen_static": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "sharpen_motion": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            **_common_inputs(),
            "artifact_guard_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
        }
        optional = {
            "flow_resolution": (FLOW_RESOLUTION_CHOICES,
                                {"default": "自动 720p（推荐）"}),
        }
        return _localized_schema({"required": required, "optional": optional})

    @classmethod
    def VALIDATE_INPUTS(cls, preset=None, quality=None, flow_mode=None, mv_path=None,
                        sharpen_mode=None, transport=None):
        # These combo inputs accept old English values from previously saved workflows.
        return True

    def upscale(self, images, preset, scale, quality, flow_mode, mv_path,
                responsive_mask, responsive_strength, depth_temporal,
                flow_consistency, mv_dilate, depth_edge, temporal_fusion,
                mfsr_enabled, mfsr_strength, mfsr_detail_boost,
                mfsr_max_injection, sharpen_mode, sharpen_static,
                sharpen_motion, transport, device, free_vram, max_output_gb,
                engine_path, work_dir, verbose, artifact_guard_strength=0.0,
                flow_resolution="auto720"):
        preset = _canonical(preset, PRESET_VALUES)
        quality = _canonical(quality, QUALITY_VALUES)
        flow_mode = _canonical(flow_mode, FLOW_VALUES)
        flow_resolution = _canonical(flow_resolution, FLOW_RESOLUTION_VALUES)
        mv_path = _canonical(mv_path, MV_VALUES)
        sharpen_mode = _canonical(sharpen_mode, SHARPEN_VALUES)
        transport = _canonical(transport, TRANSPORT_VALUES)
        source = _rgb8(images)
        original_frames, height, width, _ = source.shape
        if original_frames == 1:
            source = np.concatenate((source, source), axis=0)
        frame_count = len(source)
        out_width = max(2, int(round(width * scale)))
        out_height = max(2, int(round(height * scale)))
        out_width += out_width & 1
        out_height += out_height & 1
        _guard_output(original_frames, out_width, out_height, max_output_gb)
        engine = _engine_root(engine_path)
        work = _work_root(engine, work_dir)
        env = _environment(engine, work)
        defaults = SR_PRESETS[preset]
        flow, bidirectional = _resolve_flow(defaults, flow_mode)
        resolved_mv = defaults["mv_path"] if mv_path == "preset" else mv_path
        runtime_python = _xpu_python(engine, flow, env)
        resolved_quality = _parse_quality(quality, scale)
        resolved_transport = _transport(transport, height)
        depth_needed = resolved_mv == "lowres-depth" or flow != "dis"
        prepare = [
            runtime_python, os.fspath(engine / "prepare_sr.py"),
            "--in-w", str(width), "--in-h", str(height), "--frames", str(frame_count),
            "--engine", flow, "--temporal", str(depth_temporal),
            "--consistency", str(flow_consistency), "--dilate", str(mv_dilate),
            "--depth-edge", str(depth_edge), "--mv-path", resolved_mv,
            "--responsive-max", str(responsive_strength), "--stream",
        ]
        if bidirectional:
            prepare.append("--bidirectional")
        if flow == "sea-raft":
            prepare.extend(("--model-dir", os.fspath(engine / "models" / "sea-raft"),
                            "--device", "xpu", "--flow-resolution", flow_resolution))
            if flow_resolution == "native" and min(width, height) > 720:
                print("[ComfyUI-XeSS] warning: 原生高分辨率 SEA-RAFT 为实验模式，"
                      "速度会大幅降低，当前实测没有画质优势", flush=True)
        if depth_needed:
            prepare.extend(("--depth-model", os.fspath(engine / "models" / "depth-anything-v2-small" /
                                                       "depth_anything_v2_small.xml"),
                            "--depth-device", "GPU"))
        if responsive_mask:
            prepare.append("--responsive-mask")
        worker = [
            os.fspath(engine / "xess-vsr.exe"),
            "--in-w", str(width), "--in-h", str(height),
            "--out-w", str(out_width), "--out-h", str(out_height),
            "--frames-count", str(frame_count), "--quality", str(resolved_quality),
            "--mv-path", resolved_mv, "--responsive-max", str(responsive_strength), "--stream",
        ]
        if responsive_mask:
            worker.extend(("--mask", "stream"))
        if resolved_mv == "highres":
            worker.extend(("--mv-upsample", "bilinear" if flow == "dis" else "nearest"))
        if device >= 0:
            worker.extend(("--device", str(device)))
        if verbose:
            worker.append("--verbose")
        fusion = None
        if temporal_fusion > 0:
            fusion = [os.fspath(engine / "python" / "python.exe"),
                      os.fspath(engine / "five_frame_fusion.py"),
                      "--width", str(width), "--height", str(height),
                      "--frames", str(frame_count), "--strength", str(temporal_fusion)]
        ring = None
        if resolved_transport == "shared":
            ring = _ring(engine, width, height, depth=depth_needed, mask=responsive_mask)
            prepare.extend(ring.arguments())
            worker.extend(ring.arguments())

        transform = None
        if mfsr_enabled:
            module = _load_engine_module(engine, "five_frame_mfsr")
            reconstructor = module.MultiFrameReconstructor(
                width, height, out_width, out_height,
                mfsr_strength, mfsr_detail_boost, mfsr_max_injection,
            )

            def transform(index: int, xess_frame: np.ndarray) -> np.ndarray:
                neighbors = collections.OrderedDict(
                    (item, source[item])
                    for item in range(max(0, index - 2), min(frame_count, index + 3))
                )
                return reconstructor.reconstruct(index, neighbors, xess_frame)

        mode, static, motion = _resolve_sharpen(defaults, sharpen_mode,
                                                 sharpen_static, sharpen_motion)
        _free_vram(free_vram)
        with _ENGINE_LOCK:
            output = _run_raw_chain(
                source, frame_count, out_width, out_height, prepare, worker, engine, env,
                transport=resolved_transport, ring=ring, fusion_command=fusion,
                transform=transform, verbose=verbose,
            )
        _sharpen(output, mode, static, motion)
        _suppress_vertical_ringing(source, output, float(artifact_guard_strength))
        if original_frames == 1:
            output = output[:1]
        flow_info = flow_resolution if flow == "sea-raft" else "n/a"
        info = (f"SR1.2 {preset}/{flow} flow-res={flow_info} Q{resolved_quality} | {width}x{height} -> "
                f"{out_width}x{out_height} | {original_frames} frames | {resolved_transport}"
                f" | fusion={temporal_fusion:g} mfsr={bool(mfsr_enabled)}")
        return (_to_tensor(output), out_width, out_height, info)


class XeSSFrameGeneration:
    DESCRIPTION = "XeSS FG 1.2：IMAGE 视频批次 2× 插帧，输出 f0,G1,f1... 和 2×fps。"
    CATEGORY = "XeSS 视频处理/专家"
    FUNCTION = "interpolate"
    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("插帧后图像", "输出帧率", "输出帧数", "运行信息")

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "images": ("IMAGE",),
            "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
            "preset": (PRESET_CHOICES, {"default": "快速（速度优先）"}),
            "flow_mode": (FLOW_CHOICES, {"default": "跟随处理档位"}),
            "depth_mode": (DEPTH_CHOICES, {"default": "AI 深度（推荐）"}),
            "motion_window": (WINDOW_CHOICES, {"default": "跟随处理档位"}),
            "depth_temporal": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.01}),
            "flow_consistency": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),
            "mv_dilate": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),
            "depth_edge": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.005}),
            "temporal_motion_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
            "temporal_depth_strength": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 0.5, "step": 0.01}),
            "sharpen_mode": (SHARPEN_CHOICES, {"default": "跟随处理档位"}),
            "sharpen_static": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "sharpen_motion": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "allow_overlay": ("BOOLEAN", {"default": False}),
            **_common_inputs(),
        }
        optional = {
            "ui_mask": ("MASK",),
            "flow_resolution": (FLOW_RESOLUTION_CHOICES,
                                {"default": "自动 720p（推荐）"}),
        }
        return _localized_schema({"required": required, "optional": optional})

    @classmethod
    def VALIDATE_INPUTS(cls, preset=None, flow_mode=None, depth_mode=None,
                        motion_window=None, sharpen_mode=None, transport=None):
        # These combo inputs accept old English values from previously saved workflows.
        return True

    def interpolate(self, images, source_fps, preset, flow_mode, depth_mode,
                    motion_window, depth_temporal, flow_consistency, mv_dilate,
                    depth_edge, temporal_motion_strength, temporal_depth_strength,
                    sharpen_mode, sharpen_static, sharpen_motion, allow_overlay,
                    transport, device, free_vram, max_output_gb, engine_path,
                    work_dir, verbose, ui_mask=None, flow_resolution="auto720"):
        preset = _canonical(preset, PRESET_VALUES)
        flow_mode = _canonical(flow_mode, FLOW_VALUES)
        flow_resolution = _canonical(flow_resolution, FLOW_RESOLUTION_VALUES)
        depth_mode = _canonical(depth_mode, DEPTH_VALUES)
        motion_window = _canonical(motion_window, WINDOW_VALUES)
        sharpen_mode = _canonical(sharpen_mode, SHARPEN_VALUES)
        transport = _canonical(transport, TRANSPORT_VALUES)
        source = _rgb8(images)
        frame_count, height, width, _ = source.shape
        if frame_count < 2:
            raise XeSSNodeError("XeSS FG 至少需要连续两帧 IMAGE；单张图片不能插帧")
        output_count = frame_count * 2 - 1
        _guard_output(output_count, width, height, max_output_gb)
        engine = _engine_root(engine_path)
        work = _work_root(engine, work_dir)
        env = _environment(engine, work)
        defaults = FG_PRESETS[preset]
        flow, bidirectional = _resolve_flow(defaults, flow_mode)
        window = defaults["window"] if motion_window == "preset" else int(motion_window)
        runtime_python = _xpu_python(engine, flow, env)
        resolved_transport = _transport(transport, height)
        mask_path = None
        try:
            if ui_mask is not None:
                mask = ui_mask.detach().to(device="cpu").numpy()
                while mask.ndim > 2:
                    mask = mask[0]
                mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
                mask_path = work / f"ui-mask-{uuid.uuid4().hex}.png"
                if not cv2.imwrite(os.fspath(mask_path), mask_u8):
                    raise XeSSNodeError(f"无法写入临时 UI mask：{mask_path}")
            prepare = [
                runtime_python, os.fspath(engine / "prepare_fg.py"),
                "--in-w", str(width), "--in-h", str(height), "--frames", str(frame_count),
                "--engine", flow, "--temporal", str(depth_temporal),
                "--consistency", str(flow_consistency), "--dilate", str(mv_dilate),
                "--depth-edge", str(depth_edge), "--motion-window", str(window),
                "--temporal-motion-strength", str(temporal_motion_strength),
                "--temporal-depth-strength", str(temporal_depth_strength),
                "--mv-path", "lowres-depth", "--stream",
            ]
            if bidirectional:
                prepare.append("--bidirectional")
            if flow == "sea-raft":
                prepare.extend(("--model-dir", os.fspath(engine / "models" / "sea-raft"),
                                "--device", "xpu", "--flow-resolution", flow_resolution))
                if flow_resolution == "native" and min(width, height) > 720:
                    print("[ComfyUI-XeSS] warning: 原生高分辨率 SEA-RAFT 为实验模式，"
                          "速度会大幅降低，当前实测没有画质优势", flush=True)
            if depth_mode == "ai":
                prepare.extend(("--depth-model", os.fspath(engine / "models" / "depth-anything-v2-small" /
                                                           "depth_anything_v2_small.xml"),
                                "--depth-device", "GPU"))
            if mask_path is not None:
                prepare.extend(("--overlay-mask", os.fspath(mask_path)))
            worker = [
                os.fspath(engine / "xess-fg.exe"),
                "--width", str(width), "--height", str(height),
                "--frames-count", str(frame_count), "--fps", str(source_fps), "--stream",
                "--capture-mode", "direct",
            ]
            if mask_path is not None:
                worker.extend(("--ui-mask", "stream"))
            if device >= 0:
                worker.extend(("--device", str(device)))
            if verbose:
                worker.append("--verbose")
            if allow_overlay:
                worker.append("--allow-overlay")
            ring = None
            if resolved_transport == "shared":
                ring = _ring(engine, width, height, depth=True, mask=mask_path is not None)
                prepare.extend(ring.arguments())
                worker.extend(ring.arguments())
            mode, static, motion = _resolve_sharpen(defaults, sharpen_mode,
                                                     sharpen_static, sharpen_motion)
            _free_vram(free_vram)
            with _ENGINE_LOCK:
                output = _run_raw_chain(
                    source, output_count, width, height, prepare, worker, engine, env,
                    transport=resolved_transport, ring=ring, verbose=verbose,
                )
            _sharpen(output, mode, static, motion)
            output_fps = float(source_fps) * 2.0
            flow_info = flow_resolution if flow == "sea-raft" else "n/a"
            info = (f"FG1.2 {preset}/{flow} flow-res={flow_info} depth={depth_mode} window={window} | "
                    f"{width}x{height} | {frame_count}->{output_count} frames | "
                    f"{source_fps:g}->{output_fps:g} fps | {resolved_transport}")
            return (_to_tensor(output), output_fps, output_count, info)
        finally:
            if mask_path is not None:
                try:
                    mask_path.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"[ComfyUI-XeSS] UI mask 清理失败：{exc}", flush=True)


def _make_video(source_video, components, images: torch.Tensor, frame_rate):
    """Build a native ComfyUI VIDEO while preserving the source audio."""
    try:
        from comfy_api.latest import InputImpl, Types
    except ImportError as exc:
        raise XeSSNodeError("当前 ComfyUI 版本不支持原生 VIDEO 类型") from exc
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=images,
            audio=components.audio,
            frame_rate=frame_rate,
        ),
        bit_depth=source_video.get_bit_depth(),
    )


class XeSSVideoSuperResolutionExpert(XeSSSuperResolution):
    DESCRIPTION = "原生 VIDEO→VIDEO XeSS SR 1.2；自动保留帧率和音轨。"
    FUNCTION = "upscale_video"
    RETURN_TYPES = ("VIDEO", "INT", "INT", "STRING")
    RETURN_NAMES = ("处理后视频", "宽度", "高度", "运行信息")

    @classmethod
    def INPUT_TYPES(cls):
        parent = super().INPUT_TYPES()
        required = {"video": ("VIDEO", {"display_name": "输入视频", "tooltip": "连接 ComfyUI 原生 Load Video。"})}
        required.update((name, spec) for name, spec in parent["required"].items()
                        if name != "images")
        result = {"required": required}
        if "optional" in parent:
            result["optional"] = dict(parent["optional"])
        return result

    def upscale_video(self, video, **kwargs):
        components = video.get_components()
        images, width, height, info = super().upscale(components.images, **kwargs)
        output = _make_video(video, components, images, components.frame_rate)
        return output, width, height, f"{info} | audio=passthrough"


class XeSSVideoFrameGenerationExpert(XeSSFrameGeneration):
    DESCRIPTION = "原生 VIDEO→VIDEO XeSS FG 1.2；自动读取帧率、翻倍 fps 并保留音轨。"
    FUNCTION = "interpolate_video"
    RETURN_TYPES = ("VIDEO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("插帧后视频", "输出帧率", "输出帧数", "运行信息")

    @classmethod
    def INPUT_TYPES(cls):
        parent = super().INPUT_TYPES()
        required = {"video": ("VIDEO", {"display_name": "输入视频", "tooltip": "自动读取帧率并保留音频。"})}
        required.update((name, spec) for name, spec in parent["required"].items()
                        if name not in {"images", "source_fps"})
        result = {"required": required}
        if "optional" in parent:
            result["optional"] = dict(parent["optional"])
        return result

    def interpolate_video(self, video, ui_mask=None, **kwargs):
        components = video.get_components()
        source_fps = float(components.frame_rate)
        images, output_fps, output_count, info = super().interpolate(
            components.images,
            source_fps=source_fps,
            ui_mask=ui_mask,
            **kwargs,
        )
        output = _make_video(video, components, images, components.frame_rate * 2)
        return output, output_fps, output_count, f"{info} | audio=passthrough"


SIMPLE_MODE_CHOICES = ("极速模式（最低挡）", "极致画质（最高挡）")
SIMPLE_MODE_VALUES = {
    "极速模式（最低挡）": "fast",
    "极致画质（最高挡）": "quality",
    # Accept values from older workflows and direct API calls.
    "快速（速度优先）": "fast", "均衡（质量优先）": "quality", "高质量（最慢）": "quality",
    "fast": "fast", "balanced": "quality", "quality": "quality",
}


class XeSSVideoSuperResolution:
    DESCRIPTION = "两挡自动 VIDEO→VIDEO 超分：只选极速或极致画质，其余参数自动配置。"
    CATEGORY = "XeSS 视频处理"
    FUNCTION = "upscale_video"
    RETURN_TYPES = ("VIDEO", "INT", "INT", "STRING")
    RETURN_NAMES = ("处理后视频", "宽度", "高度", "自动配置说明")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO", {"display_name": "输入视频", "tooltip": "连接 ComfyUI 原生 Load Video。"}),
            "mode": (SIMPLE_MODE_CHOICES, {
                "default": "极速模式（最低挡）", "display_name": "质量挡位",
                "tooltip": "极速：DIS；极致：双向 SEA-RAFT 自动限制到 720p 分析，再恢复到原尺寸。",
            }),
            "scale": ("FLOAT", {
                "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                "display_name": "放大倍率", "tooltip": "480p 到 720p 通常填 1.5。",
            }),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, mode=None):
        return True

    def upscale_video(self, video, mode, scale):
        resolved = SIMPLE_MODE_VALUES.get(mode, mode)
        if resolved not in {"fast", "quality"}:
            raise XeSSNodeError(f"未知质量挡位：{mode}")
        high = resolved == "quality"
        output = XeSSVideoSuperResolutionExpert().upscale_video(
            video,
            preset=resolved,
            scale=scale,
            quality="auto",
            flow_mode="preset",
            mv_path="preset",
            responsive_mask=True,
            responsive_strength=0.8,
            depth_temporal=0.25,
            flow_consistency=1.5,
            mv_dilate=1,
            depth_edge=0.04,
            temporal_fusion=0.35 if high else 0.0,
            mfsr_enabled=False,
            mfsr_strength=1.8,
            mfsr_detail_boost=0.9,
            mfsr_max_injection=22.0,
            sharpen_mode="preset",
            sharpen_static=-1.0,
            sharpen_motion=-1.0,
            transport="auto",
            device=-1,
            free_vram=True,
            max_output_gb=12.0,
            engine_path="auto",
            work_dir="auto",
            verbose=False,
            artifact_guard_strength=0.75,
            flow_resolution="auto720",
        )
        video_out, width, height, info = output
        label = "极致画质：双向 SEA-RAFT（自动 720p）+ AI 深度 + 五帧融合 + 自适应锐化 + 边缘振铃保护" if high else (
            "极速模式：DIS 光流 + 响应遮罩 + 固定锐化 + 边缘振铃保护"
        )
        return video_out, width, height, f"{label} | {info}"


class XeSSVideoFrameGeneration:
    DESCRIPTION = "两挡自动 VIDEO→VIDEO 插帧：自动读取 fps，只选极速或极致画质。"
    CATEGORY = "XeSS 视频处理"
    FUNCTION = "interpolate_video"
    RETURN_TYPES = ("VIDEO", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("插帧后视频", "输出帧率", "输出帧数", "自动配置说明")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"display_name": "输入视频", "tooltip": "自动读取帧率并保留音频。"}),
                "mode": (SIMPLE_MODE_CHOICES, {
                    "default": "极速模式（最低挡）", "display_name": "质量挡位",
                    "tooltip": "极速：DIS；极致：双向 SEA-RAFT 自动限制到 720p 分析，再恢复到原尺寸。",
                }),
            },
            "optional": {
                "ui_mask": ("MASK", {"display_name": "字幕 / UI 遮罩", "tooltip": INPUT_TOOLTIPS["ui_mask"]}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, mode=None):
        return True

    def interpolate_video(self, video, mode, ui_mask=None):
        resolved = SIMPLE_MODE_VALUES.get(mode, mode)
        if resolved not in {"fast", "quality"}:
            raise XeSSNodeError(f"未知质量挡位：{mode}")
        high = resolved == "quality"
        output = XeSSVideoFrameGenerationExpert().interpolate_video(
            video,
            preset=resolved,
            flow_mode="preset",
            depth_mode="ai",
            motion_window="preset",
            depth_temporal=0.25,
            flow_consistency=1.5,
            mv_dilate=1,
            depth_edge=0.04,
            temporal_motion_strength=0.65,
            temporal_depth_strength=0.18,
            sharpen_mode="preset",
            sharpen_static=-1.0,
            sharpen_motion=-1.0,
            allow_overlay=False,
            transport="auto",
            device=-1,
            free_vram=True,
            max_output_gb=12.0,
            engine_path="auto",
            work_dir="auto",
            verbose=False,
            flow_resolution="auto720",
            ui_mask=ui_mask,
        )
        video_out, output_fps, output_count, info = output
        label = "极致画质：双向 SEA-RAFT（自动 720p）+ AI 深度 + 5帧窗口 + 自适应锐化" if high else (
            "极速模式：DIS 光流 + AI 深度 + 2帧窗口 + 固定锐化"
        )
        return video_out, output_fps, output_count, f"{label} | {info}"
