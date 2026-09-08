"""独立的 R4 离线视频节点。

本模块故意不复用旧 ``xess_nodes.py`` 的大节点。它只负责 ComfyUI 的
输入/输出合同、能力展示和 ``pipeline/offline_toolbox.py`` 子进程控制；
真正的超分、插帧、编码和音频处理始终由共享离线 worker 完成。

为了兼容不同 ComfyUI 版本，输入端使用原生 ``VIDEO``，但只接受能解析出
保留容器文件的 VIDEO。纯内存 IMAGE 批次不会被偷偷转存成 GPU 路线。
输出端优先构造 ``InputImpl.VideoFromFile``，另返回绝对路径供 SaveVideo
或调试节点查看。
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from typing import Any


NODE_DIR = pathlib.Path(__file__).resolve().parent
SCHEMA_VERSION = "xess.offline.request.v1"
CAPABILITIES_SCHEMA = "xess.offline.capabilities.v1"

BACKEND_LABELS = {
    "auto": "自动（按能力）",
    "gpu-block": "GPU Block（快速）",
    "cpu-dis": "CPU DIS（稳定）",
    "intel-vpl-ai": "Intel 视频接口",
    "gpu-dis": "GPU DIS（实验）",
    "amd-of": "AMD 光流（实验）",
}
BACKEND_IDS = {label: backend for backend, label in BACKEND_LABELS.items()}
ENCODER_LABELS = {
    "auto": "自动",
    "h264_qsv": "H.264 QSV",
    "hevc_qsv": "HEVC QSV",
    "libx264": "libx264",
    "libx265": "libx265",
    "ffv1": "FFV1（无损）",
}
ENCODER_IDS = {label: encoder for encoder, label in ENCODER_LABELS.items()}
DEPTH_LABELS = {"ai": "AI 深度", "constant": "固定深度"}
DEPTH_IDS = {label: depth for depth, label in DEPTH_LABELS.items()}
SCALE_LABELS = {"1.33×": 1.33, "1.5×": 1.5, "2×": 2.0, "自定义": None}

DEFAULT_ENCODERS = tuple(ENCODER_LABELS)
DEFAULT_BACKENDS = tuple(BACKEND_LABELS)


class OfflineNodeError(RuntimeError):
    """用户可理解的节点错误，不把子进程 EOF 当作根因。"""


class _ProcessResult:
    def __init__(self, returncode: int, payload: dict[str, Any] | None,
                 stderr: str, cancelled: bool = False) -> None:
        self.returncode = returncode
        self.payload = payload
        self.stderr = stderr
        self.cancelled = cancelled


_CAP_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CAP_CACHE_LOCK = threading.Lock()


def _runtime_root() -> pathlib.Path | None:
    """Resolve only portable/configurable runtime locations.

    No machine-specific E: path is embedded.  ``XESS_RUNTIME_ROOT`` is the
    preferred override; package-relative ``.runtime/runtime`` is for bundles.
    """

    configured = os.environ.get("XESS_RUNTIME_ROOT", "").strip()
    if configured:
        return pathlib.Path(os.path.expandvars(os.path.expanduser(configured))).resolve()
    try:
        from . import runtime_manager
    except ImportError:
        import runtime_manager
    try:
        managed = runtime_manager.default_engine()
        if managed.is_dir():
            return managed
    except runtime_manager.RuntimeManagerError:
        pass
    bundled = NODE_DIR / ".runtime" / "runtime"
    if bundled.is_dir():
        return bundled.resolve()
    package_runtime = NODE_DIR / "runtime"
    if package_runtime.is_dir():
        return package_runtime.resolve()
    return None


def _toolbox_script() -> pathlib.Path:
    runtime = _runtime_root()
    candidates = [NODE_DIR / "pipeline" / "offline_toolbox.py"]
    if runtime is not None:
        candidates.extend((runtime.parent / "pipeline" / "offline_toolbox.py",
                           runtime / "pipeline" / "offline_toolbox.py",
                           runtime / "offline_toolbox.py"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "、".join(os.fspath(item) for item in candidates)
    raise OfflineNodeError(
        "缺少离线控制入口 pipeline/offline_toolbox.py；已检查：" + checked)


def _toolbox_python(runtime: pathlib.Path | None = None) -> str:
    if runtime is not None:
        candidates = (runtime / "python" / "python.exe",
                      runtime / "python" / "python",
                      runtime / "Scripts" / "python.exe")
        for candidate in candidates:
            if candidate.is_file():
                return os.fspath(candidate)
    return sys.executable


def _work_root() -> pathlib.Path:
    configured = os.environ.get("XESS_OFFLINE_WORK_ROOT", "").strip()
    if configured:
        root = pathlib.Path(os.path.expandvars(os.path.expanduser(configured)))
    else:
        try:
            import folder_paths
            output_directory = getattr(folder_paths, "get_output_directory", None)
            output_directory = output_directory() if callable(output_directory) else None
        except (ImportError, OSError, TypeError):
            output_directory = None
        # Comfy's configured output folder is user-writable. Source-level
        # tests without Comfy use a package-local work folder; never silently
        # choose the system C: temp directory.
        root = (pathlib.Path(output_directory) / ".xess-work"
                if output_directory else NODE_DIR / "work")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_json_lines(stdout: str) -> dict[str, Any] | None:
    # The toolbox may print human-readable progress.  The protocol guarantees
    # that its final non-empty stdout line is JSON.
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


class _BoundedTail:
    """Bounded text tail for noisy worker diagnostics."""

    def __init__(self, limit: int = 64 * 1024) -> None:
        self.limit = limit
        self._chunks: deque[str] = deque()
        self._size = 0

    def append(self, value: str) -> None:
        if len(value) >= self.limit:
            self._chunks.clear()
            self._chunks.append(value[-self.limit:])
            self._size = self.limit
            return
        self._chunks.append(value)
        self._size += len(value)
        while self._size > self.limit and self._chunks:
            removed = self._chunks.popleft()
            excess = self._size - self.limit
            if len(removed) <= excess:
                self._size -= len(removed)
            else:
                self._chunks.appendleft(removed[excess:])
                self._size -= excess
                break

    def text(self) -> str:
        return "".join(self._chunks)


def _drain(stream, sink: _BoundedTail) -> None:
    try:
        for line in iter(stream.readline, ""):
            if line:
                sink.append(line)
    finally:
        stream.close()


def _interrupt_requested() -> bool:
    try:
        import comfy.model_management as model_management
    except ImportError:
        return False
    # Comfy's public helper raises rather than returning a flag.
    checker = getattr(model_management, "throw_exception_if_processing_interrupted", None)
    if not callable(checker):
        return False
    try:
        checker()
    except BaseException:
        return True
    return False


def _terminate_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # TerminateProcess alone does not guarantee that ffmpeg/native workers
        # spawned by the toolbox are gone.  Kill the process tree first.
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False, timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            process.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
        return
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=1.5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def _run_command(args: list[str], *, timeout: float = 30.0,
                 cancel_file: pathlib.Path | None = None,
                 allow_interrupt: bool = True) -> _ProcessResult:
    """Run one toolbox command with bounded waits and continuously drained stderr."""

    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        start_new_session = True
    try:
        environment = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONPATH"):
            environment.pop(key, None)
        environment.update(PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            start_new_session=start_new_session,
            env=environment,
        )
    except OSError as exc:
        raise OfflineNodeError(f"无法启动离线控制入口：{exc}") from exc
    stdout = _BoundedTail()
    stderr = _BoundedTail()
    stdout_thread = threading.Thread(target=_drain, args=(process.stdout, stdout), daemon=True)
    stderr_thread = threading.Thread(target=_drain, args=(process.stderr, stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    started = time.monotonic()
    cancelled = False
    while process.poll() is None:
        if cancel_file is not None and cancel_file.is_file():
            cancelled = True
            _terminate_tree(process)
            break
        if allow_interrupt and _interrupt_requested():
            cancelled = True
            if cancel_file is not None:
                try:
                    cancel_file.write_text("cancelled by ComfyUI\n", encoding="utf-8")
                except OSError:
                    pass
            _terminate_tree(process)
            break
        if time.monotonic() - started > timeout:
            _terminate_tree(process)
            raise OfflineNodeError("离线控制入口超时；已结束子进程树，请检查运行时日志")
        time.sleep(0.10)
    try:
        returncode = process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        _terminate_tree(process)
        returncode = process.poll()
        if returncode is None:
            returncode = -1
    stdout_thread.join(timeout=2.0)
    stderr_thread.join(timeout=2.0)
    payload = _read_json_lines(stdout.text())
    return _ProcessResult(int(returncode), payload, stderr.text(), cancelled)


def _capability_items(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    raw = payload.get("backends", [])
    if isinstance(raw, Mapping):
        raw = [dict(value, id=key) if isinstance(value, Mapping) else {"id": key, "status": value}
               for key, value in raw.items()]
    result: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                result.append(dict(item))
    return result


def _is_available(item: Mapping[str, Any]) -> bool:
    if item.get("available") is False:
        return False
    status = item.get("status", False)
    if isinstance(status, bool):
        return status
    return str(status).strip().lower() in {
        "supported", "available", "ready", "experimental", "stable", "ok", "true", "enabled",
    }


def _backend_item(backend: str, payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    for item in _capability_items(payload):
        if item.get("id") == backend:
            return item
    return None


def _supports_mode(item: Mapping[str, Any] | None, mode: str) -> bool:
    if item is None or not _is_available(item):
        return False
    modes = item.get("modes", [])
    return not modes or mode in modes


def _supports_effect(item: Mapping[str, Any] | None, effect: str, mode: str) -> bool:
    if item is None:
        return False
    effects = item.get("effects", {})
    value = effects.get(effect, []) if isinstance(effects, Mapping) else []
    if value is True:
        return True
    if isinstance(value, str):
        return value in {mode, "all", "true", "supported"}
    if isinstance(value, (list, tuple, set)):
        return mode in value or "all" in value
    return False


def _capabilities(force: bool = False) -> dict[str, Any]:
    runtime = _runtime_root()
    cache_key = os.fspath(runtime) if runtime is not None else "<code-only>"
    now = time.monotonic()
    with _CAP_CACHE_LOCK:
        cached = _CAP_CACHE.get(cache_key)
        if cached and not force and now - cached[0] < 5.0:
            return cached[1]
    script = _toolbox_script()
    args = [_toolbox_python(runtime), os.fspath(script), "capabilities"]
    if runtime is not None:
        args.extend(("--runtime-root", os.fspath(runtime)))
    result = _run_command(args, timeout=15.0, allow_interrupt=False)
    if result.payload is None:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "无 JSON 能力结果"
        raise OfflineNodeError(f"能力探测失败：{detail}")
    if result.returncode != 0 or result.payload.get("ok") is False:
        detail = result.payload.get("error") or result.stderr.strip() or "能力探测返回失败"
        raise OfflineNodeError(f"能力探测失败：{detail}")
    with _CAP_CACHE_LOCK:
        _CAP_CACHE[cache_key] = (now, result.payload)
    return result.payload


def clear_capability_cache() -> None:
    with _CAP_CACHE_LOCK:
        _CAP_CACHE.clear()


def _backend_choices(mode: str) -> tuple[str, ...]:
    try:
        payload = _capabilities()
    except OfflineNodeError:
        # Keep the requested GPU Block default visible even before the runtime
        # is installed; execution remains fail-closed and never falls to CPU.
        return (BACKEND_LABELS["gpu-block"], BACKEND_LABELS["auto"])
    choices = [BACKEND_LABELS["auto"]]
    items = _capability_items(payload)
    for item in items:
        backend = str(item.get("id"))
        if backend in BACKEND_LABELS and _supports_mode(item, mode):
            choices.append(BACKEND_LABELS[backend])
    # Comfy combo widgets cannot represent a disabled default. Keep the
    # explicit GPU Block choice visible when the report knows the route but
    # marks its runtime files unavailable; plan then fails with the exact
    # missing-file diagnostic instead of silently switching to auto/CPU.
    if (any(item.get("id") == "gpu-block" for item in items)
            and BACKEND_LABELS["gpu-block"] not in choices):
        choices.insert(0, BACKEND_LABELS["gpu-block"])
    return tuple(dict.fromkeys(choices))


def _encoder_choices(mode: str) -> tuple[str, ...]:
    try:
        payload = _capabilities()
    except OfflineNodeError:
        return (ENCODER_LABELS["auto"],)
    available: set[str] = set()
    for item in _capability_items(payload):
        if not _supports_mode(item, mode):
            continue
        values = item.get("encoders", [])
        if isinstance(values, list):
            available.update(str(value) for value in values)
    available.add("auto")
    return tuple(ENCODER_LABELS[value] for value in DEFAULT_ENCODERS if value in available)


def _canonical_backend(value: str) -> str:
    return BACKEND_IDS.get(value, value)


def _canonical_encoder(value: str) -> str:
    return ENCODER_IDS.get(value, value)


def _canonical_depth(value: str) -> str:
    return DEPTH_IDS.get(value, value)


def _scale_value(value: Any, custom_scale: float | None = None) -> float:
    if isinstance(value, str):
        text = value.strip()
        if text in SCALE_LABELS:
            selected = SCALE_LABELS[text]
            if selected is None:
                value = custom_scale
            else:
                value = selected
        else:
            text = text.replace("×", "").replace("x", "").strip()
            try:
                value = float(text)
            except ValueError as exc:
                raise OfflineNodeError(f"无法识别 SR 倍率：{value}") from exc
    if value is None:
        value = custom_scale
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OfflineNodeError("SR 倍率必须是 1.0 到 4.0 之间的数字") from exc
    if not 1.0 <= result <= 4.0:
        raise OfflineNodeError(f"SR 倍率 {result:g} 超出 1.0–4.0 范围")
    return round(result, 4)


def _source_path(video: Any) -> pathlib.Path:
    if isinstance(video, (str, os.PathLike)):
        candidate = pathlib.Path(os.fspath(video)).expanduser()
    else:
        candidate = None
        for method_name in ("get_stream_source", "get_source_path", "get_path"):
            method = getattr(video, method_name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except Exception:
                continue
            if isinstance(value, (str, os.PathLike)):
                candidate = pathlib.Path(os.fspath(value)).expanduser()
                break
    if candidate is None:
        raise OfflineNodeError(
            "输入 VIDEO 只有内存组件，R4 离线节点需要保留的视频文件路径；"
            "请直接连接 Load Video 的文件型 VIDEO，不要连接仅有 IMAGE 批次的节点。")
    candidate = candidate.resolve()
    if not candidate.is_absolute() or not candidate.is_file():
        raise OfflineNodeError(f"输入视频文件不存在或不是文件：{candidate}")
    return candidate


def _new_job(mode: str, encoder: str = "auto") -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    root = _work_root()
    job = root / f"comfy-r4-{mode}-{uuid.uuid4().hex}"
    job.mkdir(parents=True, exist_ok=False)
    request_path = job / "request.json"
    cancel_path = job / "cancel.signal"
    status_path = job / "status.json"
    output_path = job / ("output.mkv" if encoder == "ffv1" else "output.mp4")
    return job, request_path, cancel_path, status_path, output_path


def _request(*, mode: str, source: pathlib.Path, output: pathlib.Path,
             backend: str, scale: float, encoder: str, depth: str,
             sharpen: bool, five_frame: bool, anti_stripe: bool,
             arc_a_compat: bool = False) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "backend": backend,
        "input": os.fspath(source),
        "output": os.fspath(output),
        "scale": scale,
        "encoder": encoder,
        "depth": depth,
        "sharpen": bool(sharpen),
        "five_frame": bool(five_frame),
        "anti_stripe": bool(anti_stripe),
        "arc_a_compat": arc_a_compat,
    }


def _write_request(path: pathlib.Path, request: Mapping[str, Any]) -> None:
    try:
        path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    except OSError as exc:
        raise OfflineNodeError(f"无法写入离线请求文件：{exc}") from exc


def _invoke_toolbox(command: str, request_path: pathlib.Path, *,
                    cancel_path: pathlib.Path | None = None,
                    status_path: pathlib.Path | None = None,
                    timeout: float = 60.0) -> _ProcessResult:
    runtime = _runtime_root()
    if runtime is None:
        raise OfflineNodeError(
            "未找到运行时根目录；请设置 XESS_RUNTIME_ROOT，或把运行时放到节点包的 .runtime/runtime")
    script = _toolbox_script()
    args = [_toolbox_python(runtime), os.fspath(script), command,
            "--request", os.fspath(request_path),
            "--runtime-root", os.fspath(runtime),
            "--work-root", os.fspath(_work_root())]
    if command == "run":
        if cancel_path is not None:
            args.extend(("--cancel-file", os.fspath(cancel_path)))
        if status_path is not None:
            args.extend(("--status-file", os.fspath(status_path)))
    return _run_command(args, timeout=timeout, cancel_file=cancel_path,
                        allow_interrupt=True)


def _payload_error(result: _ProcessResult, action: str) -> str:
    payload = result.payload or {}
    if result.cancelled or payload.get("status") == "cancelled":
        return "ComfyUI 已取消离线任务"
    error = payload.get("error") or payload.get("reason")
    if isinstance(error, Mapping):
        error = error.get("message") or error.get("reason")
    if error:
        return str(error)
    stderr = result.stderr.strip()
    if stderr:
        return stderr.splitlines()[-1]
    return f"{action}未返回可诊断的 JSON 结果（退出码 {result.returncode}）"


def _make_output_video(path: pathlib.Path) -> Any:
    try:
        from comfy_api.latest import InputImpl
    except ImportError as exc:
        raise OfflineNodeError(
            "当前 ComfyUI 没有 comfy_api.latest.InputImpl.VideoFromFile；"
            "请升级到支持原生 VIDEO 的 ComfyUI，或在外部查看输出路径") from exc
    constructor = getattr(InputImpl, "VideoFromFile", None)
    if not callable(constructor):
        raise OfflineNodeError("当前 ComfyUI 缺少 VideoFromFile VIDEO 构造器")
    try:
        return constructor(os.fspath(path))
    except Exception as exc:
        raise OfflineNodeError(f"无法把成片包装为 ComfyUI VIDEO：{exc}") from exc


class _OfflineNodeBase:
    CATEGORY = "XeSS R4 离线视频"
    OUTPUT_NODE = False
    MODE = "sr"

    @classmethod
    def _compat_inputs(cls):
        # Append optional widgets: existing R4 workflows retain widget order.
        return {"arc_a_compat": ("BOOLEAN", {
            "default": False, "display_name": "Arc A 系列兼容模式",
            "tooltip": "GPU Block、GPU DIS、AMD 光流：Arc A 显卡出现绿/紫色竖条时开启。改为 GPU 转 RGBA 后共享；不是抗竖纹滤镜。正常画面保持关闭。",
        })}

    @classmethod
    def _common_inputs(cls, mode: str) -> dict[str, Any]:
        backend_choices = _backend_choices(mode)
        backend_default = (BACKEND_LABELS["gpu-block"]
                           if BACKEND_LABELS["gpu-block"] in backend_choices
                           else backend_choices[0])
        inputs = {
            "backend": (backend_choices, {
                "default": backend_default,
                "display_name": "后端",
                "tooltip": "只显示 capabilities 报告为可用的后端；plan 会再次按真实组合校验。",
            }),
            "encoder": (_encoder_choices(mode), {
                "default": ENCODER_LABELS["auto"],
                "display_name": "编码器",
                "tooltip": "编码器能力按所选后端由 plan 验证，不支持时明确拒绝。",
            }),
            "depth": (tuple(DEPTH_LABELS.values()), {
                "default": DEPTH_LABELS["ai"],
                "display_name": "深度",
                "tooltip": "Intel 视频接口不使用 XeSS 深度；其他后端按 capabilities 验证。",
            }),
            "sharpen": ("BOOLEAN", {
                "default": False, "display_name": "末尾锐化",
                "tooltip": "默认关闭；只在后端明确支持时执行，组合管线最多末尾一次。",
            }),
            "five_frame": ("BOOLEAN", {
                "default": False, "display_name": "五帧融合",
                "tooltip": "适合静态画面，动态可能反效果。GPU Block/GPU DIS/AMD 光流使用当前帧＋前四帧；CPU DIS 使用前后各两帧。",
            }),
            "anti_stripe": ("BOOLEAN", {
                "default": False, "display_name": "抗竖纹",
                "tooltip": "CPU DIS、GPU Block、GPU DIS、AMD 光流支持超分和超分后补帧；默认关闭。Intel 视频接口不支持。",
            }),
        }
        if mode == "fg":
            # SR-only controls never belong on the independent FG node.
            del inputs["five_frame"], inputs["anti_stripe"]
        return inputs

    @classmethod
    def VALIDATE_INPUTS(cls, backend=None, encoder=None, sharpen=False, depth=None,
                        five_frame=False, anti_stripe=False, arc_a_compat=False, **_kwargs):
        """Give Comfy an early capability error while retaining plan as authority."""
        resolved_backend = _canonical_backend(backend or BACKEND_LABELS["auto"])
        if not isinstance(arc_a_compat, bool):
            return "Arc A 系列兼容模式必须是布尔开关。"
        if arc_a_compat and resolved_backend not in ("gpu-block", "gpu-dis", "amd-of", "auto"):
            return "Arc A 系列兼容模式仅支持 GPU Block、GPU DIS、AMD 光流。"
        if resolved_backend in ("gpu-block", "gpu-dis", "amd-of") and depth is not None and _canonical_depth(depth) != "ai":
            return "此 GPU 路线需要 AI 深度，请选择「AI 深度」。"
        if resolved_backend == "auto":
            return True
        try:
            payload = _capabilities()
        except OfflineNodeError:
            # A missing runtime is diagnosed by execution; do not break node
            # discovery merely because the portable runtime is not installed.
            return True
        item = _backend_item(resolved_backend, payload)
        if not _supports_mode(item, cls.MODE):
            return f"后端 {resolved_backend} 当前不支持 {cls.MODE}。"
        if encoder is not None:
            resolved_encoder = _canonical_encoder(encoder)
            encoders = item.get("encoders", []) if item else []
            if resolved_encoder != "auto" and resolved_encoder not in encoders:
                return f"后端 {resolved_backend} 不支持编码器 {resolved_encoder}。"
        for enabled, effect in ((sharpen, "sharpen"),
                                (five_frame, "five_frame"),
                                (anti_stripe, "anti_stripe")):
            if enabled and not _supports_effect(item, effect, cls.MODE):
                return f"后端 {resolved_backend} 的当前模式不支持 {effect}；不会静默忽略。"
        return True

    def _execute(self, *, video: Any, mode: str, backend: str, scale: float,
                 encoder: str, depth: str, sharpen: bool, five_frame: bool,
                 anti_stripe: bool, arc_a_compat: bool = False):
        source = _source_path(video)
        if not os.environ.get("XESS_RUNTIME_ROOT"):
            try:
                from . import runtime_manager
            except ImportError:
                import runtime_manager
            try:
                runtime_manager.ensure_runtime()
                clear_capability_cache()
            except runtime_manager.RuntimeManagerError as exc:
                raise OfflineNodeError(str(exc)) from exc
        resolved_encoder = _canonical_encoder(encoder)
        resolved_backend = _canonical_backend(backend)
        if resolved_backend == "auto":
            payload = _capabilities()
            priority = ("gpu-block", "intel-vpl-ai", "cpu-dis", "gpu-dis", "amd-of")
            for candidate in priority:
                if _supports_mode(_backend_item(candidate, payload), mode):
                    resolved_backend = candidate
                    break
            if resolved_backend == "auto":
                raise OfflineNodeError(
                    f"没有 capabilities 声明可用的 {mode} 后端；不会自动伪装成 CPU 或静默改路线")
        job, request_path, cancel_path, status_path, output = _new_job(mode, resolved_encoder)
        request = _request(
            mode=mode,
            source=source,
            output=output,
            backend=resolved_backend,
            scale=scale,
            encoder=resolved_encoder,
            depth=_canonical_depth(depth),
            sharpen=sharpen,
            five_frame=five_frame,
            anti_stripe=anti_stripe,
            arc_a_compat=arc_a_compat,
        )
        _write_request(request_path, request)
        plan = _invoke_toolbox("plan", request_path, timeout=45.0)
        if plan.returncode != 0 or not plan.payload or plan.payload.get("ok") is False:
            raise OfflineNodeError("离线计划拒绝请求：" + _payload_error(plan, "plan"))
        planned = plan.payload.get("plan")
        run = _invoke_toolbox("run", request_path, cancel_path=cancel_path,
                              status_path=status_path, timeout=24 * 60 * 60)
        if run.returncode != 0 or run.cancelled or not run.payload:
            raise OfflineNodeError("离线任务失败：" + _payload_error(run, "run"))
        payload = run.payload
        if payload.get("status") != "complete" or payload.get("ok") is False:
            raise OfflineNodeError("离线任务未完成：" + _payload_error(run, "run"))
        raw_output = payload.get("output") or os.fspath(output)
        final_output = pathlib.Path(str(raw_output)).expanduser().resolve()
        if not final_output.is_file() or final_output.stat().st_size <= 0:
            raise OfflineNodeError(f"运行时声称完成，但成片不存在或为空：{final_output}")
        video_out = _make_output_video(final_output)
        detail = payload.get("message") or payload.get("diagnostic") or "R4 离线成片完成"
        backend_name = (planned or {}).get("backend") if isinstance(planned, Mapping) else None
        if backend_name:
            detail = f"后端={backend_name} | {detail}"
        # Comfy's execution engine requires one returned item per declared
        # RETURN_TYPES socket. Diagnostics belong in the log, not a third,
        # undeclared output that fails after a successful video render.
        print(f"[XeSS R4] {detail}", file=sys.stderr)
        return video_out, os.fspath(final_output)


class XeSSR4OfflineSuperResolution(_OfflineNodeBase):
    """R4 独立中文 VIDEO→VIDEO 超分节点。"""

    DESCRIPTION = "视频超分；默认 GPU Block，CPU DIS 为稳定路线。"
    FUNCTION = "upscale_video"
    MODE = "sr"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("超分后视频", "成片绝对路径")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO", {"display_name": "输入视频",
                                  "tooltip": "仅接受能定位到保留视频文件的 Load Video VIDEO。"}),
            "scale": (tuple(SCALE_LABELS), {"default": "1.5×", "display_name": "SR 倍率",
                                              "tooltip": "常用 1.33×、1.5×、2×；选择自定义后填写自定义倍率。"}),
            "custom_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0,
                                         "step": 0.01, "display_name": "自定义倍率"}),
            **cls._common_inputs("sr"),
        }, "optional": cls._compat_inputs()}

    def upscale_video(self, video, scale="1.5×", custom_scale=1.5,
                      backend=BACKEND_LABELS["auto"], encoder=ENCODER_LABELS["auto"],
                      depth=DEPTH_LABELS["ai"], sharpen=False, five_frame=False,
                      anti_stripe=False, arc_a_compat=False):
        return self._execute(video=video, mode="sr", backend=backend,
                             scale=_scale_value(scale, custom_scale), encoder=encoder,
                             depth=depth, sharpen=sharpen, five_frame=five_frame,
                             anti_stripe=anti_stripe, arc_a_compat=arc_a_compat)


class XeSSR4OfflineFrameGeneration(_OfflineNodeBase):
    """R4 独立中文 VIDEO→VIDEO 2× 插帧节点。"""

    DESCRIPTION = "R4 离线视频 2× 插帧：FG 倍率固定为 2，调用统一 offline_toolbox。"
    FUNCTION = "interpolate_video"
    MODE = "fg"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("2× 插帧后视频", "成片绝对路径")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO", {"display_name": "输入视频",
                                  "tooltip": "仅接受能定位到保留视频文件的 Load Video VIDEO。"}),
            **cls._common_inputs("fg"),
        }, "optional": cls._compat_inputs()}

    @classmethod
    def VALIDATE_INPUTS(cls, backend=None, encoder=None, sharpen=False, depth=None,
                        five_frame=False, anti_stripe=False, **kwargs):
        # Accept stale API/workflow fields from older nodes, but never pass
        # these removed SR-only settings to FG. Keep all other validation.
        return super().VALIDATE_INPUTS(backend=backend, encoder=encoder, depth=depth,
                                      sharpen=sharpen, five_frame=False,
                                      anti_stripe=False, **kwargs)

    def interpolate_video(self, video, backend=BACKEND_LABELS["auto"],
                          encoder=ENCODER_LABELS["auto"], depth=DEPTH_LABELS["ai"],
                          sharpen=False, five_frame=False, anti_stripe=False,
                          arc_a_compat=False):
        # These two optional arguments are a compatibility adapter only;
        # they are deliberately absent from INPUT_TYPES and new workflows.
        if five_frame or anti_stripe:
            print("[XeSS R4] 旧工作流迁移：独立插帧已移除五帧融合和抗竖纹，"
                  "本次不执行这两项；末尾锐化保持原设置。", file=sys.stderr)
        return self._execute(video=video, mode="fg", backend=backend, scale=2.0,
                             encoder=encoder, depth=depth, sharpen=sharpen,
                             five_frame=False, anti_stripe=False, arc_a_compat=arc_a_compat)


class XeSSR4OfflineSuperResolutionFrameGeneration(_OfflineNodeBase):
    """R4 独立中文 VIDEO→VIDEO SR→2×FG 组合节点。"""

    DESCRIPTION = "R4 离线视频组合：先按倍率超分，再固定 2× 插帧，后端由统一计划决定。"
    FUNCTION = "process_video"
    MODE = "sr-fg"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("SR→2×FG 成片", "成片绝对路径")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO", {"display_name": "输入视频",
                                  "tooltip": "仅接受能定位到保留视频文件的 Load Video VIDEO。"}),
            "scale": (tuple(SCALE_LABELS), {"default": "1.5×", "display_name": "SR 倍率",
                                              "tooltip": "组合节点只对 SR 使用该倍率；FG 固定 2×。"}),
            "custom_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0,
                                         "step": 0.01, "display_name": "自定义倍率"}),
            **cls._common_inputs("sr-fg"),
        }, "optional": cls._compat_inputs()}

    def process_video(self, video, scale="1.5×", custom_scale=1.5,
                      backend=BACKEND_LABELS["auto"], encoder=ENCODER_LABELS["auto"],
                      depth=DEPTH_LABELS["ai"], sharpen=False, five_frame=False,
                      anti_stripe=False, arc_a_compat=False):
        return self._execute(video=video, mode="sr-fg", backend=backend,
                             scale=_scale_value(scale, custom_scale), encoder=encoder,
                             depth=depth, sharpen=sharpen, five_frame=five_frame,
                             anti_stripe=anti_stripe, arc_a_compat=arc_a_compat)


NODE_CLASS_MAPPINGS = {
    "XeSSR4OfflineSuperResolution": XeSSR4OfflineSuperResolution,
    "XeSSR4OfflineFrameGeneration": XeSSR4OfflineFrameGeneration,
    "XeSSR4OfflineSuperResolutionFrameGeneration": XeSSR4OfflineSuperResolutionFrameGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XeSSR4OfflineSuperResolution": "R4 离线视频超分（独立）",
    "XeSSR4OfflineFrameGeneration": "R4 离线视频 2× 插帧（独立）",
    "XeSSR4OfflineSuperResolutionFrameGeneration": "R4 离线视频超分→2×插帧（独立）",
}

__all__ = [
    "OfflineNodeError", "XeSSR4OfflineSuperResolution",
    "XeSSR4OfflineFrameGeneration", "XeSSR4OfflineSuperResolutionFrameGeneration",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "clear_capability_cache",
]
