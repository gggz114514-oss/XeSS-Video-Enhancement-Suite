"""Pinned isolated R4 runtime. No pip or Comfy changes; R3 engine is untouched."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import sys
import threading
import time
import urllib.request
import uuid
import zipfile

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "runtime_manifest.json"
RUNTIME_ENV = "COMFYUI_XESS_RUNTIME"
ASSET_ENV = "COMFYUI_XESS_RUNTIME_ASSET"
STATE_NAME = ".runtime-state.json"
_thread = None
_thread_lock = threading.Lock()


class RuntimeManagerError(RuntimeError):
    pass


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise RuntimeManagerError(f"无法读取运行时清单：{path}：{exc}") from exc


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeManagerError(f"不安全的包内路径：{value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or PureWindowsPath(value).drive or any(
        s in ("", ".", "..") or s.endswith((".", " ")) or
        any(c in s for c in ':<>"|?*') or _reserved_windows_name(s)
        for s in value.split("/")
    ):
        raise RuntimeManagerError(f"不安全的包内路径：{value!r}")
    return Path(*p.parts)


def _reserved_windows_name(value):
    # Python 3.13 deprecates PurePath.is_reserved; retain older hosts too.
    if hasattr(os.path, "isreserved"):
        return os.path.isreserved(value)
    stem = value.split(".", 1)[0].upper()
    return stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or bool(
        re.fullmatch(r"(COM|LPT)[1-9¹²³]", stem))


def load_manifest():
    m = _load_json(MANIFEST_PATH)
    if m.get("schema_version") != 2 or m.get("layout") != "comfy-r4-nested-v1":
        raise RuntimeManagerError("运行时清单版本不匹配，请完整更新本节点目录。")
    for name in ("runtime_version", "asset_name", "archive_root"):
        _relative(m.get(name))
        if "/" in m[name]:
            raise RuntimeManagerError("运行时标识不能包含子目录。")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", m.get("sha256", "")):
        raise RuntimeManagerError("运行时尚未封包：缺少有效 SHA256，不能下载。")
    if not m.get("file_hashes") or not m.get("required_files"):
        raise RuntimeManagerError("运行时文件校验清单为空。")
    for path, digest in m["file_hashes"].items():
        _relative(path)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise RuntimeManagerError("文件哈希无效：" + path)
    if not set(m["required_files"]).issubset(m["file_hashes"]):
        raise RuntimeManagerError("必需文件没有全部纳入哈希校验。")
    if m.get("archive_size", 0) <= 0 or m.get("installed_size", 0) <= 0:
        raise RuntimeManagerError("运行时大小缺失，无法做空间检查。")
    if not re.fullmatch(r"https://github\.com/gggz114514-oss/XeSS-Video-Enhancement-Suite/releases/download/[^/]+/[^/]+", m.get("download_url", "")):
        raise RuntimeManagerError("下载地址必须是本仓库的固定 Release 资产。")
    return m


def runtime_base():
    value = os.environ.get(RUNTIME_ENV, "").strip()
    if value:
        p = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
        return p.parent if p.name.casefold() == "engine" else p
    return REPO_ROOT / ".runtime"


def default_engine(manifest=None):
    m = manifest or load_manifest()
    return runtime_base() / "versions" / (m["runtime_version"] + "-" + m["sha256"][:16])


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _fingerprints(engine, m):
    result = {}
    for name in m["file_hashes"]:
        f = Path(engine) / _relative(name)
        if not f.is_file() or f.is_symlink():
            return None
        s = f.stat()
        result[name] = [s.st_size, s.st_mtime_ns]
    return result


def engine_compatible(engine, manifest=None, *, full=False):
    m = manifest or load_manifest()
    try:
        fingerprints = _fingerprints(engine, m)
        if fingerprints is None:
            return False
        if not full:
            try:
                state = _load_json(Path(engine) / STATE_NAME)
                if state.get("asset_sha256") == m["sha256"] and state.get("files") == fingerprints:
                    return True
            except RuntimeManagerError:
                pass
        return all(sha256_file(Path(engine) / _relative(n)).lower() == h.lower()
                   for n, h in m["file_hashes"].items())
    except OSError:
        return False


def _write_state(engine, m):
    data = dict(asset_sha256=m["sha256"], runtime_version=m["runtime_version"],
                installed_unix=time.time(), files=_fingerprints(engine, m))
    temporary = Path(engine) / (STATE_NAME + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, Path(engine) / STATE_NAME)


@contextlib.contextmanager
def _install_lock(root, timeout=1200):
    root.mkdir(parents=True, exist_ok=True)
    # OS lock releases on crash; no unsafe stale PID/mtime heuristics.
    with (root / "install.lock").open("a+b") as f:
        # Windows permits locking beyond EOF. Do not write an initialization
        # byte: another process may have acquired its lock on the empty file.
        deadline = time.monotonic() + timeout
        while True:
            try:
                f.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeManagerError("等待其他运行时安装进程超时，请稍后重试。")
                time.sleep(.2)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def _safe_remove(path, root):
    p, r = Path(path).resolve(), Path(root).resolve()
    if p == r or r not in p.parents or Path(path).is_symlink():
        raise RuntimeManagerError("拒绝清理工作目录外的路径：" + str(path))
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


def _check_space(root, m):
    required = m["archive_size"] + m["installed_size"] + 512 * 1024**2
    free = shutil.disk_usage(root).free
    if free < required:
        raise RuntimeManagerError(f"运行时安装盘空间不足：需要 {required/1024**3:.2f} GiB，剩余 {free/1024**3:.2f} GiB。路径：{root}")


def _download(url, destination, expected_size):
    part = destination.with_suffix(".zip.partial")
    print("[XeSS R4] 正在下载配套运行时；无需重装 ComfyUI：" + url, flush=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-XeSS-R4"})
        with urllib.request.urlopen(request, timeout=30) as response, part.open("wb") as out:
            size, last = 0, time.monotonic()
            while True:
                block = response.read(1024**2)
                if not block:
                    break
                size += len(block)
                if size > expected_size:
                    raise RuntimeManagerError("下载大小超过清单，已拒绝该文件。")
                out.write(block)
                if time.monotonic() - last >= 5:
                    print(f"[XeSS R4] 下载 {size/1024**2:.0f}/{expected_size/1024**2:.0f} MiB", flush=True)
                    last = time.monotonic()
            if size != expected_size:
                raise RuntimeManagerError("下载提前结束，请检查网络后重新执行节点。")
        os.replace(part, destination)
    finally:
        part.unlink(missing_ok=True)


def _extract_archive(archive, root, m):
    staging = root / ("installing-" + uuid.uuid4().hex)
    staging.mkdir()
    seen, size = set(), 0
    try:
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                relative = _relative(member.filename.rstrip("/"))
                if relative.parts[0] != m["archive_root"]:
                    raise RuntimeManagerError("压缩包根目录不匹配。")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise RuntimeManagerError("不允许压缩包中的符号链接。")
                if member.is_dir():
                    continue
                child = Path(*relative.parts[1:]).as_posix()
                key = child.casefold()
                if key in seen or child not in m["file_hashes"]:
                    raise RuntimeManagerError("压缩包包含重复或未声明的文件：" + child)
                seen.add(key)
                size += member.file_size
                if size > m["installed_size"]:
                    raise RuntimeManagerError("解压大小超过清单。")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024**2)
        candidate = staging / m["archive_root"]
        if len(seen) != len(m["file_hashes"]) or size != m["installed_size"] or not engine_compatible(candidate, m, full=True):
            raise RuntimeManagerError("运行时解压校验失败，原有版本未改动。")
        return candidate
    except BaseException:
        _safe_remove(staging, root)
        raise


def ensure_runtime(*, force=False, asset=None):
    m = load_manifest()
    override = asset or os.environ.get(ASSET_ENV, "").strip()
    if m.get("release_status") != "published" and not override:
        raise RuntimeManagerError("配套运行时尚未发布；不会下载占位文件。")
    root, engine = runtime_base(), default_engine(m)
    if not force and engine_compatible(engine, m):
        return engine
    try:
        with _install_lock(root):
            if engine_compatible(engine, m, full=force):
                return engine
            if not override and os.environ.get("COMFYUI_XESS_SKIP_RUNTIME_DOWNLOAD") == "1":
                raise RuntimeManagerError("自动下载已关闭，请先手动安装本版本运行时。")
            _check_space(root, m)
            archive = Path(override).expanduser().resolve() if override else root / m["asset_name"]
            candidate = None
            try:
                if not override:
                    _download(m["download_url"], archive, m["archive_size"])
                if archive.stat().st_size != m["archive_size"] or sha256_file(archive).lower() != m["sha256"].lower():
                    raise RuntimeManagerError("运行时 SHA256/大小校验失败；原有版本保留，请重新下载。")
                candidate = _extract_archive(archive, root, m)
                _write_state(candidate, m)
                engine.parent.mkdir(parents=True, exist_ok=True)
                backup = None
                if engine.exists():
                    backup = root / ("repair-backup-" + uuid.uuid4().hex)
                    engine.rename(backup)
                try:
                    candidate.rename(engine)
                except BaseException:
                    if backup is not None:
                        backup.rename(engine)
                    raise
                print("[XeSS R4] 运行时校验通过：" + str(engine), flush=True)
                return engine
            finally:
                if candidate is not None and candidate.parent.exists():
                    _safe_remove(candidate.parent, root)
                if not override:
                    archive.unlink(missing_ok=True)
    except RuntimeManagerError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeManagerError(f"运行时安装失败，原有版本未覆盖。检查网络、空间和文件占用后重试：{exc}") from exc


def start_background_update():
    global _thread
    if os.environ.get("XESS_RUNTIME_ROOT") or os.environ.get("COMFYUI_XESS_SKIP_RUNTIME_DOWNLOAD") == "1":
        return
    with _thread_lock:
        if _thread is not None:
            return
        def update():
            try:
                ensure_runtime()
            except Exception as exc:
                print("[XeSS R4] " + str(exc) + "；节点仍可加载，执行时可重试。", file=sys.stderr)
        _thread = threading.Thread(target=update, name="XeSS-runtime-install", daemon=True)
        _thread.start()


def main():
    parser = argparse.ArgumentParser(description="XeSS R4 运行时：不修改 ComfyUI Python")
    parser.add_argument("command", choices=("ensure", "status"))
    parser.add_argument("--asset")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "ensure":
            print(ensure_runtime(force=args.force, asset=args.asset))
            return 0
        m = load_manifest()
        engine = default_engine(m)
        ok = engine_compatible(engine, m, full=args.force)
        print(json.dumps(dict(engine=str(engine), runtime_version=m["runtime_version"], compatible=ok), ensure_ascii=False))
        return 0 if ok else 1
    except RuntimeManagerError as exc:
        print("[XeSS R4] " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
