from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
import urllib.request
import zipfile


REPO_ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "runtime_manifest.json"
OVERLAY_ROOT = REPO_ROOT / "pipeline"
STATE_NAME = ".runtime-state.json"
RUNTIME_ENV = "COMFYUI_XESS_RUNTIME"
ASSET_ENV = "COMFYUI_XESS_RUNTIME_ASSET"


class RuntimeManagerError(RuntimeError):
    pass


def _load_json(path: pathlib.Path) -> dict:
    try:
        # Windows PowerShell 5 writes `Set-Content -Encoding UTF8` with a BOM,
        # while PowerShell 7 and Python normally do not.  Accept both forms so
        # a runtime installed by install_runtime.bat is immediately readable.
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeManagerError(f"cannot read {path}: {exc}") from exc


def load_manifest() -> dict:
    manifest = _load_json(MANIFEST_PATH)
    required = (
        "schema_version", "runtime_version", "asset_name", "download_url",
        "sha256", "archive_root", "required_files", "file_hashes",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        raise RuntimeManagerError(f"runtime manifest is missing: {', '.join(missing)}")
    if manifest["schema_version"] != 1:
        raise RuntimeManagerError(
            f"unsupported runtime manifest schema: {manifest['schema_version']}"
        )
    return manifest


def runtime_base() -> pathlib.Path:
    configured = os.environ.get(RUNTIME_ENV, "").strip()
    if configured:
        path = pathlib.Path(os.path.expandvars(os.path.expanduser(configured)))
        if path.name.casefold() == "engine":
            return path.resolve().parent
        return path.resolve()
    return (REPO_ROOT / ".runtime").resolve()


def default_engine() -> pathlib.Path:
    return runtime_base() / "engine"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: str) -> pathlib.Path:
    candidate = pathlib.PurePosixPath(path.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeManagerError(f"unsafe relative path in manifest: {path}")
    return pathlib.Path(*candidate.parts)


def engine_compatible(engine: os.PathLike[str] | str, manifest: dict | None = None) -> bool:
    engine_path = pathlib.Path(engine)
    manifest = manifest or load_manifest()
    if not engine_path.is_dir():
        return False
    for name in manifest["required_files"]:
        if not (engine_path / _relative(name)).is_file():
            return False
    state_path = engine_path / STATE_NAME
    if state_path.is_file():
        try:
            state = _load_json(state_path)
            if (state.get("runtime_version") == manifest["runtime_version"] and
                    str(state.get("asset_sha256", "")).casefold() ==
                    str(manifest["sha256"]).casefold()):
                return True
        except RuntimeManagerError:
            pass
    for name, expected in manifest["file_hashes"].items():
        path = engine_path / _relative(name)
        if not path.is_file() or sha256_file(path).casefold() != str(expected).casefold():
            return False
    return True


def prepare_existing_engine(engine: os.PathLike[str] | str) -> bool:
    engine_path = pathlib.Path(engine).resolve()
    manifest = load_manifest()
    if not engine_compatible(engine_path, manifest):
        return False
    sync_overlay(engine_path)
    _write_state(engine_path, manifest)
    return True


def _same_file(source: pathlib.Path, destination: pathlib.Path) -> bool:
    if not destination.is_file() or source.stat().st_size != destination.stat().st_size:
        return False
    return sha256_file(source) == sha256_file(destination)


def sync_overlay(engine: os.PathLike[str] | str) -> int:
    engine_path = pathlib.Path(engine).resolve()
    if not engine_path.is_dir():
        raise RuntimeManagerError(f"runtime engine does not exist: {engine_path}")
    if not OVERLAY_ROOT.is_dir():
        raise RuntimeManagerError(f"pipeline source is missing: {OVERLAY_ROOT}")
    copied = 0
    for source in sorted(OVERLAY_ROOT.rglob("*")):
        if not source.is_file() or source.suffix.casefold() in {".pyc", ".pyo"}:
            continue
        if "__pycache__" in source.parts:
            continue
        destination = engine_path / source.relative_to(OVERLAY_ROOT)
        if _same_file(source, destination):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".sync-partial")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        copied += 1
    return copied


def _safe_remove(path: pathlib.Path, root: pathlib.Path) -> None:
    root = root.resolve()
    candidate = path.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeManagerError(f"refusing to remove path outside runtime root: {candidate}") from exc
    if candidate == root:
        raise RuntimeManagerError(f"refusing to remove runtime root: {candidate}")
    if candidate.is_dir():
        shutil.rmtree(candidate)
    elif candidate.exists():
        candidate.unlink()


@contextlib.contextmanager
def _install_lock(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "install.lock"
    deadline = time.monotonic() + 20 * 60
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()} time={time.time()}\n".encode("ascii"))
        except FileExistsError:
            try:
                stale = time.time() - lock.stat().st_mtime > 60 * 60
            except OSError:
                stale = False
            if stale:
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise RuntimeManagerError(f"timed out waiting for runtime installer: {lock}")
            time.sleep(1.0)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink(missing_ok=True)


def _check_space(root: pathlib.Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    archive = int(manifest.get("archive_size", 0))
    installed = int(manifest.get("installed_size", 0))
    required = archive + installed + 512 * 1024 * 1024
    free = shutil.disk_usage(root).free
    if free < required:
        raise RuntimeManagerError(
            f"not enough free space at {root}: need about {required / 1024**3:.2f} GiB, "
            f"free {free / 1024**3:.2f} GiB"
        )


def _download(url: str, destination: pathlib.Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "ComfyUI-XeSS-Runtime/1.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    print(f"[XeSS runtime] downloading {url}", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        if not append:
            offset = 0
        received = offset
        last_report = time.monotonic()
        with partial.open(mode) as output:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                received += len(block)
                if time.monotonic() - last_report >= 5:
                    print(f"[XeSS runtime] received {received / 1024**2:.1f} MiB", flush=True)
                    last_report = time.monotonic()
    os.replace(partial, destination)


def _extract_archive(archive: pathlib.Path, root: pathlib.Path, manifest: dict) -> pathlib.Path:
    staging = root / f"installing-{os.getpid()}-{int(time.time())}"
    if staging.exists():
        _safe_remove(staging, root)
    staging.mkdir(parents=True)
    archive_root = str(manifest["archive_root"]).strip("/\\")
    limit = max(int(manifest.get("installed_size", 0)) * 2, 1024 * 1024 * 1024)
    extracted = 0
    try:
        with zipfile.ZipFile(archive, "r") as package:
            for member in package.infolist():
                pure = pathlib.PurePosixPath(member.filename.replace("\\", "/"))
                if not pure.parts or pure.parts[0] != archive_root:
                    raise RuntimeManagerError(f"unexpected archive entry: {member.filename}")
                if pure.is_absolute() or ".." in pure.parts:
                    raise RuntimeManagerError(f"unsafe archive entry: {member.filename}")
                extracted += member.file_size
                if extracted > limit:
                    raise RuntimeManagerError("runtime archive expands beyond the manifest safety limit")
                target = staging.joinpath(*pure.parts)
                resolved = target.resolve()
                try:
                    resolved.relative_to(staging.resolve())
                except ValueError as exc:
                    raise RuntimeManagerError(f"unsafe archive target: {resolved}") from exc
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
        candidate = staging / archive_root
        if not engine_compatible(candidate, manifest):
            raise RuntimeManagerError("extracted runtime failed file/hash validation")
        return candidate
    except BaseException:
        _safe_remove(staging, root)
        raise


def _write_state(engine: pathlib.Path, manifest: dict) -> None:
    state = {
        "runtime_version": manifest["runtime_version"],
        "asset_name": manifest["asset_name"],
        "asset_sha256": manifest["sha256"],
        "installed_unix": time.time(),
    }
    temporary = engine / (STATE_NAME + ".partial")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, engine / STATE_NAME)


def _activate(candidate: pathlib.Path, root: pathlib.Path, engine: pathlib.Path) -> None:
    backup = root / f"engine-backup-{os.getpid()}"
    if backup.exists():
        _safe_remove(backup, root)
    had_previous = engine.exists()
    if had_previous:
        engine.rename(backup)
    try:
        candidate.rename(engine)
    except BaseException:
        if had_previous and backup.exists() and not engine.exists():
            backup.rename(engine)
        raise
    if backup.exists():
        _safe_remove(backup, root)
    staging = candidate.parent
    if staging.exists():
        _safe_remove(staging, root)


def ensure_runtime(*, force: bool = False, asset: str | None = None) -> pathlib.Path:
    manifest = load_manifest()
    root = runtime_base()
    engine = root / "engine"
    if not force and engine_compatible(engine, manifest):
        copied = sync_overlay(engine)
        _write_state(engine, manifest)
        if copied:
            print(f"[XeSS runtime] synchronized {copied} updated pipeline files", flush=True)
        return engine

    with _install_lock(root):
        if not force and engine_compatible(engine, manifest):
            sync_overlay(engine)
            _write_state(engine, manifest)
            return engine
        _check_space(root, manifest)
        override = asset or os.environ.get(ASSET_ENV, "").strip()
        downloaded = False
        if override:
            archive = pathlib.Path(os.path.expandvars(os.path.expanduser(override))).resolve()
            if not archive.is_file():
                raise RuntimeManagerError(f"runtime asset does not exist: {archive}")
        else:
            downloads = root / "downloads"
            downloads.mkdir(parents=True, exist_ok=True)
            archive = downloads / manifest["asset_name"]
            if not archive.is_file() or sha256_file(archive).casefold() != manifest["sha256"].casefold():
                _download(manifest["download_url"], archive)
                downloaded = True
        actual = sha256_file(archive)
        if actual.casefold() != manifest["sha256"].casefold():
            if downloaded:
                archive.unlink(missing_ok=True)
            raise RuntimeManagerError(
                f"runtime SHA256 mismatch: expected {manifest['sha256']}, got {actual}"
            )
        candidate = None
        try:
            candidate = _extract_archive(archive, root, manifest)
            sync_overlay(candidate)
            _write_state(candidate, manifest)
            _activate(candidate, root, engine)
        finally:
            if candidate is not None and candidate.parent.exists():
                _safe_remove(candidate.parent, root)
            if downloaded:
                archive.unlink(missing_ok=True)
        print(f"[XeSS runtime] ready: {engine}", flush=True)
        return engine


def _status(engine: pathlib.Path) -> int:
    manifest = load_manifest()
    compatible = engine_compatible(engine, manifest)
    print(json.dumps({
        "engine": os.fspath(engine),
        "runtime_version": manifest["runtime_version"],
        "compatible": compatible,
        "state": _load_json(engine / STATE_NAME) if (engine / STATE_NAME).is_file() else None,
    }, ensure_ascii=False, indent=2))
    return 0 if compatible else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Install and synchronize the XeSS fixed runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("ensure")
    install.add_argument("--force", action="store_true")
    install.add_argument("--asset", default="")
    status = sub.add_parser("status")
    status.add_argument("--engine", default="")
    sync = sub.add_parser("sync")
    sync.add_argument("--engine", default="")
    args = parser.parse_args()
    try:
        if args.command == "ensure":
            ensure_runtime(force=args.force, asset=args.asset or None)
            return 0
        engine = pathlib.Path(args.engine).resolve() if args.engine else default_engine()
        if args.command == "status":
            return _status(engine)
        copied = sync_overlay(engine)
        print(f"[XeSS runtime] synchronized {copied} pipeline files")
        return 0
    except RuntimeManagerError as exc:
        print(f"[XeSS runtime] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
