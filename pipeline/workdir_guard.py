#!/usr/bin/env python3
"""Disk-safe job workspace management for the XeSS portable tools."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


GIB = 1024**3
MANIFEST_NAME = "job_manifest.json"


class WorkdirError(RuntimeError):
    """Raised before any large intermediate is created."""


def gib(value: int | float) -> float:
    return float(value) / GIB


def format_gib(value: int | float) -> str:
    return f"{gib(value):.2f} GiB"


def system_drive() -> str:
    configured = os.environ.get("SystemDrive", "C:")
    return os.path.splitdrive(os.path.abspath(configured + os.sep))[0].upper()


def drive_of(path: str | os.PathLike[str]) -> str:
    return os.path.splitdrive(os.path.abspath(os.fspath(path)))[0].upper()


def is_system_drive(path: str | os.PathLike[str]) -> bool:
    return bool(drive_of(path)) and drive_of(path) == system_drive()


def _candidate_root(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix and not resolved.is_dir():
        resolved = resolved.parent
    return resolved / ".xess-work"


def select_work_root(
    *, explicit: str, output_dir: str, package_dir: str, input_path: str,
    allow_system_drive: bool,
) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if is_system_drive(root) and not allow_system_drive:
            raise WorkdirError(
                f"work directory {root} is on the system drive; explicitly add "
                "--allow-system-drive-temp only if this is intentional"
            )
        return root
    rejected: list[str] = []
    for candidate in (output_dir, package_dir, input_path):
        if not candidate:
            continue
        root = _candidate_root(candidate)
        if is_system_drive(root) and not allow_system_drive:
            rejected.append(str(root))
            continue
        return root
    # Some systems genuinely have no data drive. In that case the streaming
    # pipeline is still safe to use on the system drive: preflight() keeps a
    # larger 25 GiB reserve there and all job directories are cleaned on exit.
    # An explicitly requested system-drive path remains opt-in above.
    if rejected:
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        fallback = (Path(local_app_data) / "XeSS-Video-Suite" / "work"
                    if local_app_data else Path(rejected[0]))
        print(
            f"[XeSS] no non-system drive was found; using guarded system-drive "
            f"workspace {fallback} (25 GiB free-space reserve)",
            file=sys.stderr,
        )
        return fallback.resolve()
    detail = ", ".join(rejected) if rejected else "no usable candidates"
    raise WorkdirError(
        "no non-system-drive work directory is available; use --work-dir to select "
        f"a data drive with enough free space. Rejected: {detail}"
    )


def estimate_sr_bytes(
    in_w: int, in_h: int, out_w: int, out_h: int, frames: int, *,
    io_mode: str, include_depth: bool, include_mask: bool, chunk_frames: int = 48,
) -> int:
    if io_mode in ("stream", "shared"):
        per_slot = in_w * in_h * (3 + 8 + (4 if include_depth else 0) + (1 if include_mask else 0))
        per_slot += out_w * out_h * 3
        return max(256 * 1024**2, per_slot * 6)
    if io_mode == "chunked":
        active_chunk = estimate_sr_bytes(
            in_w, in_h, out_w, out_h, min(frames, chunk_frames), io_mode="file",
            include_depth=include_depth, include_mask=include_mask,
        )
        compressed_segments = int(out_w * out_h * frames * 1.5)
        return active_chunk + compressed_segments
    pixels_in = in_w * in_h * frames
    pixels_out = out_w * out_h * frames
    total = pixels_in * (3 + 8 + (4 if include_depth else 0) + (1 if include_mask else 0))
    total += pixels_out * 3
    return int(total * 1.10) + 64 * 1024**2


def estimate_fg_bytes(
    width: int, height: int, frames: int, *, io_mode: str, include_depth: bool,
    chunk_frames: int = 48,
) -> int:
    if io_mode in ("stream", "shared"):
        per_slot = width * height * (3 + 8 + (4 if include_depth else 0) + 6)
        return max(384 * 1024**2, per_slot * 6)
    if io_mode == "chunked":
        active_chunk = estimate_fg_bytes(
            width, height, min(frames, chunk_frames), io_mode="file", include_depth=include_depth)
        compressed_segments = int(width * height * max(0, frames * 2 - 1) * 1.5)
        return active_chunk + compressed_segments
    pixels = width * height
    source = pixels * 3 * frames
    motion = pixels * 8 * frames
    depth = pixels * 4 * frames if include_depth else 0
    generated = pixels * 3 * max(0, frames - 1)
    interleaved = pixels * 3 * max(0, frames * 2 - 1)
    return int((source + motion + depth + generated + interleaved) * 1.10) + 64 * 1024**2


def preflight(
    root: Path, estimated_bytes: int, *, reserve_free_gb: float | None,
    max_temp_gb: float | None, label: str,
) -> dict[str, float | int | str]:
    root.mkdir(parents=True, exist_ok=True)
    reserve_gb = reserve_free_gb if reserve_free_gb is not None else (25.0 if is_system_drive(root) else 5.0)
    if reserve_gb < 0:
        raise WorkdirError("--reserve-free-gb cannot be negative")
    if max_temp_gb is not None and max_temp_gb <= 0:
        raise WorkdirError("--max-temp-gb must be greater than zero")
    if max_temp_gb is not None and gib(estimated_bytes) > max_temp_gb:
        raise WorkdirError(
            f"estimated temporary space {format_gib(estimated_bytes)} exceeds "
            f"--max-temp-gb {max_temp_gb:g} GiB"
        )
    usage = shutil.disk_usage(root)
    reserve_bytes = int(reserve_gb * GIB)
    required = int(estimated_bytes * 1.25) + reserve_bytes
    remaining = usage.free - estimated_bytes
    print(f"[{label}] work directory: {root}")
    print(f"[{label}] estimated peak temporary space: {format_gib(estimated_bytes)}")
    print(f"[{label}] currently free: {format_gib(usage.free)}")
    print(f"[{label}] safety reserve: {reserve_gb:.2f} GiB")
    print(f"[{label}] estimated free space after peak: {format_gib(remaining)}")
    if usage.free < required:
        raise WorkdirError(
            f"insufficient free space: at least {format_gib(required)} is required "
            f"(25% margin plus reserve), but only {format_gib(usage.free)} is available"
        )
    return {"root": str(root), "estimated_bytes": estimated_bytes, "free_bytes": usage.free,
            "reserve_bytes": reserve_bytes, "required_bytes": required}


def warn_stale_jobs(root: Path, *, label: str) -> None:
    if not root.is_dir():
        return
    stale: list[Path] = []
    try:
        for child in root.iterdir():
            manifest = child / MANIFEST_NAME
            if not child.is_dir() or not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stale.append(child)
                continue
            if data.get("state") not in ("complete", "cleaned"):
                stale.append(child)
    except OSError as exc:
        print(f"[{label}] unable to inspect stale jobs: {exc}", file=sys.stderr)
        return
    if stale:
        print(f"[{label}] found {len(stale)} incomplete job(s); they will not be deleted automatically:",
              file=sys.stderr)
        for path in stale[:10]:
            print(f"[{label}]   {path}", file=sys.stderr)


def _contained(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _path_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0


@dataclass
class JobWorkspace:
    root: Path
    kind: str
    estimate: dict[str, float | int | str]
    keep: bool = False
    job_dir: Path = field(init=False)
    manifest_path: Path = field(init=False)
    targets: list[Path] = field(default_factory=list, init=False)
    state: str = field(default="running", init=False)

    def __post_init__(self) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        token = uuid.uuid4().hex[:8]
        self.job_dir = self.root / f"xess_job_{self.kind}_{stamp}_{os.getpid()}_{token}"
        self.job_dir.mkdir(parents=False, exist_ok=False)
        self.manifest_path = self.job_dir / MANIFEST_NAME
        self._write_manifest()

    def _write_manifest(self) -> None:
        payload = {"version": 1, "kind": self.kind, "pid": os.getpid(),
                   "created_unix": time.time(), "state": self.state, "keep": self.keep,
                   "estimate": self.estimate, "targets": [str(path) for path in self.targets]}
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def path(self, name: str, *, track: bool = True) -> Path:
        result = (self.job_dir / name).resolve()
        if not _contained(result, self.job_dir):
            raise WorkdirError(f"invalid job path: {name}")
        if track and result not in self.targets:
            self.targets.append(result)
            self._write_manifest()
        return result

    def mkdir(self, name: str) -> Path:
        result = self.path(name)
        result.mkdir(parents=True, exist_ok=True)
        return result

    def child_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        temp_dir = self.mkdir("temp")
        cache_dir = self.mkdir("cache")
        mappings = {"TEMP": temp_dir, "TMP": temp_dir, "TMPDIR": temp_dir,
                    "TORCH_HOME": cache_dir / "torch", "HF_HOME": cache_dir / "huggingface",
                    "XDG_CACHE_HOME": cache_dir, "NUMBA_CACHE_DIR": cache_dir / "numba",
                    "OPENVINO_CACHE_DIR": cache_dir / "openvino", "PIP_CACHE_DIR": cache_dir / "pip"}
        for key, value in mappings.items():
            Path(value).mkdir(parents=True, exist_ok=True)
            environment[key] = str(value)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        return environment

    def driver_environment(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Use a bounded, persistent cache root for D3D driver helper services.

        Intel's shader-cache service can retain an open handle after the worker
        exits.  Keeping that tiny cache under the selected work disk avoids an
        undeletable per-job directory and, critically, never falls back to C:.
        """
        environment = dict(base or os.environ)
        cache_root = self.root / "runtime-cache"
        temp_dir = cache_root / "temp"
        shader_dir = cache_root / "shader"
        openvino_dir = cache_root / "openvino"
        torch_dir = cache_root / "torch"
        hf_dir = cache_root / "huggingface"
        for directory in (temp_dir, shader_dir, openvino_dir, torch_dir, hf_dir):
            directory.mkdir(parents=True, exist_ok=True)
        environment.update({"TEMP": str(temp_dir), "TMP": str(temp_dir),
                            "TMPDIR": str(temp_dir), "INTEL_CACHE_DIR": str(shader_dir),
                            "OPENVINO_CACHE_DIR": str(openvino_dir),
                            "XDG_CACHE_HOME": str(cache_root),
                            "TORCH_HOME": str(torch_dir), "HF_HOME": str(hf_dir)})
        return environment

    def mark_complete(self) -> None:
        self.state = "complete"
        self._write_manifest()

    def cleanup(self, *, label: str) -> bool:
        if self.keep:
            self.state = "kept"
            self._write_manifest()
            print(f"[{label}] intermediate files kept at: {self.job_dir}")
            return True
        errors: list[str] = []
        for target in sorted(self.targets, key=lambda path: len(path.parts), reverse=True):
            if not _contained(target, self.job_dir):
                errors.append(f"refused to clean a path outside the manifest: {target}")
                continue
            last_error = None
            for attempt in range(8):
                try:
                    if target.is_dir():
                        shutil.rmtree(target)
                    elif target.exists() or target.is_symlink():
                        target.unlink()
                    last_error = None
                    break
                except OSError as exc:
                    last_error = exc
                    # Intel's shader-cache service may briefly retain TEMP\Intel
                    # after the worker exits.  Bounded retries keep cleanup exact
                    # without ever expanding the deletion scope.
                    if attempt < 7:
                        time.sleep(0.15 * (attempt + 1))
            if last_error is not None:
                errors.append(f"{target}: {last_error}")
        self.state = "cleaned" if not errors else "cleanup_failed"
        try:
            self._write_manifest()
        except OSError as exc:
            errors.append(f"failed to write cleanup state: {exc}")
        if not errors:
            try:
                self.manifest_path.unlink(missing_ok=True)
                self.job_dir.rmdir()
            except OSError as exc:
                errors.append(f"failed to remove job directory: {exc}")
        if errors:
            print(f"[{label}] cleanup incomplete; approximately "
                  f"{format_gib(_path_size(self.job_dir))} remains:", file=sys.stderr)
            for error in errors:
                print(f"[{label}]   {error}", file=sys.stderr)
            return False
        return True


def create_workspace(
    *, kind: str, explicit_work_dir: str, output_dir: str, package_dir: str,
    input_path: str, allow_system_drive: bool, reserve_free_gb: float | None,
    max_temp_gb: float | None, estimated_bytes: int, keep: bool, label: str,
) -> JobWorkspace:
    if keep and not explicit_work_dir:
        raise WorkdirError("--keep requires an explicit --work-dir")
    root = select_work_root(explicit=explicit_work_dir, output_dir=output_dir,
                            package_dir=package_dir, input_path=input_path,
                            allow_system_drive=allow_system_drive)
    warn_stale_jobs(root, label=label)
    estimate = preflight(root, estimated_bytes, reserve_free_gb=reserve_free_gb,
                         max_temp_gb=max_temp_gb, label=label)
    return JobWorkspace(root=root, kind=kind, estimate=estimate, keep=keep)


def partial_output_path(final_path: str | os.PathLike[str]) -> Path:
    final = Path(final_path)
    return final.with_name(final.stem + ".partial" + final.suffix)


def finalize_output(partial: str | os.PathLike[str], final: str | os.PathLike[str]) -> None:
    partial_path = Path(partial)
    final_path = Path(final)
    if not partial_path.is_file() or partial_path.stat().st_size == 0:
        raise WorkdirError(f"partial output is missing or empty: {partial_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_path, final_path)
