from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
import zipfile


ARCHIVE_ROOT = "xess-runtime-windows-x64"
STATIC_FILES = (
    "ffmpeg.exe",
    "xess-vsr.exe",
    "xess-fg.exe",
    "libxess.dll",
    "libxess_fg.dll",
    "libxell.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "INTEL_XESS_SDK_LICENSE.txt",
    "INTEL_XESS_THIRD_PARTY_PROGRAMS.txt",
    "THIRD_PARTY_NOTICES.md",
)
STATIC_DIRS = ("python", "models")
REQUIRED_FILES = (
    "ffmpeg.exe",
    "xess-vsr.exe",
    "xess-fg.exe",
    "libxess.dll",
    "libxess_fg.dll",
    "libxell.dll",
    "python/python.exe",
    "models/depth-anything-v2-small/depth_anything_v2_small.xml",
    "models/depth-anything-v2-small/depth_anything_v2_small.bin",
    "models/sea-raft/sea_raft_s_full.safetensors",
)
HASHED_FILES = REQUIRED_FILES


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excluded(relative: pathlib.Path) -> bool:
    lowered = tuple(part.casefold() for part in relative.parts)
    if "__pycache__" in lowered or relative.suffix.casefold() in {".pyc", ".pyo"}:
        return True
    return len(lowered) >= 3 and lowered[0] == "python" and lowered[1:3] == ("lib", "test")


def collect(source: pathlib.Path, sdk_source: pathlib.Path | None) -> list[tuple[pathlib.Path, pathlib.Path]]:
    files: list[tuple[pathlib.Path, pathlib.Path]] = []
    for name in STATIC_FILES:
        path = source / name
        if path.is_file():
            files.append((path, pathlib.Path(name)))
    for directory in STATIC_DIRS:
        root = source / directory
        if not root.is_dir():
            raise SystemExit(f"missing fixed runtime directory: {root}")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if not excluded(relative):
                files.append((path, relative))
    if sdk_source is not None:
        for directory in ("inc", "lib"):
            root = sdk_source / directory
            if not root.is_dir():
                raise SystemExit(f"missing XeSS SDK developer directory: {root}")
            for path in root.rglob("*"):
                if path.is_file():
                    files.append((path, pathlib.Path("sdk/official") / path.relative_to(sdk_source)))
        for name in ("LICENSE.txt", "README.md", "SECURITY.md", "third-party-programs.txt"):
            path = sdk_source / name
            if path.is_file():
                files.append((path, pathlib.Path("sdk/official") / name))
    missing = [name for name in REQUIRED_FILES if not (source / pathlib.Path(name)).is_file()]
    if missing:
        raise SystemExit("missing required runtime files: " + ", ".join(missing))
    return sorted(files, key=lambda item: item[1].as_posix().casefold())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the fixed XeSS Windows runtime Release asset")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sdk-source", default="")
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repository", default="gggz114514-oss/XeSS-Video-Enhancement-Suite")
    args = parser.parse_args()

    source = pathlib.Path(args.source).resolve()
    sdk_source = pathlib.Path(args.sdk_source).resolve() if args.sdk_source else None
    output = pathlib.Path(args.output_dir).resolve()
    manifest_path = pathlib.Path(args.manifest).resolve()
    output.mkdir(parents=True, exist_ok=True)
    asset_name = f"xess-runtime-windows-x64-{args.runtime_version}.zip"
    asset = output / asset_name
    partial = output / (asset_name + ".partial")
    files = collect(source, sdk_source)
    installed_size = sum(path.stat().st_size for path, _ in files)
    build_info = {
        "runtime_version": args.runtime_version,
        "built_unix": time.time(),
        "source": os.fspath(source),
        "sdk_source": os.fspath(sdk_source) if sdk_source is not None else None,
    }
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for index, (path, relative) in enumerate(files, 1):
            archive.write(path, (pathlib.Path(ARCHIVE_ROOT) / relative).as_posix())
            if index % 500 == 0:
                print(f"[runtime-asset] archived {index}/{len(files)} files", flush=True)
        archive.writestr(
            f"{ARCHIVE_ROOT}/RUNTIME_BUILD.json",
            json.dumps(build_info, ensure_ascii=False, indent=2) + "\n",
        )
    with zipfile.ZipFile(partial, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"ZIP CRC validation failed: {bad}")
    os.replace(partial, asset)
    asset_hash = sha256_file(asset)
    sidecar = asset.with_suffix(asset.suffix + ".sha256")
    sidecar.write_text(f"{asset_hash}  {asset.name}\n", encoding="ascii")
    file_hashes = {
        name.replace("\\", "/"): sha256_file(source / pathlib.Path(name))
        for name in HASHED_FILES
    }
    manifest = {
        "schema_version": 1,
        "runtime_version": args.runtime_version,
        "release_tag": args.tag,
        "asset_name": asset.name,
        "download_url": (
            f"https://github.com/{args.repository}/releases/download/{args.tag}/{asset.name}"
        ),
        "sha256": asset_hash,
        "archive_root": ARCHIVE_ROOT,
        "archive_size": asset.stat().st_size,
        "installed_size": installed_size,
        "required_files": [name.replace("\\", "/") for name in REQUIRED_FILES],
        "file_hashes": file_hashes,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[runtime-asset] asset: {asset}")
    print(f"[runtime-asset] installed: {installed_size / 1024**2:.2f} MiB")
    print(f"[runtime-asset] archive: {asset.stat().st_size / 1024**2:.2f} MiB")
    print(f"[runtime-asset] sha256: {asset_hash}")
    print(f"[runtime-asset] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
