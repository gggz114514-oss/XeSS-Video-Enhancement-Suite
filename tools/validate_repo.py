from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import py_compile
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".runtime", "build", "dist", "work", "__pycache__"}
FORBIDDEN_SUFFIXES = {".dll", ".exe", ".safetensors", ".raw", ".mp4", ".avi"}
LOCAL_WORKSPACE_PATTERN = re.compile(r"(?i)[a-z]:\\[^\r\n]*(?:xess-tools|comfyui-aki)")
REQUIRED = (
    "__init__.py", "xess_nodes.py", "runtime_manager.py", "runtime_manifest.json",
    "install.py", "install_runtime.ps1", "requirements.txt", "pyproject.toml",
    "pipeline/run_xess.py", "pipeline/run_fg.py", "pipeline/prepare_sr.py",
    "pipeline/prepare_fg.py", "src/xess_vsr.cpp", "src/xess_fg.cpp",
    "src/shm_ring_win.h", "workflows/xess超分帧生成.json",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the source-only Git repository")
    parser.add_argument("--asset", default="")
    args = parser.parse_args()
    errors: list[str] = []
    for name in REQUIRED:
        if not (ROOT / pathlib.Path(name)).is_file():
            errors.append(f"missing repository file: {name}")
    for path in source_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file larger than 10 MiB belongs in Releases: {relative}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            errors.append(f"fixed/binary asset belongs in Releases: {relative}")
        if path.suffix.casefold() in {".py", ".pyw"}:
            try:
                py_compile.compile(path, doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(str(exc))
        if path.suffix.casefold() in {".py", ".ps1", ".bat", ".md", ".json", ".toml", ".yml"}:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeError:
                continue
            if LOCAL_WORKSPACE_PATTERN.search(text):
                errors.append(f"local workspace path leaked into: {relative}")
    try:
        workflow = json.loads((ROOT / "workflows/xess超分帧生成.json").read_text(encoding="utf-8-sig"))
        types = {node.get("type") for node in workflow.get("nodes", [])}
        expected = {"XeSSVideoSuperResolution", "XeSSVideoFrameGeneration"}
        if not expected.issubset(types):
            errors.append(f"workflow is missing nodes: {sorted(expected - types)}")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid workflow JSON: {exc}")
    manifest = json.loads((ROOT / "runtime_manifest.json").read_text(encoding="utf-8"))
    if args.asset:
        asset = pathlib.Path(args.asset).resolve()
        if asset.name != manifest["asset_name"]:
            errors.append(f"asset name mismatch: {asset.name} != {manifest['asset_name']}")
        elif sha256(asset).casefold() != manifest["sha256"].casefold():
            errors.append("asset SHA256 does not match runtime_manifest.json")
    if errors:
        for error in errors:
            print(f"[validate] ERROR: {error}", file=sys.stderr)
        return 1
    print("[validate] repository structure, sources, workflow and runtime manifest are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
