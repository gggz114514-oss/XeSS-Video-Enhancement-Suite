"""Validate an Intel XeSS SDK directory before compiling native workers.

The source-only repository deliberately does not carry Intel's binary SDK.  This
check makes the external SDK root explicit and prevents compiling against one
set of headers/import libraries while copying another set of DLLs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "tools" / "xess_sdk_3_0_2.json"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Intel XeSS SDK 3.0.2")
    parser.add_argument("sdk_root", type=pathlib.Path)
    args = parser.parse_args()
    try:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"[sdk] cannot read {SPEC_PATH}: {exc}", file=sys.stderr)
        return 2

    root = args.sdk_root.expanduser().resolve()
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"SDK root does not exist: {root}")
    for relative, expected in spec["required_files"].items():
        path = root / pathlib.PurePosixPath(relative)
        if not path.is_file():
            errors.append(f"missing SDK file: {relative}")
            continue
        actual = sha256(path)
        if actual.casefold() != str(expected).casefold():
            errors.append(f"SHA256 mismatch for {relative}: expected {expected}, got {actual}")

    if errors:
        for error in errors:
            print(f"[sdk] ERROR: {error}", file=sys.stderr)
        return 1
    print(f"[sdk] validated Intel XeSS SDK {spec['sdk_package']} ({root})")
    print(f"[sdk] headers/import libs/DLLs are one signed package: {spec['release_asset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
