from __future__ import annotations

import os
import sys

from runtime_manager import RuntimeManagerError, ensure_runtime


def main() -> int:
    if os.environ.get("COMFYUI_XESS_SKIP_RUNTIME_DOWNLOAD", "").strip() == "1":
        print("[ComfyUI-XeSS] fixed-runtime download skipped by environment")
        return 0
    try:
        engine = ensure_runtime()
    except RuntimeManagerError as exc:
        print(f"[ComfyUI-XeSS] runtime installation failed: {exc}", file=sys.stderr)
        return 1
    print(f"[ComfyUI-XeSS] runtime ready: {engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

