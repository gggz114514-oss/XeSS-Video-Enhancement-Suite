# Maintainer release guide

## Code-only release

Use this for changes under `xess_nodes.py`, `pipeline/`, `src/`, workflows or documentation.

1. Update the semantic version in `pyproject.toml` and `CHANGELOG.md` when appropriate.
2. Run Python compilation, the ComfyUI self-test and a short standalone smoke test.
3. Commit and push the Git changes.
4. Do not rebuild or upload the fixed runtime asset.

On the next ComfyUI execution, `runtime_manager.py` copies the updated `pipeline/` files into `.runtime/engine`. The user's 303 MiB asset is not downloaded again.

## Fixed-runtime release

Use this only when an exe, DLL, ffmpeg, model or portable Python environment changes.

```bat
.runtime\engine\python\python.exe tools\build_runtime_asset.py ^
  --source "C:\path\to\shipping-runtime" ^
  --sdk-source "C:\path\to\xess-sdk" ^
  --output-dir "C:\path\to\release-assets" ^
  --manifest runtime_manifest.json ^
  --runtime-version 2026.08.21-r2 ^
  --tag runtime-2026.08.21-r2
```

The builder creates the ZIP, its `.sha256` sidecar and rewrites `runtime_manifest.json` with the exact URL, sizes and compatibility hashes.

Before publishing:

1. Test `install_runtime.ps1 -AssetPath <local-zip>` in a new empty directory.
2. Run SR and FG through the Git-root `.bat` launchers.
3. Run the ComfyUI `self_test.py` against the newly installed runtime.
4. Create the exact tag named by `release_tag`.
5. Upload the exact `asset_name` and its `.sha256` sidecar to that Release.
6. Only after the asset is reachable, commit/push the new `runtime_manifest.json`.

This order prevents a source update from pointing users at a Release asset that does not exist yet.

## Files that must stay out of Git

- `.runtime/`
- `python/`, models, ffmpeg, XeSS/XeFG/XeLL DLLs and built executables
- videos, raw frames, optical-flow/depth intermediates and caches
- local `xess_config.json`

CI rejects files larger than 10 MiB as an additional guard.
