# Changelog

## Unreleased

- Added opt-in per-stage timing for the SR pipeline: `--stage-timing` on
  `run_xess.py` (or `XESS_STAGE_TIMING=1`) makes every Python component print
  one machine-readable `[timing] component=<name> {...}` line to stderr, and
  xess-vsr.exe reports CPU wall clock plus D3D12 GPU timestamp deltas for
  upload/execute/readback.  Disabled by default; the normal path performs no
  extra threads, copies, or logging.
- Retired SEA-RAFT from the mainline after B580 benchmarks showed DIS is both
  faster and at least as stable.  All presets and nodes now run native OpenCV
  DIS; old workflows that still select `sea-raft`/`sea-raft-single`
  automatically migrate to native Fast DIS with a one-time log notice.  The
  PyTorch-XPU subprocess probing, `safetensors` loading, sea-raft model
  discovery, and the archived experiment core are gone from the source tree
  (research code stays on the `experiment/sea-raft-xpu` branch).  The runtime
  manifest no longer requires the bundled SEA-RAFT checkpoint.

## 1.1.0 - 2026-08-21

- Split frequently updated source code from fixed Release runtime assets.
- Added SHA256-pinned runtime manifest and automatic first-install downloader.
- Added automatic pipeline synchronization after Git/launcher updates.
- Added GitHub/ComfyUI Registry metadata and Windows validation workflow.
- Replaced WGC as the default FG recovery path with direct native swap-chain readback.
- Documented the DXGI factory interception and native back-buffer recovery path.
- Fixed the combined SR -> FG entry point to always select the direct capture backend.
- Kept the legacy `window` capture mode for diagnostics.
