# Changelog

## Unreleased

- Retired SEA-RAFT from the mainline after B580 benchmarks showed DIS is both
  faster and at least as stable.  All presets and nodes now run native OpenCV
  DIS; old workflows that still select `sea-raft`/`sea-raft-single`
  automatically migrate to native Fast DIS with a one-time log notice.  The
  PyTorch-XPU subprocess probing, `safetensors` loading, sea-raft model
  discovery, and the archived experiment core are gone from the source tree
  (research code stays on the `experiment/sea-raft-xpu` branch).  The runtime
  manifest no longer requires the bundled SEA-RAFT checkpoint.
- Fixed Balanced/Quality mode reporting zero XPU devices in child processes on
  some Arc A-series installations.  The node now probes in the same import
  order as SEA-RAFT, performs a real XPU allocation, preserves the launcher's
  working environment, and retries with Level Zero selectors when required.

## 1.1.0 - 2026-08-21

- Split frequently updated source code from fixed Release runtime assets.
- Added SHA256-pinned runtime manifest and automatic first-install downloader.
- Added automatic pipeline synchronization after Git/launcher updates.
- Added GitHub/ComfyUI Registry metadata and Windows validation workflow.
- Replaced WGC as the default FG recovery path with direct native swap-chain readback.
- Documented the DXGI factory interception and native back-buffer recovery path.
- Fixed the combined SR -> FG entry point to always select the direct capture backend.
- Kept the legacy `window` capture mode for diagnostics.
