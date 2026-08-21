# Changelog

## Unreleased

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
