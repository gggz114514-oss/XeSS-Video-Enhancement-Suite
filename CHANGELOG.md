# Changelog

## Unreleased

- Pipelined the SR critical path: xess-vsr.exe now runs a three-slot
  upload/execute/readback pipeline (multiple frames in flight, CPU/GPU stage
  overlap) with per-slot fences (frame order and GPU results unchanged), and
  the former adaptive-sharpen + edge-ringing-guard process pair is fused into
  a single-process `sr_postprocess.py` stage with a prefetch thread for the
  guard's guide analysis.  Same-round benchmark on 300 frames 1080p→1440p
  fast preset: 78.814s → 63.977s (≈18.8%) on B580, byte-identical output.
  The older 83.0s → 64.0s (≈23%) figures come from an earlier session and are
  not a same-round A/B (A770 未实测).
- xess-vsr.exe now drains in-flight GPU work before tearing down: the last
  submitted fence is awaited with a bounded timeout on every exit path
  (early input end, downstream close, write failure), with a `[drain]`
  summary line; the unreachable writeDoneSem was removed.
- SR postprocess guide-producer thread exits via a stop event and is joined
  on shutdown; ffmpeg decoder stderr tails are reported on failure.
- Direct `prepare_sr.py --engine sea-raft --bidirectional` calls are forced
  to one-way Fast DIS with a one-time notice (expert `--engine dis
  --bidirectional` is unchanged).
- CI runs the dependency-light unit tests (numpy + opencv-python-headless;
  a minimal torch stub replaces a real torch install for import checks).
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
