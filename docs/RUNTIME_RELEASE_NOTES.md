# Runtime 2026.08.27-r2

Fixed Windows/Intel Arc runtime for XeSS Video Enhancement Suite source release 1.2.x.

## Release tag

`runtime-2026.08.27-r2`

## Assets

- `xess-runtime-windows-x64-2026.08.27-r2.zip`
- `xess-runtime-windows-x64-2026.08.27-r2.zip.sha256`

SHA256:

`54194b102b7faaa9d2bf5208569e5de6bdb36b43d42b2b8d946e3997d5df071e`

The asset contains only fixed resources: ffmpeg, XeSS/XeFG/XeLL binaries, the portable Python/OpenVINO environment, depth models, and XeSS 2.1 developer headers/import libraries. Frequently updated node and pipeline sources stay in Git.

Runtime r2 contains the pipelined `xess-vsr.exe` used by source 1.2.0.  SR keeps multiple D3D12 frames in flight, uses reduced-copy shared-memory transport and AVX2 RGB conversion where supported, while retaining a scalar fallback and byte-identical output.

SEA-RAFT has been retired from the mainline.  The r2 archive no longer bundles its checkpoint, reducing the download from 303.32 MiB to 271.72 MiB. Existing installations may keep the old unused file until `.runtime` is reinstalled or removed.

FG uses DXGI factory interception and direct native swap-chain back-buffer readback by default. It recovers the actual XeFG-presented frame without desktop capture, so it is not affected by RTSS desktop OSD, WGC service availability, high DPI window coordinates, window minimization or window occlusion. The legacy WGC path remains available only as a diagnostic fallback.
