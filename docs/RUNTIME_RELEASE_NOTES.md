# Runtime 2026.08.21-r1

Fixed Windows/Intel Arc runtime for XeSS Video Enhancement Suite source release 1.1.x.

## Release tag

`runtime-2026.08.21-r1`

## Assets

- `xess-runtime-windows-x64-2026.08.21-r1.zip`
- `xess-runtime-windows-x64-2026.08.21-r1.zip.sha256`

SHA256:

`0f69db8f652d4b63d849bd8f27fa6cc8950cd7ef98bfea94461230230e85f78b`

The asset contains only fixed resources: ffmpeg, XeSS/XeFG/XeLL binaries, the portable Python/OpenVINO environment, depth models, and XeSS 2.1 developer headers/import libraries. Frequently updated node and pipeline sources stay in Git.

Note: since SEA-RAFT was retired from the mainline, `runtime_manifest.json` no longer lists the bundled `models/sea-raft` checkpoint as required or hashed; existing installations may keep the unused file on disk.

FG uses DXGI factory interception and direct native swap-chain back-buffer readback by default. It recovers the actual XeFG-presented frame without desktop capture, so it is not affected by RTSS desktop OSD, WGC service availability, high DPI window coordinates, window minimization or window occlusion. The legacy WGC path remains available only as a diagnostic fallback.
