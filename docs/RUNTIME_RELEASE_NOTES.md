# Runtime 2026.08.28-r3

Fixed Windows/Intel Arc runtime for XeSS Video Enhancement Suite source release 1.3.x.

## Release tag

`runtime-2026.08.28-r3`

## Assets

- `xess-runtime-windows-x64-2026.08.28-r3.zip`
- `xess-runtime-windows-x64-2026.08.28-r3.zip.sha256`

SHA256:

`ff5ed90119adb51a00f215a39602896c4f8e0ca86de855987b2676fc7cb8db18`

The asset contains only fixed resources: ffmpeg, XeSS/XeFG/XeLL binaries, the portable Python/OpenVINO environment, depth models, and XeSS 2.1 developer headers/import libraries. Frequently updated node and pipeline sources stay in Git.

Runtime r3 contains the optimized `xess-vsr.exe` used by source 1.3.0.  In
addition to the existing three-slot D3D12 pipeline, it can write completed RGB
frames directly to a second shared-memory ring, vectorizes velocity upsampling
with runtime-detected AVX2, and retains scalar and pipe fallbacks.  The normal
source launcher enables the output ring together with four ordered postprocess
workers for 720p-or-larger output.

On B580, same-session interleaved medians improved by 50.4% for a 243-frame
480p→720p worker chain and 20.1% for a 300-frame
1080×1920→1440×2560 chain.  Raw RGB output hashes remained byte-identical to
Runtime r2.  A770 was not available for this release and remains untested.

SEA-RAFT has been retired from the mainline.  The r3 archive does not bundle its checkpoint, keeping the download at 271.72 MiB. Existing installations may keep an old unused file until `.runtime` is reinstalled or removed.

FG uses DXGI factory interception and direct native swap-chain back-buffer readback by default. It recovers the actual XeFG-presented frame without desktop capture, so it is not affected by RTSS desktop OSD, WGC service availability, high DPI window coordinates, window minimization or window occlusion. The legacy WGC path remains available only as a diagnostic fallback.
