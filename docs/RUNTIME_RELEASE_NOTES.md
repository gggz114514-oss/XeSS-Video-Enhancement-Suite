# ComfyUI R4 / Source 1.4.0 / Runtime 2026.09.07-r4

仅发布 ComfyUI 视频节点及其共享离线引擎。独立工具箱和实时工具箱不在此包中。

## 更新

- 三个新节点：超分、2× 插帧、超分后 2× 插帧。
- GPU Block（默认）、CPU DIS、Intel 视频接口、GPU DIS/AMD 光流（实验）。
- QSV H.264/HEVC、x264/x265、FFV1 无损，按模式联动的锐化/五帧/抗竖纹开关。
- git pull 后重启自动安装匹配运行时，不安装/替换 ComfyUI 的 Python 包。
- **旧 R3 节点不再注册，旧工作流需要手动替换节点。** 原有输出和 R3 engine 保留。

## 资产

- `xess-comfy-runtime-windows-x64-2026.09.07-r4.zip`
- 同名 `.zip.sha256`

ZIP SHA256：`f9012e3e2eaf5d9caa83dc6957e4bae147bd7aa06804bba7edae83df90165bda`。
ZIP 369,083,229 字节，解包文件合计 834,548,629 字节。自动安装额外预留 512 MiB，
不会为了腾空间删除旧引擎或用户视频。源码、节点、工作流在 Git，二进制/模型在 ZIP。

只在 Windows / Intel Arc B580 上验收；A770 及其他厂商不作本版完整支持承诺。
AMD 光流是算法名称。离线 2×，不包含实时 3×/4×。SDR 8bit/CFR 是本版 GPU 输入合同。

详细安装、升级、参数限制见仓库 README；发布验收见
`docs/reports/COMFY_R4_RELEASE_ACCEPTANCE.md`。实验路线不保证优于 CPU DIS。

---

# 历史：Runtime 2026.08.28-r3

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
