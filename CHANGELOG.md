# Changelog

## 1.4.2 - 2026-09-08 — 三路 GPU 共享入口兼容 / Runtime R4.2

- 「Arc A 系列兼容模式」从 GPU Block 扩展至 GPU DIS 和 AMD 光流，三个视频节点均支持，默认仍关闭。问题位于公共解码纹理入口，不应按光流算法独占开关。
- GPU DIS / AMD 光流编译引擎接入同一 GPU RGBA 共享路径，光流内核、深度、Mask、后处理和颜色/HDR限制不变。GPU Block 与 R4.1 的二进制完全相同。
- CPU DIS 和 Intel 视频接口不经过这段 NV12→D3D12 共享入口，继续隐藏并拒绝该开关；原有功能不变。GPU DIS 与 AMD 光流仍保留实验项标注。
- R4/R4.1 更新 Git 后重启 ComfyUI 自动获取 R4.2 运行库，不覆写旧 Release 资产或用户环境。A770 其他算法的硬件反馈仍需积累，不把 B580 回归宣传为全 A 系列验证。

## 1.4.1 - 2026-09-08 — Arc A 系列兼容开关 / Runtime R4.1

- 三个 R4 节点新增可选的「Arc A 系列兼容模式」，仅 GPU Block 可用，默认关闭。旧 R4 工作流无需替换节点，原有字段顺序不变。
- 开启后使用 GPU NV12→RGBA 转换再共享，绕过直接共享 NV12 的兼容风险，用于处理 Arc A 系列反馈的绿/紫色竖条。保持光流、深度、Mask、锐化算法和编码器选择不变；不是抗竖纹滤镜。
- 配套运行库仅更新 GPU Block 编译引擎及新增两个编译着色器；其他运行库、模型和其他路线的二进制不变。通过固定版本和 SHA256 自动部署，无需覆盖诊断包或重装 ComfyUI 环境。
- A770 用户反馈兼容路径有效；维护者本机完成 B580 回归，不宣称已覆盖全部 A 系列/驱动。B580 原/兼容路径完整 311 帧 SR 和 621 帧 SR→FG 视频哈希一致，1000 帧短稳定性通过。
- 当前开关不适用于 GPU DIS / AMD 光流 / Intel 视频接口 / CPU DIS；这些后端不会显示该选项，API 错配明确拒绝。详见 docs/ARC_A_COMPATIBILITY.md。

## R4 公开历史清理 - 2026-09-07

- 经维护者授权，重写含 GPU DIS / GPU Block 源码的 R4 历史及相关标签；R3 及更早历史不变。
- 节点、视频管线、Runtime 二进制与哈希不变。旧 R4 克隆遇分叉时需备份后重新克隆，禁止把旧历史合并回公开主线。
- PR #9 只读引用和旧 SHA 缓存属于 GitHub 平台残留，需另行申请处理，不能宣称全网副本已撤回。

## R4 源码分发调整 - 2026-09-07

- GPU DIS / GPU Block 改为仅分发编译运行库，当前源码树移除对应算法及内嵌实现。
- 两个档位和配套 Runtime 不变；已安装 R4 用户无需重新下载运行库。
- 公共构建边界和历史可见性见 docs/GPU_DIS_BINARY_DISTRIBUTION.md；旧 Git 历史未清除。

## 1.4.0 - 2026-09-07 — ComfyUI R4

- 本次仅发布 ComfyUI 节点和共享离线引擎，独立离线/实时工具箱另行发布。
- 三个文件型 VIDEO 节点：超分、2× 插帧、超分后 2× 插帧。GPU Block
  默认，CPU DIS 兜底，Intel 视频接口，以及标注实验性的 GPU DIS/AMD 光流。
- H.264/HEVC QSV、x264/x265 软件编码、FFV1/MKV 无损编码。软件编码仅在
  最终输出边界回读，不宣称全链无 CPU 参与。
- 锐化、五帧融合、抗竖纹按算法/模式显示；独立插帧没有五帧和抗竖纹。
  锐化默认关闭，建议 SR/FG 组合只在末端开启一次。
- R3 用户正常 git pull 后重启即可自动获取匹配运行时；按版本隔离、
  完整哈希校验、并发安装锁、损坏重装与空间检查。无需修改 Comfy Python
  依赖，R3 engine 保留。网络不通可离线指定 Release ZIP。
- **不保留 R3 节点注册，旧工作流必须替换节点。** 此为破坏性界面升级，
  不是旧工作流无缝兼容。安装过并行试用插件的用户应移走试用插件。
- 本版硬件实测限 B580。A770/AMD/NVIDIA 不作本版全链兼容承诺；
  “AMD 光流”是算法来源而不是显卡兼容性标签。当前离线只发布 2× 插帧。

## 1.3.0 - 2026-08-28

- Added a symmetric six-slot shared-memory ring from `xess-vsr.exe` to the
  fused SR postprocess stage.  The normal launcher now enables both shared
  rings for 720p-or-larger output instead of leaving the measured r3 path
  reachable only from benchmark scripts.
- Parallelized sharpening and vertical-ringing protection with four ordered
  workers by default.  `--post-threads 1..16` is available for diagnostics;
  file/MFSR and explicit stream transports keep their compatible paths.
- Vectorized velocity bilinear upsampling with runtime-detected AVX2 and a
  scalar fallback, and cached DIS sampling grids while removing redundant
  frame-analysis copies.
- Fixed a shutdown hang where an ffmpeg guide-decoder failure could be
  dropped when the postprocess ready queue was full.
- Same-session, three-run interleaved B580 medians (worker chain, encoder
  excluded): 480p→720p/243 frames 15.488s → 7.683s (50.4% faster), and
  1080×1920→1440×2560/300 frames 57.702s → 46.117s (20.1% faster).  Both
  raw RGB output hashes are byte-identical to 1.2.0.  A770 remains untested.
- Added low-level `xess-vsr.exe --f16-rne` and `--f16-hw` experiments for
  correct IEEE round-to-nearest-even velocity conversion.  They remain
  opt-in because changing the historical one-ULP behavior breaks the strict
  byte-identity gate even though the measured visual difference is tiny.

## 1.2.0 - 2026-08-27

- Fixed file-mode packet generation to serialize optional depth and mask
  payloads by type instead of relying on Python object identity.
- Fixed runtime status/validation after installation by Windows PowerShell 5,
  whose UTF-8 state files include a byte-order mark.
- Made runtime SHA256 verification independent of `Get-FileHash`, which can
  disappear when portable ComfyUI/oneAPI environments replace `PSModulePath`.
- Fixed the Runtime r2 asset builder to allow-list production depth models;
  the retired SEA-RAFT checkpoint is no longer shipped in new archives.

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
