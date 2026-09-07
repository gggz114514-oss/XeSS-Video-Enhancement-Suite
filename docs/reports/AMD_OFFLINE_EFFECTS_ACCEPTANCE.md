# AMD 光流增强开关验收

日期：2026-09-07。起点：`aacc3c7`。范围：本地离线工具箱与共用控制层的 R4 ComfyUI 节点；未发布 GitHub。

## 当前能力

| AMD 光流模式 | 抗竖纹 | 五帧融合 | 锐化 |
| --- | --- | --- | --- |
| 超分 | 可独立开启 | 可独立开启 | 末尾一次 |
| 超分→2×补帧 | 在 SR 后、FG 前处理 | 在 SR 后、FG 前处理 | 成片末尾一次 |
| 独立 2×补帧 | 不显示 SR 专用功能 | 不显示 SR 专用功能 | 末尾一次 |

三项默认关闭。Intel 视频接口仍不开放这三项。GPU Block、GPU DIS、CPU DIS 保持原有能力，选择变化仍使用既有局部更新，不重新创建整页。

重点是抗竖纹：复用上一轮已经对齐 CPU 的 GPU 引导图方案，**不需要同时开启五帧或锐化**。此轮没有发明新的抗竖纹公式，也没有更换 AMD 光流算法。增强会偏向原片引导图，可能影响细节；是否消除用户关注的所有竖纹仍由全片主观观看确认。

## 接线与实现

- `offline_toolbox.py` 能力声明、计划时 shader 缺失检查、原生命令都包含 AMD；不静默丢弃参数。
- AMD 原生入口复用 `vpl_gpu_full_fg.cpp` 的 `NativeSrEffects`，放行 SR/SR-FG；独立 FG 仍拒绝 SR 专用效果。
- AMD 的 current→previous 源像素光流经已有公共适配器送入历史融合，复用计算结果。官方预热语义保留；12 帧测试覆盖预热之后的非初始阶段。
- 五帧为当前＋前四帧因果窗口，开启时至少五槽，不是 CPU 前后各两帧版本。静态画面适用，动态可能产生副作用。
- 抗竖纹采用 cubic 原片引导、Sobel-x、sigma 2.5/21 taps 高斯、强度 0.90 混合。每槽 scratch 和描述符隔离，不额外回读到 CPU。效果合成后一次 GPU 内部 copy，不称零拷贝。
- 锐化原本已经接通，此次验证其在 SR、FG、SR-FG 中均每输出帧执行一次，没有 SR/FG 双重锐化。
- ComfyUI 更新能力合同和中文 tooltip，不修改 ComfyUI Python、旧节点或启动配置。

## 验证结果

运行产物统一位于 `work/r4-offline-delivery-20260907/amd-effects/`。

1. `reference/result.json`：AMD old/off/guard/five/both 五种配置，各 12 帧。新 off 与部署前旧 EXE 的像素逐字节一致；抗竖纹、五帧、二者组合对独立 CPU/NumPy 参考最大误差均为 **1 色阶**。抗竖纹最差帧 MAE 0.031225 色阶。开关确实改变图像，五帧历史样本数 38；AMD 报告 11 对源帧分析、官方预热 5 对，未偷偷切换其他光流。
2. `guard-only-full/results.json`、`off-full/results.json`：**4/4**。使用原始完整 480p 人脸文件，不先转码测试素材；AMD SR 与 SR-FG，单独抗竖纹开/关，243→243/485 帧、1296×720、音轨与 PTS 通过。原片包含用户关注的 8–9 秒。单次墙钟分别 6.635/10.259 秒（开）、6.531/10.268 秒（关），不是重复交错性能基准，不据此宣传无开销。
3. `combined-4k/results.json`：**4/4**。1080p→3840×2160，12→12/23 帧，三开关全开，SR/SR-FG × H.264 QSV/FFV1。实际效果帧数、38 次历史采样和锐化 dispatch 数与请求一致。FFV1 编码前后 RGB 无损门通过；QSV 路线完整像素 CPU 上传/回读为 0；软件编码仅在末端有界回读。
4. `installed-fg/results.json`：直接调用已更新本地工具箱，AMD 独立 FG＋锐化 12→23 帧通过；原生报告恰为 23 次锐化 dispatch。
5. 定向 Python 单测 **54/54**，包含 AMD 能力/原生命令/五槽/Comfy 合同，且 Comfy 测试能力表与生产声明自动对账。实际 app.js 行为测试通过，AMD SR/SR-FG 三项显示、独立 FG 仅锐化，选择不重建页面；UI 校验器 20/20，compileall、validate_repo、diff check 通过。未冒充全仓测试或实际 Comfy GPU 工作流复跑。

完整观看文件：

- `guard-only-full/amd-of-sr-fg-h264_qsv/output.mp4`：只开抗竖纹，超分＋补帧。
- `off-full/amd-of-sr-fg-h264_qsv/output.mp4`：全部关闭，同源对照。
- 同目录的 `amd-of-sr-h264_qsv/output.mp4` 是各自独立超分全片。

## 构建、更新与回退

- 使用已有 `tools/build_amd_of_native.cmd <SDK_CHECKOUT> <OUT>`，输出 `amd-effects/build`；没有安装编译器/驱动或修改全局环境。构建日志 `build.log` 与 `build/build-*.log`。
- AMD EXE SHA256：`20e17b62e746d36ed88d4608b709f21cab418731df5291583b28606901883d8b`。
- 共用 shader SHA256：`b1152483b5b38e1ba3231ea1511c4cdb59e9df91168ef1b0a245775d10028650`，与上一轮相同。
- `tools/update_offline_gpu_effects.ps1 -NativeRoutes amd-of` 仅更新 AMD EXE 与共用控制文件，逐项备份并刷新本地 runtime 清单。其他路线 EXE、SDK DLL、Python 未修改。更新记录与旧文件在 `amd-effects/before`；本机 runtime 是目录链接，不可递归删除共享目标。
- 已部署控制层能力查询返回 AMD 三项支持、无缺失文件。工具箱右上角刷新重新读取能力；已运行的 ComfyUI 需重启才能加载新的 Python 模块。
- 清理本轮 720 个可重现诊断 `.bin`，共 1,457,856,960 字节；保留原始素材、完整视频、JSON、日志、对照图、源码和旧版备份。没有删除其他实验资源。

本机 B580 实测，不等于已在 AMD 品牌显卡上测试；AMD 是光流算法名称。没有新增长时间稳定性、HDR 或其他显卡的支持承诺。
