# XeSS 视频增强节点 / XeSS Video Enhancement for ComfyUI

Windows 上的视频超分与 2× 插帧。R4 提供三个中文节点：**视频超分、视频插帧、超分后插帧**。

本仓库本次只发布 ComfyUI 节点与配套引擎。独立离线工具箱、实时捕获工具箱将分项目发布，不包含 OBS、桌面捕获或 HTML 前端。

[R4.2 下载与更新说明](https://github.com/gggz114514-oss/XeSS-Video-Enhancement-Suite/releases/tag/runtime-2026.09.08-r4.2)

**R4.2 兼容修复：** Arc A 系列使用 GPU Block、GPU DIS 或 AMD 光流时若出现绿/紫色竖条，可开启节点的「Arc A 系列兼容模式」。默认关闭，支持超分、2× 插帧及组合节点；它与「抗竖纹」画质滤镜不同。CPU DIS 和 Intel 视频接口不走这段共享路径，无需此开关。已安装 R4/R4.1 的用户更新节点并重启 ComfyUI、刷新页面即可自动获取配套运行库。[详细说明](docs/ARC_A_COMPATIBILITY.md)

GPU DIS 与 GPU Block 仅提供运行库中的编译版，当前源码目录不再包含其 C++ / HLSL 实现；节点功能与安装方式不变。构建边界见 [GPU 算法发布说明](docs/GPU_DIS_BINARY_DISTRIBUTION.md)。

## 安装

关闭 ComfyUI，在 `ComfyUI/custom_nodes` 中打开终端：

```powershell
git clone https://github.com/gggz114514-oss/XeSS-Video-Enhancement-Suite.git
```

然后正常启动 ComfyUI。节点自动下载**与当前源码匹配**的运行时，校验成功后可用；首次需要联网及足够磁盘空间。无需手动装 PyTorch、OpenVINO、oneAPI 或改 ComfyUI 的 Python 包。运行时位于本节点目录 `.runtime/versions/`。

无法访问 GitHub 时，从本版本 Release 下载 `xess-comfy-runtime-*.zip`，在节点目录运行：

```powershell
.\install_runtime.bat -AssetPath "D:\Downloads\xess-comfy-runtime-windows-x64-2026.09.08-r4.2.zip"
```

盘符只是示例，可以安装在任意可写目录。脚本未找到 Python 时，用 `-Python "你的ComfyUI Python完整路径"` 指定。**不要把运行时压缩包覆盖到 ComfyUI 根目录。**

## 从 R3 升级

在已有 `custom_nodes/XeSS-Video-Enhancement-Suite` 目录执行：

```powershell
git pull --ff-only
```

重新启动 ComfyUI、刷新浏览器。程序自动检查并下载 R4 配套运行时；纯源码更新不会重复下载相同运行时。网络失败会显示中文错误，修复网络后重新执行节点即可重试。

**R4 不保留旧节点。旧工作流里的 R3 节点需要替换为新节点**，不能承诺旧工作流原样运行。原有视频、模型、ComfyUI 配置与 R3 `.runtime/engine` 不会删除。以前安装过本地试用版 `XeSS-R4-Offline-Local` 的用户，关闭 ComfyUI 后先将该试用插件移出 `custom_nodes`，避免同名节点重复注册。

如果 `git pull` 提示有本地修改，先备份或提交修改；不要强制覆盖。回退步骤见 [升级与回退](docs/COMFY_R4_UPGRADE.md)。

**2026-09-07 R4 历史清理提醒：** 如果已经安装过当天清理前的 R4，`git pull --ff-only` 可能提示历史分叉。请关闭 ComfyUI，把旧节点目录备份到 `custom_nodes` 外，再重新克隆；可将备份中的 `.runtime` 拷回新节点目录，由程序重新校验。不要合并或推送旧 R4 历史。R3 及更早正常更新不受影响。

## 使用

加载 [最小组合工作流](workflows/r4_offline_quickstart.json)，或者搜索菜单 `XeSS R4 离线视频`：

| 节点 | 输入 → 输出 |
| --- | --- |
| R4 离线视频超分 | VIDEO → 放大后 VIDEO |
| R4 离线视频 2× 插帧 | VIDEO → 同分辨率、2× 帧率 VIDEO |
| R4 离线视频超分→2×插帧 | VIDEO → 放大并插帧 VIDEO |

使用 ComfyUI 原生加载视频节点输入**文件型 VIDEO**，输出接保存视频节点。保存时建议格式/编码器保持 auto，避免再次压缩。三个节点也返回成片路径。纯 IMAGE 批次和第三方特殊 VIDEO 类型不是本版输入合同。

超分倍率：1.33×、1.5×、2×、自定义（1–4）；尺寸按后端要求对齐到 16 的倍数，实际尺寸以成片为准。独立插帧不改变分辨率。

### 算法

| 档位 | 定位与边界 |
| --- | --- |
| **GPU Block（默认）** | 自研 GPU 光流，快速路线；不是保证所有素材画质最好的档位 |
| CPU DIS | 传统光流，稳定兜底；CPU 计算光流，XeSS/XeFG 仍使用 GPU |
| Intel 视频接口 | Intel 原生 AI 视频处理，不是 XeSS；速度优先，受硬件、尺寸和颜色格式限制 |
| GPU DIS（实验） | GPU 迁移路线，允许使用，但不保证与 CPU DIS 输出完全相同或更快 |
| AMD 光流（实验） | 使用 FidelityFX Optical Flow 算法；**不代表整条视频链已支持 AMD 显卡** |

显式选择的后端失败时会报错，不会悄悄切换算法。GPU Block/DIS/AMD 路线目前必须使用 AI 深度；CPU DIS 可选固定深度。界面随算法切换隐藏不支持的选项。

### 增强开关

- **末尾锐化**：默认关闭；同时超分和插帧时建议只在最后锐化一次。
- **五帧融合**：适合静态画面，动态可能出现反效果。GPU 路线使用当前帧和前四帧；CPU DIS 仅独立超分支持前后各两帧融合。
- **抗竖纹**：减轻 XeSS 在脸部等区域的竖纹，可能减少局部细节并增加耗时；默认关闭。支持 CPU DIS/GPU Block/GPU DIS/AMD 光流的超分与组合节点。

独立插帧只有锐化开关，没有五帧融合或抗竖纹。Intel 视频接口不提供这三个增强开关。

### 编码器

支持自动（H.264 QSV）、H.264 QSV、HEVC QSV、libx264、libx265、FFV1。

QSV 使用 Intel 硬件编码；x264/x265/FFV1 为软件编码，会回读最终画面并增加 CPU 耗时。**FFV1 使用 MKV，编码本身无损，但不能恢复输入视频已丢失的细节。**音频在处理链中保留；封装不支持原音频格式时可能转换。

## 环境与限制

- 本轮目标：Windows x64、支持 DirectX 12 的 Intel Arc 与可用的视频驱动；**B580 真机验收，A770 未真机覆盖本次完整矩阵**。不宣称 NVIDIA/AMD 已支持全部路线。
- ComfyUI 需要有原生 `VIDEO` 和 `comfy_api.latest.InputImpl.VideoFromFile`。开发使用 `ComfyUI-aki-v3-IntelArc` 20260722 整合包，当前本地内核已更新为 **ComfyUI 0.33.1 / Python 3.13.11**；整合包初始 0.28.0 不等于本次实测版本。不要求用户更换整合包。
- 配套 OpenCV/OpenVINO/Python 使用独立运行时，与 Comfy 的 torch/XPU 环境隔离。无需 SEA-RAFT。
- 当前 GPU 入口：8-bit、有限范围 SDR、恒定帧率 H.264/HEVC 4:2:0、偶数宽高、方形像素；不支持 HDR/P010、VFR 或只靠旋转元数据的输入，不会静默裁切或降质。
- XeFG 2× 输出 `2N−1` 帧；Intel 插帧输出 `2N` 帧。离线版不提供 3×/4×。
- 请关闭 RTSS 等可能注入视频工作进程的叠加层；实时捕获功能不在本仓库本次发布范围。

## 故障与磁盘

首次运行时下载日志显示在 ComfyUI 控制台；失败不会覆盖已安装版本。任务日志保留在 ComfyUI 输出目录 `.xess-work/`，报错会包含子进程根因，不只显示 EOF。

- 缺 DLL：重新执行运行时安装脚本，加 `-Force` 完整校验；不要从第三方网站单独下载 DLL。
- 提示不支持格式/尺寸：按提示调整输入，或明确选择 CPU DIS；不会自动使用低质量替代。
- 内存/磁盘不足：缩短测试视频、降低倍率或释放空间。FFV1 成片可能很大，注意**最终输出盘**也需要空间。
- 安装不修改 ComfyUI 配置和依赖；取消处理会结束该任务的子进程，不结束其他用户程序。

提交问题请附：算法、模式、倍率、编码器、GPU/驱动版本、输入视频信息和报错前后的日志。不要只截最后一行。

## 开发与许可

源码与固定资产分开：Git 保存节点、控制层、C++/HLSL；Release 保存经哈希校验的 EXE/DLL、模型和独立 Python。`src/realtime` 是历史目录名，其中本版编译的是共用离线 GPU 工作器，不是实时工具箱。

第三方组件及源码入口见 [第三方说明](licenses/THIRD_PARTY_NOTICES.md)。各 SDK、模型与工具遵守各自许可证；本项目不修改第三方条款，也不作全链跨显卡支持保证。
