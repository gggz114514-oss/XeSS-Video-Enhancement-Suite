# XeSS 视频工具箱：SR 1.2 + FG 1.2 + ComfyUI 节点

这是一个面向 Windows 和 Intel Arc 的视频处理发布包，包含：

- 独立便携版 XeSS 视频超分：SR 1.2；
- 独立便携版 XeSS 2× 帧生成：FG 1.2；
- 一次执行“先超分、后插帧”的联合管线；
- 中文 ComfyUI 原生 `VIDEO → VIDEO` 节点；
- 可直接导入的 `xess超分帧生成.json` 工作流；
- 环境检测、节点安装和配置脚本；
- C++ 核心源码、Python 管线源码与构建所需的 XeSS/XeLL 头文件和导入库。

> 这是社区工具，不是 Intel 官方产品。当前实机验证平台为 Windows 11 + Intel Arc B580。其他 Arc 型号预计可用，但请自行测试。XeSS FG 使用桌面呈现与抓取路径，运行要求比普通离线编码更严格。

## 下载与解压

从 GitHub Releases 下载 `XeSS-Video-Suite-2026.08.zip`，完整解压后再运行。不要直接在压缩软件里双击脚本，也不要单独移动 `xess-portable-pipeline` 文件夹。

推荐解压到剩余空间充足的位置，例如：

```text
D:\XeSS-Video-Suite-2026.08
```

没有 D/E 盘也能用。只有 C 盘时，工具会使用流式处理，并在 `%LOCALAPPDATA%\XeSS-Video-Suite\work` 建立受保护的工作目录；系统盘默认保留至少 25 GiB 空间，达不到条件时会在创建大文件前停止。

首次使用建议双击：

```text
check_environment.bat
```

它只检测，不改文件。会检查便携运行时、显卡、RTSS、可用空间、ComfyUI Python，以及极致画质模式所需的 Intel PyTorch XPU/OpenVINO。

## 一、独立便携版

不需要系统 Python、ffmpeg 或 ComfyUI。进入 `xess-portable-pipeline` 文件夹，在地址栏输入 `cmd` 回车，然后使用下面的命令。

### 1. 视频超分

480p 放大 1.5 倍到 720p，快速档：

```bat
run_xess.bat "D:\video\input.mp4" 1.5 --preset fast
```

极致画质档：

```bat
run_xess.bat "D:\video\input.mp4" 1.5 --preset quality --five-frame-fusion --fusion-strength 0.35
```

输出默认写到输入视频旁边，文件名包含 `xess_sr12`、档位、倍率和分辨率。SR 1.2 默认开启强度 0.75 的竖向边缘振铃保护；它用于抑制脸部鼻梁、下颌、硬阴影旁出现的细竖线。要关闭可加 `--edge-guard-strength 0`。

### 2. 2× 帧生成

24 fps 输入会生成 48 fps 输出：

```bat
run_fg.bat "D:\video\input.mp4" --preset fast
```

极致画质档：

```bat
run_fg.bat "D:\video\input.mp4" --preset quality
```

FG 对 N 帧输入输出 `2N-1` 帧，顺序为 `f0, G1, f1, G2...`，并复制原音轨。

### 3. 先超分、后插帧

```bat
run_pipeline.bat "D:\video\input.mp4" --scale 1.5 --sr-preset fast --fg-preset fast
```

联合管线不会让 FG 复用 SR 的运动数据；SR 与 FG 分别计算适合自身语义的运动/深度信息，但共用流式传输、空间保护、AI 深度和质量档位体系。

### 档位差异

| 档位 | 光流/深度 | 特点 | 环境要求 |
|---|---|---|---|
| `fast` | DIS + AI 深度 | 速度优势最大，适合日常批量处理 | 便携包即可 |
| `balanced` | 单向 SEA-RAFT + AI 深度 | 遮挡边缘更稳，明显更慢 | Intel XPU PyTorch + OpenVINO |
| `quality` | 双向 SEA-RAFT + AI 深度 | 最慢，运动和遮挡信息最多 | Intel XPU PyTorch + OpenVINO |

安装 ComfyUI 节点时，脚本会自动检测 ComfyUI 的 Intel XPU Python，并把路径写给独立版。未安装节点时，也可以手动指定：

```bat
run_xess.bat "D:\video\input.mp4" 1.5 --preset quality --torch-python "D:\ComfyUI\python\python.exe"
```

### 工作盘与过程文件保护

- `auto/stream/shared` 模式不会落地整段 RGB raw、运动矢量或深度序列；
- 每个任务使用独立工作目录，成功或失败都会尝试清理；
- 输出先写 `.partial.mp4`，通过分辨率、帧率和帧数验证后再原子改名；
- 非系统盘默认至少保留 5 GiB，系统盘至少保留 25 GiB；
- 要指定工作目录可加 `--work-dir "D:\XeSS-work"`；
- `--keep` 仅用于调试，可能保留大量过程文件，不建议普通用户使用；
- `--io-mode file` 会产生大 raw，普通使用不要选择。

查看全部参数：

```bat
run_xess.bat --help
run_fg.bat --help
run_pipeline.bat --help
```

## 二、ComfyUI 节点版

### 自动安装

1. 完整解压本套件；
2. 关闭正在运行的 ComfyUI；
3. 双击根目录的 `install_comfyui.bat`；
4. 如果没有自动找到 ComfyUI，输入包含 `main.py` 的 `ComfyUI` 文件夹；
5. 安装完成后重启 ComfyUI，搜索 `XeSS`。

也可以从 PowerShell 明确指定路径：

```powershell
.\install_comfyui.ps1 -ComfyUIPath "D:\ComfyUI\ComfyUI" -WorkDir "D:\XeSS-Video-Work"
```

安装脚本会：

- 检测便携引擎是否完整；
- 检测 Intel Arc、RTSS、磁盘空间和 Python 模块；
- 把 `ComfyUI-XeSS` 复制到 `ComfyUI/custom_nodes`；
- 自动写入实际引擎路径和安全工作目录；
- 把随包工作流复制到 `ComfyUI/user/default/workflows`；
- 发现旧版节点或同名工作流时先创建时间戳备份；
- 仅在缺少 NumPy/OpenCV 时尝试安装这两个基础依赖，不会擅自重装 PyTorch。

如果不希望脚本安装 Python 基础依赖：

```powershell
.\install_comfyui.ps1 -ComfyUIPath "D:\ComfyUI\ComfyUI" -SkipPythonInstall
```

### 手动安装

1. 把根目录的 `ComfyUI-XeSS` 复制到 `ComfyUI/custom_nodes/ComfyUI-XeSS`；
2. 把 `workflows/xess超分帧生成.json` 导入 ComfyUI；
3. 复制 `ComfyUI-XeSS/xess_config.example.json` 为 `xess_config.json`；
4. 修改其中的 `engine_path` 与 `work_dir` 为本机绝对路径；
5. 重启 ComfyUI。

### 节点怎么用

普通用户只需两个中文节点：

- `XeSS 视频超分（两挡自动）`；
- `XeSS 视频插帧（两挡自动）`。

推荐链路：

```text
Load Video → XeSS 视频超分（两挡自动） → XeSS 视频插帧（两挡自动） → Save Video
```

两个节点都使用 ComfyUI 原生 `VIDEO` 类型，会读取源帧率并透传音频；FG 自动将帧率翻倍。每个节点只显示两个主档位：

- `极速模式（最低挡）`：DIS 路线，适合快速成片；
- `极致画质（最高挡）`：双向 SEA-RAFT、AI 深度和五帧信息，适合短片或最终输出。

需要精调时再使用 `XeSS 视频处理/专家` 下的节点。随包的 `xess超分帧生成.json` 已同时放好快速、极致和专家示例。

注意：ComfyUI 的 `VIDEO` 最终仍会把输出 IMAGE 批次放在内存中。超长视频应分段处理；节点默认在预计输出超过 12 GiB 时提前停止，避免把内存和页面文件挤满。

### 专家模式推荐参数

专家节点不是简单地把所有数值拉满。完整的逐项解释、风险范围和十套配置见 [ComfyUI-XeSS/EXPERT_GUIDE.md](ComfyUI-XeSS/EXPERT_GUIDE.md)。这里先给最常用的四套：

| 用途 | 核心配置 | 五帧/MFSR | 锐化与保护 |
|---|---|---|---|
| 480p→720p 快速批量 | 快速、Q 自动、DIS、高分辨率 MV、响应 0.80 | 融合 0、MFSR 关 | 固定 0.25、振铃保护 0.75 |
| 真人脸部 SR | 高质量、Q5、双向 SEA-RAFT、深度 MV、一致性 1.0 | 融合 0.25、MFSR 关 | 自适应 0.20/0.08、保护 0.90 |
| 真人脸部 FG | 高质量、双向 SEA-RAFT、AI 深度、5 帧、一致性 1.0 | 运动修正 0.55、深度 0.12 | 自适应 0.12/0.04 |
| 复杂遮挡 FG | 高质量、双向 SEA-RAFT、AI 深度、扩张 2、深度边缘 0.025～0.03 | 运动修正 0.70、深度 0.15 | 关闭或 0.08/0.03 |

SR 专家节点已单独暴露 `竖向边缘振铃保护`：默认 0.75，真人脸部推荐 0.90，0 为关闭。MFSR 仅推荐静态风景/建筑使用保守组合：注入 0.8、细节增强 0.45、最大注入 10、锐化 0.16/0.05；真人近景或暗部硬阴影不要开启。

建议每次用相同的 3～5 秒片段 A/B，只改一个参数。出现竖线时按“关闭 MFSR → 降低锐化 → 提高振铃保护”处理；出现拖影时先降低五帧融合/运动修正，再把光流一致性阈值调低。

## FG 特别注意事项

1. 运行 FG 前完全退出 RTSS/MSI Afterburner/性能叠加层。只关监控窗口不一定会结束 `RTSS.exe`。
2. FG 运行时不要锁屏、切换用户或让远程桌面断开。
3. 不要最小化 XeSS 的捕获桌面会话。
4. 如果必须保留 RTSS，先给 `xess-fg.exe` 建独立配置并设 `Detection level=None`、关闭 OSD，再在专家节点显式允许；默认节点会安全停止。

## 常见问题

### 找不到 XeSS pipeline 引擎

请保持解压后的目录结构不变并重新运行安装脚本。节点读取 `custom_nodes/ComfyUI-XeSS/xess_config.json`，移动整套文件后需要重新安装一次来刷新路径。

### 极速档能用，极致档报 torch.xpu/OpenVINO 不可用

极致档的 SEA-RAFT 需要 ComfyUI Python 中的 Intel XPU 版 PyTorch 和 OpenVINO。运行 `check_environment.bat` 查看检测结果。普通 CUDA PyTorch 不能在 Intel Arc 上直接提供这条路径；环境不满足时先使用极速档。

### 鼻子或硬阴影旁出现细竖线

这是 XeSS 高频振铃被多帧残差或锐化放大的典型表现。SR 1.2 默认启用边缘保护，简易节点也默认开启。不要为了“更锐”盲目把 MFSR 注入和锐化同时拉满；专家节点中先关闭 MFSR，再降低锐化。

### 遮挡边缘破碎或重影

快速档的 DIS 光流在复杂遮挡处信息有限。可以改用极致档，AI 深度和双向 SEA-RAFT 会改善遮挡判断，但不能恢复源视频中从未出现的真实细节。

### 系统盘空间不足

工具默认流式处理并保留 25 GiB 系统盘空间。若仍被拒绝，清理空间或用 `-WorkDir`/`--work-dir` 指向其他磁盘。不要使用 `--keep` 或 `--io-mode file`。

### 输出没有生成

查看命令窗口最后的错误；失败任务不会把 `.partial.mp4` 当成成品。FG 问题优先确认 RTSS 已完全退出、桌面会话保持活动、显卡驱动正常。

## 目录说明

```text
XeSS-Video-Suite-2026.08/
├─ README.md
├─ check_environment.bat
├─ install_comfyui.bat
├─ install_comfyui.ps1
├─ xess-portable-pipeline/   独立运行时与命令行工具
├─ ComfyUI-XeSS/             ComfyUI 自定义节点源码
├─ workflows/                可直接导入的工作流
├─ source/                   C++/Python 构建源码与 SDK 开发文件
├─ licenses/                 许可和第三方声明
└─ SHA256SUMS.txt            包内关键文件校验
```

## 许可与声明

- Intel XeSS/XeLL 二进制、头文件和导入库遵循包内 Intel SDK 许可及第三方声明；
- 其他第三方组件见 `licenses/THIRD_PARTY_NOTICES.md`；
- 本项目不宣称与 Intel 存在官方隶属或背书关系；
- 上传 GitHub 时，请把大 ZIP 作为 **GitHub Release 附件**，不要直接提交到普通 Git 历史。

处理在本机离线完成，不会自动上传用户视频。
