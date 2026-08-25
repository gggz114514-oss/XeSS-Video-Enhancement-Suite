# XeSS 视频增强工具箱 / XeSS Video Enhancement Suite

面向 Windows 与 Intel Arc 的视频超分、抗锯齿和 2× 帧生成工具，同时提供独立命令行入口与中文 ComfyUI 原生 `VIDEO → VIDEO` 节点。

当前版本：源码/节点 `1.1.0`，SR `1.2`，FG `1.2`，固定运行时 `2026.08.21-r1`。

> **实验分支提示：** 当前分支 `experiment/sea-raft-xpu` 仅用于归档 SEA-RAFT/XPU 融合核实验。该路线已在 B580 的 SR/FG 实测中确认不适合作为主线：速度慢约 2.8–3 倍，画质没有稳定质变，并增加显存和运行时依赖。详见 [`docs/SEA_RAFT_EXPERIMENT_STATUS.md`](docs/SEA_RAFT_EXPERIMENT_STATUS.md)。

> 这是社区项目，不是 Intel 官方产品。实机验证平台为 Windows 11 与 Intel Arc B580。
> ## 实测环境基线

以下是项目当前实际测试环境，不代表唯一支持版本。

| 项目 | 当前实测值 |
|---|---|
| 整合包 | ComfyUI-aki-v3-IntelArc_20260722 |
| ComfyUI 核心 | 0.33.1 |
| Python | 3.13.11 |
| PyTorch | 2.13.0+xpu |
| XPU 状态 | `torch.xpu.is_available() = True` |
| XPU 设备数 | 1 |
| OpenVINO | 2025.4.1 |
| 显卡 | Intel Arc B580 |
| XeSS 节点提交 | `e20e986` |
| XeSS Runtime | `2026.08.21-r1` |
| 操作系统 | Windows 11 |

### 依赖说明

- 极速模式使用 DIS 光流，不需要 SEA-RAFT。
- 均衡和极致画质模式使用 SEA-RAFT；默认在超过 720p 时等比例把短边缩到
  720 像素分析，再把光流恢复到原尺寸。它需要在 ComfyUI 实际使用的
  Python 环境中提供：
  - Intel XPU 版 PyTorch；
  - `safetensors`；
  - OpenVINO。
- `.runtime/engine` 是 XeSS 固定运行时，不是 ComfyUI 的 Python 依赖环境。
- 不要把普通 CPU/CUDA 版 PyTorch 当作 Intel XPU 版使用。

### 反馈问题时请提供

请同时提供以下信息：

1. ComfyUI 核心版本；
2. Python 版本；
3. PyTorch 版本；
4. `torch.xpu.is_available()` 的结果；
5. OpenVINO 版本；
6. XeSS 节点 Git 提交或目录版本；
7. XeSS Runtime 版本；
8. 完整的 ComfyUI 控制台日志；
9. 使用的节点挡位和光流模式。

## 本次版本重点：直接拦截 XeFG 交换链

这一版不再把窗口画面当成帧生成结果。`xess-fg.exe` 在 XeFG 初始化期间包装 DXGI 工厂，记录 XeFG 内部创建的原生交换链；每次代理交换链完成 Present 后，程序直接从最后呈现的 D3D12 后缓冲回读生成帧，再送入流式编码管线。

```text
输入帧 + 光流/深度
        ↓
XeFG 代理交换链 Present
        ↓
DXGI 工厂包装器记录原生交换链
        ↓
回读最后呈现的 D3D12 后缓冲
        ↓
f0, G1, f1, G2 ... → 2× fps 视频
```

因此默认 `direct` 模式具有这些特性：

- 拿到的是 XeFG 实际生成帧，而不是帧复制或普通光流合成结果；
- 不依赖 Windows Graphics Capture，不需要录制桌面或裁剪隐藏窗口；
- 不受 Windows 高 DPI 坐标缩放、窗口遮挡、最小化和黄色捕获边框影响；
- RTSS/MSI Afterburner 的桌面 OSD 不会混进输出视频；
- 独立版和 ComfyUI 节点使用同一套拦截式 FG 执行链路。

旧的 `window` 后端仅保留给开发者诊断，不建议普通用户启用。

## 这次为什么改成 Git + Release

仓库现在按“经常更新的代码”和“很少变化的大资源”拆分：

| 位置 | 内容 | 更新方式 |
|---|---|---|
| Git 仓库 | ComfyUI 节点、Python 管线、C++ 源码、工作流、安装器和文档 | 秋叶启动器/ComfyUI Manager/Git pull |
| GitHub Release | XeSS/XeFG/XeLL 二进制、ffmpeg、便携 Python、OpenVINO 与模型 | 仅在 `runtime_manifest.json` 指向新版本时下载 |
| `.runtime/` | 本机已安装的固定运行时 | 被 `.gitignore` 忽略，源码更新不会删除或重复下载 |

普通代码更新只拉取几十个小文件。每次运行前会把最新 `pipeline/` 同步到本机运行时，通常不到一秒；只有 exe、DLL、模型或便携 Python 确实变化时，才需要发布并下载新的 Runtime 资产。

## 一、秋叶启动器 / ComfyUI 安装

在秋叶启动器的自定义节点管理中选择“通过 Git URL 安装”，填写：

```text
https://github.com/gggz114514-oss/XeSS-Video-Enhancement-Suite.git
```

安装过程会：

1. 克隆本仓库到 `ComfyUI/custom_nodes`；
2. 安装缺失的 NumPy/OpenCV 基础依赖；
3. 执行 `install.py`；
4. 从固定 Release 下载一次约 303 MiB 的运行时；
5. 校验 SHA256 后解压到节点目录的 `.runtime/engine`。

安装完成后重启 ComfyUI，搜索 `XeSS`。以后在秋叶启动器点击“更新”即可，不需要重新安装节点，也不会重复下载未变化的运行时。

首次安装需要能够访问 GitHub Release。源码更新和固定 Runtime 是分开的：更新节点通常只下载少量文本文件，只有清单中的 Runtime 版本变化时才会重新下载大文件。

### 手动 Git 安装

```bat
cd /d "你的ComfyUI目录\custom_nodes"
git clone https://github.com/gggz114514-oss/XeSS-Video-Enhancement-Suite.git
cd XeSS-Video-Enhancement-Suite
install_runtime.bat
```

然后重启 ComfyUI。没有 D/E 盘也能安装；运行时、缓存和输出路径都按仓库实际位置或用户配置计算，不写死盘符。

### 从旧整包迁移

旧版 `ComfyUI-XeSS` 是普通复制目录，秋叶启动器无法对它执行 Git 更新。迁移到本仓库只需做一次：

1. 关闭 ComfyUI；
2. 备份旧 `ComfyUI/custom_nodes/ComfyUI-XeSS` 中的 `xess_config.json`；
3. 将旧目录改名为 `ComfyUI-XeSS.old`；
4. 用秋叶启动器通过上面的 Git URL 安装；
5. 通常保持 `engine_path=auto`、`work_dir=auto` 即可；如有特殊工作盘设置，再复制旧配置。

完成这一次迁移后，后续版本只需点“更新”。

## 二、ComfyUI 节点

普通用户主要使用：

- `XeSS 视频超分（两挡自动）`
- `XeSS 视频插帧（两挡自动）`

两者都接受原生 `VIDEO`，自动读取帧率并保留音频。推荐工作流：

```text
Load Video → XeSS 视频超分（两挡自动） → XeSS 视频插帧（两挡自动） → Save Video
```

主档位只有两套：

- `极速模式（最低挡）`：DIS 光流，速度优先；
- `极致画质（最高挡）`：双向 SEA-RAFT（自动 720p 分析）、AI 深度与五帧信息，适合复杂运动。

输入短边不超过 720 像素时 SEA-RAFT 使用原生分辨率。专家节点可切换到
`原生分辨率（实验 / 极慢）`，但高分辨率下会大幅降低速度，当前实测没有
画质优势，仅建议用于研究和对照。

需要逐项调节时使用 `XeSS 视频处理/专家` 分类。完整参数方案见 [EXPERT_GUIDE.md](EXPERT_GUIDE.md)，示例工作流位于 [workflows/xess超分帧生成.json](workflows/xess超分帧生成.json)。

## 三、独立版

克隆仓库后双击一次 `install_runtime.bat`。它只把固定运行时安装到当前仓库的 `.runtime`，不要求系统 Python。

480p 放大到 720p：

```bat
run_xess.bat "C:\Videos\input.mp4" 1.5 --preset fast
```

24fps 插帧到 48fps：

```bat
run_fg.bat "C:\Videos\input.mp4" --preset fast
```

高质量模式默认自动使用 720p SEA-RAFT 分析：

```bat
run_xess.bat "C:\Videos\input.mp4" 1.5 --preset quality
run_fg.bat "C:\Videos\input_xess_sr12_quality_1.5x_1296x720.mp4" --preset quality
```

如需进行原生高分辨率对照，可额外传入 `--flow-resolution native`。该选项
速度极慢，不作为推荐生产配置。SR 与 FG 建议顺序运行；同时启动两套
SEA-RAFT 的联合管线暂不作为支持目标。

入口脚本每次启动会先检查 Runtime 清单并同步 Git 中的新管线代码。Runtime 版本没变时不会联网下载。

## 四、交换链拦截式帧生成

FG 默认并强制从上层管线选择 `direct` 后端：通过 DXGI 工厂包装器记录 XeFG 创建的原生交换链，等待代理 Present 完成后直接回读实际生成帧。

- 不使用 Windows Graphics Capture；
- 不受高 DPI、窗口遮挡、最小化或黄色捕获边框影响；
- RTSS/MSI Afterburner 的桌面 OSD 不会进入输出；
- 命令行仍保留 `--capture-mode window` 作为旧版诊断回退。

N 个输入帧严格输出 `2N-1` 帧，顺序为 `f0,G1,f1,G2...`，输出帧率为输入的两倍。

运行时日志出现下面一行，表示正在使用交换链拦截路径：

```text
[capture] mode=direct (native swap-chain readback)
```

如日志显示 `window`，说明手动传入了旧诊断参数；删除 `--capture-mode window` 即可恢复默认模式。ComfyUI 普通节点不需要设置该参数。

## 五、过程文件和磁盘保护

- 默认使用流式管道或共享内存，不落地整段 RGB raw、光流或深度序列；
- 每个任务使用独立工作目录，结束后清理临时数据；
- 输出先写 `.partial.mp4`，通过分辨率、帧数和帧率验证后再原子改名；
- 非系统盘默认保留至少 5 GiB，系统盘默认保留至少 25 GiB；
- `.runtime` 固定资源约 624 MiB，只在 Runtime 版本变化时更新；
- `--keep` 与 `--io-mode file` 只用于调试，可能产生大量文件。

如需指定工作盘：

```bat
run_fg.bat "C:\Videos\input.mp4" --work-dir "F:\XeSS-Work"
```

## 六、Runtime 版本与校验

当前固定资产：

```text
Tag: runtime-2026.08.21-r1
Asset: xess-runtime-windows-x64-2026.08.21-r1.zip
SHA256: 0f69db8f652d4b63d849bd8f27fa6cc8950cd7ef98bfea94461230230e85f78b
Archive: 303.32 MiB
Installed: 624.49 MiB
```

下载地址和逐文件兼容哈希由 [runtime_manifest.json](runtime_manifest.json) 固定。安装器拒绝 SHA256 不匹配、路径穿越或超出清单安全上限的压缩包。

手动检查：

```bat
.runtime\engine\python\python.exe runtime_manager.py status
```

强制重新安装：

```bat
install_runtime.bat -Force
```

## 七、源码构建

C++ 源码位于 `src/`。固定 Runtime 已包含 Intel XeSS SDK 2.1 的开发头文件和导入库；另外需要 Visual Studio 2022 Build Tools 与 Windows SDK。

安装 Runtime 后直接运行 `build.bat`。如需使用另一套 SDK，可将 `XESS_SDK_ROOT` 指向包含 `inc`、`lib` 和 `bin` 的目录：

```bat
set "XESS_SDK_ROOT=C:\SDK\XeSS"
build.bat
```

构建产物写入本仓库 `build/`，不会写入系统临时盘。

## 八、维护者发布规则

- 只改节点/Python/C++源码/文档：更新 Git 即可，不创建 Runtime Release；
- 改 exe、DLL、模型、ffmpeg 或便携 Python：构建新 Runtime 资产，发布新 `runtime-*` 标签，并提交更新后的 `runtime_manifest.json`；
- 不把 `.runtime`、模型、DLL、exe、视频或 raw 提交到 Git；CI 会拒绝超过 10 MiB 的固定资产。

详细流程见 [docs/MAINTAINER_RELEASE.md](docs/MAINTAINER_RELEASE.md)。

## 许可与声明

Intel XeSS/XeLL 与其他第三方组件保留各自许可。相关文本见 [licenses/](licenses/) 和 [THIRD_PARTY_NOTICES.md](licenses/THIRD_PARTY_NOTICES.md)。本项目不宣称获得 Intel 官方隶属或背书。
