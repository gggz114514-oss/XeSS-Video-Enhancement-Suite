# ComfyUI R4 发布维护

本仓库只发布节点与离线工作器。不得把 apps、OBS/WGC 前端、大视频、开发缓存、模型或 DLL 提交到 Git。历史目录 `src/realtime` 包含共用工作器源码，不表示发布实时 GUI。

## 源码更新

纯节点/控制层更新可复用匹配的 Runtime；pipeline 在 Git 目录直接运行，不再复制到旧 `.runtime/engine`。EXE、DLL、HLSL、模型或独立 Python 改动必须重新封包与验收。

## 固定资产

```text
python tools/build_comfy_runtime.py --runtime <已验收的nested-runtime> --supplement <本轮重建文件及许可证> --output <隔离发布目录> --version 2026.09.07-r4
```

封包只允许 bin/media/probe/python/models/shaders/licenses。每文件 SHA256、ZIP 大小和展开大小进入清单，初始状态为 not-published。不要覆盖已有发布资产或复用不同内容的 tag。

## 必需证据

1. 当前源码 C++ 工作器构建记录；未重建着色器逐项记录沿用来源和 SHA256，不冒充新编译。
2. `python -W error::ResourceWarning -m unittest discover -s tests -v`；`python tools/validate_repo.py --asset <ZIP>`；JavaScript 选项联动测试。
3. 解压自包含安装，运行 `self_test.py`；R3 Git checkout → 新提交的 fast-forward → 正常 Comfy 启动自动安装。
4. 真实 Comfy 三节点、五路线的 SR/FG/SRFG、编码器矩阵、增强开关组合、长于 841 帧、取消与故障清理；数据分别记录。
5. 输入原片不覆盖；磁盘空间门、所有过程文件仅测试工作目录；不可删除旧 runtime junction 的目标。

## 发布顺序

先完成本地候选提交与验收，再 push 分支并建立 PR；上传固定 tag 的 Release ZIP 和 SHA256。确认上传文件可下载且哈希相同后，将清单标为 published，PR 合并 main。普通用户一次 pull + 重启就能走同一安装器。

旧节点不注册是本次明确的破坏性变更，Release notes 必须注明需要替换 R3 工作流节点。不能声称 A770/NVIDIA/AMD 硬件已测，除非提供该机真实日志。

## 开发构建

MSVC 2022、Windows SDK DXC、XeSS SDK、oneVPL、OpenVINO、OpenCL 头文件和 FidelityFX SDK 均通过显式环境变量传入；使用 `tools/build_offline_encoders.cmd`、`tools/build_vsr_fg.bat`、`tools/build_vpl.bat`。终端构建环境只对子进程有效。最终报告记录确切源提交、SDK 版本、构建参数及产物哈希。
