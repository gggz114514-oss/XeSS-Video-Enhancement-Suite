# ComfyUI R4 发布验收

状态：本地候选验证中，尚未发布。后续发布提交补齐真实 ComfyUI 与公开下载证据。

## 范围

从公开 R3 main（3f44b57）新建独立分支，选取已交付离线引擎；源实验工作树
b21142e 不变，旧节点按用户要求移除。独立 HTML/WPF、WGC/OBS 工具箱不发布。
三个新节点以及 CLI 共享离线引擎；不改现有 ComfyUI 配置、依赖或用户视频。

## 已通过

- 160 个 unittest（包含 15 个安装/注册故障测试、4 个现行原生入口静态生命周期
  合同测试、2 个护边逐帧输出回归）；ResourceWarning 作为错误。
- 75/75 真机成片：5 路线 × SR/FG/SRFG × 5 编码器，8 帧输入，独立检查解码帧数、
  codec、音轨。XeFG 为 2N−1，Intel FI 为 2N。FFV1 额外验证终端 RGB 无损哈希。
- 三条 GPU 路线各 12 帧 1080p→4K SR/SRFG，锐化+五帧+抗竖纹均开，6/6
  原生派发计数匹配；三条 GPU 同尺寸独立 FG 锐化 3/3（源为 1080p，不冒称 4K FG）。
- 配套 ZIP 本地解包安装与全文件哈希；隔离 Python、FFmpeg、含依赖 DLL 的 ffprobe
  启动；源码/工作流/清单校验；JS 选项联动测试。
- CPU/GPU/VPL 原生 EXE 由本发布工作树重新构建。DIS 编译着色器继承已验收运行时；
  公共深度/Mask/后处理着色器本轮重新构建。不是每个 shader 均本轮重新编译。

## 本轮发现与修复

- R3 平铺安装器不能加载 R4 嵌套运行时：按版本安装、锁、哈希、失败保留旧版。
- 并发安装锁初始化字节与已锁文件冲突：只锁字节范围，不在获取锁前写入文件。
- GPU 路线必须 AI 深度：隐藏不适用选项且后端拒绝固定深度，避免运行时才报错。
- 旧 pytest 函数未被 unittest 收集且检查已退役 main：替换为实际构建入口的
  4 个静态测试；不把静态检查当作 GPU 故障注入成功。
- CPU 243 帧 SRFG+护边复现 sidecar 反压环：中间 guard 改逐帧输出并 flush，
  仅该中间阶段单帧处理，避免下一输入与 FG sidecar 消费互相等待。
  算法公式不变；这不是一次新性能优化。完整视频复验结果随后补齐。

## 待闭环

长片/CPU 护边复验、隔离真实 ComfyUI 三节点执行、R3 Git 快进升级和公开 Release
下载。所有待项完成前，不把本文件的候选状态当作正式发布。

## 边界

B580 实测，其他显卡未测；单机短矩阵不是所有素材画质保证。五帧为可选项，
快速运动不保证改善。GPU native TerminalPost 历史 applied 字符串是通用描述，
开关判断以 requested 与 dispatch_count 为准。系统已有 Python/torch 配置不改变。

原始证据位于维护者工作目录 r4-comfy-release-20260907（不随 Git 发布媒体）：
matrix75/results.json、effects4k/results.json、fg4k/results.json、unit-160.log；
错误轮 full243 保留，不覆盖为成功轮。
