# ComfyUI R4 发布验收

状态：本地发布门通过。公开上传/下载及主线合并状态将在最终发布记录补齐。

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
  算法公式不变；这不是一次新性能优化。243 帧同一源全片复验成功：GPU Block
  10.337 秒、CPU DIS 32.116 秒，均为 485 帧，含音轨。单轮时间仅是完整性记录，
  不能当作受控性能排名。

## 安装与完整视频复验

- 1000 帧固定帧率测试输入（重复样片，用于稳定性而非画质评级）→ GPU Block
  SRFG 1999 帧，30.463 秒，跨 841，PTS/音轨通过。原先循环拼接素材存在 PTS
  不连续，后端正确拒绝；重新明确生成 CFR 测试素材后通过，没有放松生产 VFR 门。
- 编码器提前退出：3.017 秒有界失败、不输出假成片；活动软件编码取消：
  0.140 秒退出，status=cancelled，无最终假成片。
- 独立 E 盘 ComfyUI 0.33.1 / Python 3.13.11 / B580 环境，Git checkout R3
  3f44b57 → 一次 pull --ff-only → 53b1e89。只通过正常启动导入插件自动安装，
  使用 ZIP 离线资产参数，不设置开发 XESS_RUNTIME_ROOT，不手工复制已安装引擎。
- 真实 HTTP prompt 执行三个新节点：8 帧 SR 输出 8 帧 1296×720；独立 FG
  输出 15 帧 864×480；组合输出 15 帧 1296×720；VIDEO→SaveVideo、音轨均通过。
  SR/组合五帧与抗竖纹开启，三者终端锐化开启。旧 R3 节点未注册。
- R3 engine 哨兵和用户设置哨兵未变；真实用户现有 ComfyUI/plugins/配置未改。
  此处不是重复验收 R3 推理，而是 Git 快进、旧目录保留和新节点执行验收。
- 本地 clone 发现开发仓库存有不兼容 Windows 文件名的历史实验 tag；测试 clone
  使用 --no-tags。已核验公开 origin 没有该 tag，发布只推本分支，绝不推所有 tags。

公开 Release 网络下载验证仍待上传；不能用离线资产安装测试冒充网络下载成功。

## 边界

B580 实测，其他显卡未测；单机短矩阵不是所有素材画质保证。五帧为可选项，
快速运动不保证改善。GPU native TerminalPost 历史 applied 字符串是通用描述，
开关判断以 requested 与 dispatch_count 为准。系统已有 Python/torch 配置不改变。

原始证据位于维护者工作目录 r4-comfy-release-20260907（不随 Git 发布媒体）：
matrix75/results.json、effects4k/results.json、fg4k/results.json、unit-latest.log、
full243-fixed/results.json、long1000-cfr/results.json、encoder-faults/results.json、
comfy-real/result.json、comfy-server.log；
错误轮 full243 保留，不覆盖为成功轮。
