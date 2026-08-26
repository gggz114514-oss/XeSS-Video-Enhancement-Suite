# r4 全链路优化探索计划（交给 Zcode）

## 0. 任务定位

本轮是在已验收并发布的 1.3.0 / Runtime r3 之上继续探索，不允许直接修改或覆盖
r3 工作树、r3 Release 资产和历史测量目录。目标不是堆更多算法，而是寻找仍能被数据证明的
端到端收益，并把独立版与 ComfyUI 节点的实际用户路径一起纳入验收。

已确认的主线边界：

- 光流主线只保留 OpenCV DIS；不要把 SEA-RAFT、PyTorch XPU 或其模型重新带回主线。
- 暂不开发 SR→FG 联合入口；SR、FG 分开测、分开优化。
- 默认输出必须稳定、顺序正确、过程文件受控；不能用“可能更好”替代量化和人工检查。
- 所有任务文件、缓存、视频和构建产物都放 E 盘；不得使用系统临时目录保存大文件。
- 未获得明确授权前不得 push、建 PR、建 Release、改 tag 或合并。

## 1. r3 封存与新工作区

先更新 `origin/main`，确认它包含 1.3.0 / `runtime-2026.08.28-r3`，然后执行：

1. 记录 main 提交、Runtime 清单、r3 ZIP SHA256、`xess-vsr.exe` SHA256。
2. 在 E 盘创建只读 Git bundle 和 `BASELINE_SEAL.md`。
3. 新建分支 `codex/r4-quality-runtime-exploration`。
4. 在 E 盘项目根目录下新建工作树：
   `<E盘项目根目录>\git\worktrees\r4-quality-runtime-exploration`
5. 所有运行产物固定放到：
   `<E盘项目根目录>\work\r4-quality-runtime-exploration`

封存报告必须证明 r3 工作树在任务开始和结束时提交号、状态均未变化。

## 2. 必须先复现的 r3 基线

使用同会话交错三次、中位数、流式 SHA256：

| 场景 | r2 基线 | r3 基线 | r3 原始 RGB SHA256 |
|---|---:|---:|---|
| A：864×480→1296×720，243 帧，Fast，不含编码 | 15.488s | 7.683s | `31ddf78ca075ecb17b765cf09a04b72275e75bc946964aba5046811a1e328dcf` |
| B：1080×1920→1440×2560，300 帧，Fast，不含编码 | 57.702s | 46.117s | `5907a7894a0c55996b263b6ee435bebbc8ad23f52a52d80f5ece9c9913f634ab` |

先复现功能和哈希；机器状态造成的时间漂移允许记录，但若控制样本漂移超过 10%，不得用该轮
绝对时间宣布收益。性能比较必须交错运行，不能先跑完全部旧版再跑新版。

另外记录两个真实端到端基线：

- A、B 从视频输入到 MP4 完成的总时间、解码、prepare、XeSS、post、编码分项。
- 赛车 60→120 FG：前 300 输入帧，严格检查 `2N-1`、帧率、音轨和时长。

## 3. Phase A：ComfyUI SR 节点与独立版性能对齐（最高优先级）

r3 的独立版已经使用第二组输出共享内存和 4 线程有序后处理；ComfyUI 节点仍从
`xess-vsr.exe` stdout 逐帧读入 NumPy，并在主进程串行锐化、护边。这不是画质缺陷，
但会让节点用户拿不到完整的 r3 收益。

按以下顺序探索：

1. 给节点内存链增加对称输出 ring；直接把 slot 复制进预分配的输出数组，避免额外整帧对象。
2. 保留 pipe 回退和显式诊断开关；自动模式在输出短边达到 720 时选择 shared。
3. 把节点锐化与竖纹保护改成线程池并行、按帧序排水；不得同时修改算法公式。
4. 处理 ComfyUI 中断、worker 提前退出、输出 ring 写满、少帧/多帧、单帧复制成双帧等路径。
5. 测节点进程常驻时第二次执行，排除首次 DLL/模型加载造成的假收益。

门禁：

- 节点输出逐字节等于 r3 节点基线；IMAGE 张量形状、dtype、范围不变。
- A、B 两场景节点核心链中位数至少提升 15%，否则只保留诊断结果，不进默认。
- 中断后 10 秒内所有子进程退出，无命名共享内存残留、无 `.partial`。
- 不能引入新的 torch/safetensors 子进程依赖。

## 4. Phase B：FG prepare 真瓶颈——深度与一致性

r3 已测赛车 FG 的耗时主要在 DIS 一致性和 OpenVINO 深度，不在 XeFG 本体。先用阶段计时重新
确认占比，再逐项做相互独立的实验，禁止一次混入多个变量：

1. 深度低分辨率推理：短边 720/540/360，回原分辨率后做边缘引导上采样。
2. 深度时间复用：每 2/3/4 帧推理一次，中间帧用 DIS 光流前向投影；场景切换立即强制刷新。
3. 深度异步流水：OpenVINO 与下一帧 DIS/打包重叠；不得让同一 compiled model 并发乱序。
4. 一致性/mask 的 OpenCV 向量化和缓存复用；先证明输出一致，再谈默认启用。
5. 仅当测量证明必要时，尝试 OpenVINO GPU 的 batch 或 remote tensor；不要先做大规模重写。

画质测试不能只拿“新旧输出相似”当结论。赛车源为 60fps 时，取偶数帧构造 30fps 输入，
将生成的中间帧与被留出的真实奇数帧比较：全图 PSNR/SSIM、运动边缘区域误差、遮挡区域误差，
并人工检查轮胎文字、车窗立柱、烟雾、护栏和快速摇镜。另加人物脸部、暗部及场景切换短片。

默认候选门禁：相对当前 FG 基线端到端至少快 15%；平均 SSIM 不下降超过 0.003；运动边缘
95 分位误差不恶化超过 3%；不得新增肉眼可见重影。未过门禁的方案只能保留为专家实验。

## 5. Phase C：XeSS 正确 jitter 画质实验

当前 jitter 常量为 0。正确路线不是只改常量，而是同时实现：

- 输入帧的亚像素重采样；
- jitter 前后坐标系一致的运动矢量修正；
- 当前帧/上一帧符号和像素单位专项测试；
- 边界填充、场景切换和响应式 mask 对齐。

先构造可重复的静态斜线、细字、栅栏和亚像素平移序列，验证 jitter 是否真的增加可恢复采样，
再跑真实视频。与 SeedVR2 的“生成式锐利”不是本轮目标；只接受可解释、稳定、低成本的细节改善。

默认候选门禁：静态/缓慢运动高频指标有一致提升，真实视频不新增闪烁、爬纹、竖线或边缘分叉，
端到端耗时增加不超过 8%。否则保留为实验开关。

## 6. Phase D：F16 RNE/F16C 决策

r3 已证明正确 RNE 与 F16C 位级一致，但相对历史转换约 2.43% 输出字节变化，PSNR 约
64.2 dB、最大 ±1 色阶。补足以下测试后再决定是否改变默认：

- B580 的 A/B/C 场景与异常竖纹历史片段；
- DG2/A770 真机；
- 1000 帧以上稳定性和 CPU 不支持 F16C 的标量回退；
- XeSS 最终画面、而非仅 velocity buffer 的差异分布。

除非能证明质量修正且不存在新回归，否则 1.4 默认继续 legacy，RNE/F16C 只保留专家开关。

## 7. Phase E：真实端到端编解码探索

先用阶段计时证明 x264/软件解码在目标场景中的占比，再做 QSV POC：

- `h264_qsv`/`hevc_qsv` 编码与软件 `libx264` 的速度、码率、VMAF/SSIM、暗部块效应比较；
- QSV 解码到 CPU RGB 与现有管线的转换开销；
- 先做显式 `--encoder qsv`，不得在未知设备上自动替换默认；
- 编码失败必须删除 `.partial` 并自动给出可理解的回退提示。

不要把有损编码差异混进 XeSS 原始画面质量结论。所有算法 A/B 先比较 raw/FFV1，再单独比较编码器。

## 8. Phase F：A770 / DG2 交付包

本机没有 A770，不得写“A770 已验证”。准备一个不含源码构建要求的一键验收包，输出：

- GPU、驱动、ComfyUI/Python/OpenVINO/Runtime 版本；
- 48 帧 SR Fast/Quality、FG Fast；
- shared/pipe 回退；
- AVX2/scalar 与 legacy/RNE/F16C；
- 峰值显存、输出帧数、SHA256 和日志压缩包。

脚本只能写到用户选择的工作目录，不得默认写 C 盘或 `%TEMP%` 大文件。

## 9. 测试矩阵与故障注入

每个可能进入默认的改动都必须覆盖：

- 480p→720p、1080p→2K、奇数尺寸、1 帧/2 帧、音频有/无；
- Fast 与 Quality 两个公开预设；独立版与 ComfyUI 节点；
- 第 0 帧输入结束、中途少帧、stdout/编码器提前关闭、共享内存消费者退出；
- `io-mode=stream/shared/file` 兼容路径；
- 30 次短任务循环和 1000 帧长任务，无句柄/共享内存/显存持续增长；
- `compileall`、全仓单测、`validate_repo.py`、`git diff --check`。

## 10. 实验纪律与提交方式

每个方向单独提交，提交信息必须说明“实验/保留/拒绝”。每项至少记录：假设、实现、命令、
原始 CSV/JSON、输出哈希、性能、质量、峰值内存、结论和回滚方法。任何没有同会话基线或没有
质量证据的数据都不能写成提升。

建议提交顺序：

1. `test: seal r4 baselines and failure harness`
2. `perf: prototype ComfyUI output-ring path`
3. `perf: parallelize ordered node postprocess`
4. `experiment: evaluate FG depth scheduling`
5. `experiment: implement coordinate-correct XeSS jitter`
6. `experiment: qualify F16 RNE and F16C`
7. `experiment: benchmark opt-in QSV codec path`
8. `docs: record accepted and rejected r4 directions`

最终报告必须把“进入默认”“仅专家开关”“证伪删除”分成三张表，并列出所有提交哈希、改动文件、
实测设备、输出路径、已知风险。完成后停在本地干净分支，等待人工审查，不要自行发布。
