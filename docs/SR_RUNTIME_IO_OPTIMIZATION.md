# SR Runtime & I/O Optimization (2026-08-26 round)

状态：本轮（稳定性修复 + 发布链路 + 性能优化）执行记录；仅本地验证，未执行任何
远程发布操作（无 push / PR / Release / tag）。

## 1. 当前瓶颈构成（B580，300 帧 1080×1920 → 1440×2560，worker 链阶段计时）

一次带 `--stage-timing` 的实测（新构建，含 SIMD/预分配）：

| 阶段 | 耗时 | 说明 |
|---|---|---|
| prepare-sr analyze_total | 29.8s | DIS + confidence + scene cut + responsive mask |
| prepare-sr packet_encode | 3.1s | 协议头 + 分块 CRC |
| prepare-sr transport_write | 15.6s | ring 写入 + worker_read_wait（等消费者） |
| xess-vsr input_read | 10.7s | ring → 字段缓冲（剩余必要拷贝） |
| xess-vsr texture_upload_cpu | 22.1s | RGB→RGBA + MV 双线性上采样 + 上传填充 |
| xess-vsr stdout_write | 20.2s | RGBA→RGB + 单次 fwrite |
| xess-vsr xess_execute_gpu | 0.36s | XeSS 神经网络本体只占总时长约 0.6% |
| sr-post sharpen / guard_blend | 25.3 / 26.7s | 与上游重叠，链尾协同瓶颈 |

结论：XeSS Execute 本体不是瓶颈；时间花在 prepare 分析、像素格式转换/拷贝、
后处理三块。prepare 的 `worker_read_wait` ≈ 15s 说明 worker 消费速度是关键。

## 2. SEA-RAFT 退役边界

- 主线不再含 SEA-RAFT：无 torch 子进程、safetensors、XPU corr、模型发现逻辑。
- `sea-raft` / `sea-raft-single` 选项一律映射为原生 Fast DIS 单向（含
  `--engine sea-raft --bidirectional` → 强制单向并提示一次；专家
  `--engine dis --bidirectional` 不变）。
- 实验代码只保留在 `experiment/sea-raft-xpu`。

## 3. 三槽流水线真实工作方式（修正后的描述）

xess-vsr.exe 没有独立“写线程”：主线程按 3 槽流水线推进，每槽有独立命令
分配器/列表 + 持久映射上传/回读缓冲；交换顺序为

    提交（最多提前 3 帧）→ 等帧 N fence → RGBA→RGB 打包 → 单次写出

槽位复用由每槽 fence 保证；帧序与提交序一致。`writeDoneSem` 在循环不变量下
等待条件不可达，已删除（方案 A：保守修正，文档同步更新）。

## 4. GPU 异常收尾设计（本轮新增）

任何退出路径（输入提前结束、下游关闭、写失败、正常完成）在
`xessDestroyContext` 与 D3D12 资源析构前：

1. 记录最后一个成功提交的 fence（`lastSubmittedFence`）；
2. `wait_fence` 有界等待（15s 超时），超时/失败都打印结果并把整轮标记失败；
3. Signal/SetEventOnCompletion 返回值统一检查；
4. 收尾日志打印 `submitted / written / last_submitted_fence / drain_result`。

故障注入验证（合成包喂 stdin，真 GPU 运行）：

| 场景 | 结果 |
|---|---|
| 第 0 帧前输入结束 | rc=1，submitted=0，skipped，0.2s |
| 提交 1 帧后输入结束 | rc=1，submitted=1 written=0，drain ok |
| 下游消费 3 帧后关闭 | rc=1，submitted=6 written=3，drain ok |
| stdout 第 0 帧关闭 | rc=1，submitted=3 written=0，drain ok |
| 正常完整 + EOS | rc=0，输出字节数精确，skipped |

无卡死、无 Device Removed、无 Access Violation、无 `.partial.mp4` 残留。

## 5. shared ring 复制链路（前 → 后）

优化前每帧：

    Python ndarray → tobytes ×3 → bytes 拼接(color+motion+depth+mask)
    → encode(header+payload) → RingWriter.write 整包 slice → mmap
    → C++ packet vector → payload vector → color/motion/depth/mask vector → upload

优化后：

    Python ndarray（C 连续）→ 协议头(分块 CRC) + RingWriter.write_parts
    → mmap 槽直写（每段一次 slice，无整包 bytes）
    → C++ peek() 拿槽位视图 → 校验/CRC 原位 → 按字段拷入 StreamFrame
    → commit() 释放槽位 → swap 交接给上传缓冲

协议 v1、CRC、帧序、尺寸校验全部保留；mmap 槽在消费者复制完成后才释放
（`commit()` 先 MemoryBarrier 再推进 readSequence）。

## 6. SIMD 路径与回退

- 运行时 `__cpuid` / `_xgetbv` 检测 AVX2 + OSXSAVE，不可用自动标量。
- RGB24→RGBA8、RGBA8→RGB24：4 像素 128 位组（避免 24B/32B 跨 lane 错位），
  标量尾部；有界读取不过界。
- 逐字节一致已由 48 帧双传输对比验证（哈希不变，见 §10）。
- FP32→FP16 **保持标量**：逐位扫描发现旧标量 `f32_to_f16` 在
  “half mantissa LSB=1 且 remainder>halfway” 时丢失进位，与 IEEE RNE
  （numpy / F16C 硬件）相差 1 ULP；按质量门禁规则，硬件舍入与旧实现不一致时
  不得直接替换默认，差异已记录（约一半可上舍入场景），待后续视频/质量评估。

## 7. 后处理结构（sr_postprocess.py）

- 固定锐化 / 自适应锐化（前帧亮度差驱动）/ 振铃保护（向导帧后台解码 +
  Sobel 引导混合）合并单进程；内部仍是多个 OpenCV/NumPy pass。
- 线程收尾：guide producer 非 daemon，stop event + 有界队列操作 + join；
  ffmpeg stderr 尾部保留上报；异常路径有界退出。

## 8. Runtime 发布分离机制

源码更新不会自动同步 `xess-vsr.exe`（`sync_overlay` 只同步 pipeline/ 文本）。
二进制通过固定 Runtime 资产发布：

- 本轮候选 `runtime-2026.08.27-r2`
  `xess-runtime-windows-x64-2026.08.27-r2.zip`（303.32 MiB，
  SHA256 `53b59b4139931de6275aeacc1dcf451c4f78c7a179713cd293d045359c8ee063`）
- 包含新 `xess-vsr.exe`（SHA256 `be3f55aa570cb7760d3066c25acb30c38730dccc8840220cc246e51a7dafad04`）
- 干净 git archive 克隆 + 本地资产安装验证：`runtime status` compatible、
  `self_test.py --sr-only` 通过、48 帧 SR 与工作树构建逐字节一致。
- 未创建 GitHub Release / 未上传 / 未 push manifest（等发布授权）。

## 9. 被否决/未采纳实验

| 项目 | 结论 |
|---|---|
| CPU 写线程（方案 B） | 未实施：方案 A（无用信号量删除）已满足一致性；写线程仅在本轮记录为后续实验。 |
| FP16 F16C 批量转换 | 未采纳：与旧实现舍入 1 ULP 差异（见 §6）。 |
| XeSS Quality 档位映射调整 | 未做：默认映射不变（需人工观看）。 |
| GPU MV 上采样 / GPU 后处理 / QSV 直编码 | 可行性待后续轮次（见 §12）。 |

## 10. B580 结果（同会话、交错、三次取中位数）

worker 全链基准（解码→prepare→ring→xess-vsr→sr-post→SHA256 流式哈希），
不含编码器：

- 场景 A（480p 243 帧 → 720p，Fast，默认振铃保护）：
  新链中位 **20.537s** vs 旧链（封存版 2aa9855）**22.574s** ≈ **9.9%** 提速。
- 场景 B（1080×1920 300 帧 → 1440×2560，Fast，shared，振铃 0.75）确认轮
  （第三轮，同一开机会话、控制样本漂移 <0.1%）：
  新链中位 **53.965s** vs 旧链 **57.983s** ≈ **7.4%** 提速。
  此前两轮：ring 零拷贝轮 54.709 vs 57.680（5.4%）；最终构建轮受会话状态
  漂移影响（两配置前两跑同时偏慢 11–14s，漂移超 5%）标记“不确定”，已用
  确认轮取代（见 benchmarks/scenarioB_confirmed.csv）。

所有对比运行输出 SHA256 完全一致（场景 A `31ddf78c…`、场景 B `5907a789…`），
即优化保持逐字节一致。

## 11. A770 风险

A770 未实测（本机为 B580）。D3D12 路径与 SIMD 逻辑与显卡型号无关，
但 XeSS 驱动侧行为/性能模式可能不同，需真机验证。

## 12. 后续 GPU 后处理 / QSV 路线（可行性）

长期链路：FFmpeg/QSV 解码 → DIS/mask → XeSS → GPU 后处理（sharpen +
ringing guard 迁 D3D12 CS）→ oneVPL/QSV 编码。可消除 GPU 回读、RGBA→RGB、
rawvideo 管道。当前 CPU 后处理仍在关键路径，本轮只记录路线图，未替换默认。

## 13. 提交

见 `git log`（codex/sr-runtime-io-optimization）：

- `871d473` fix: drain in-flight SR GPU work on failures
- `8956ae7` fix: normalize retired flow modes to Fast DIS
- `e0ff403` fix: harden SR postprocess thread teardown
- `68d200c` test: cover SR postprocess and failure paths
- `a8b81e4` ci: run dependency-light unit tests
- `bac10b5` docs: correct SR pipeline and benchmark claims
- `bf14046` perf: reduce SR worker I/O copies and add SIMD conversions
- `7e741d9` build: prepare next Windows runtime candidate