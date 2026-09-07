# 离线编码器补齐验收

日期：2026-09-07。分支：`codex/r4-offline-delivery-20260907`。本轮基线：`fa86f64`。

结论：五条路线的编码出口已补齐并通过本地 B580 实测。**这是编码功能验收，不是整个工具箱的发布验收；未 push、未发布运行时。**

## 支持范围

GPU Block、CPU DIS、Intel 视频接口、GPU DIS、AMD 光流均支持 H.264 QSV、HEVC QSV、x264、x265、FFV1，覆盖 SR / FG 2× / SR+FG。前端与独立 ComfyUI 节点共用能力表。

默认仍为 H.264 QSV。软件编码直接接收最终 RGB，不先压成 H.264 再转码，不切换光流、不重新分析，不落盘全片 raw。只有软件编码出口发生最终纹理回读，不能将此模式称为“全链 GPU”。FFV1 必须使用 MKV。

使用与边界详见 [离线编码器](../OFFLINE_ENCODERS.md)。本轮没有更改用户已安装的 ComfyUI、OBS 或驱动。

## 真实验收结果

下面路径均相对于项目工作产物根 `work/r4-offline-delivery-20260907/`，不随源码入库。

| 验证 | 结果 | 原始证据 |
| --- | --- | --- |
| 五路线 × 三模式 × 五编码器，8 帧输入 | 75/75 | `encoder-accepted-75/results.json` |
| 五路线 × 五编码器，1080p→4K + 59.94→119.88 fps，8 帧输入 | 25/25 | `encoder-accepted-4k/results.json` |
| 三条原生 GPU 路线，末端锐化 + FFV1 | 3/3，RGB 编码前后哈希一致 | `encoder-post-accepted/results.json` |
| GPU Block、CPU DIS 全片 FFV1，243 帧输入 | 2/2，均输出 485 帧并保留音轨 | `encoder-full-accepted/results.json` |
| 编码器输出失败、活动管道取消 | 2/2，无假完成、无最终输出 | `encoder-failures-accepted/results.json` |
| 相关 Python 回归 | 63/63，ResourceWarning 作为错误 | `encoder-final-unit.log` |
| 三原生 worker 重编 | 通过 | `encoder-build-accepted/` |
| WPF 编译、JS 语法、compileall、validate_repo | 通过 | `encoder-ui-build/` 与验证日志 |

小矩阵输入为重新编码的运输测试素材，不是画质评价真值。独立核查实际 codec、完整解码帧数和音轨；原生路线另检查 CFR PTS、封装前后像素哈希、GPU/CPU 边界计数，FFV1 再比较原生 RGB 哈希。Intel FI 保持 2N，XeFG 保持 2N−1，没有为了对齐计数伪造末帧。

全片 GPU Block 的 FFV1 额外通过编码前后 RGB SHA256 一致；CPU DIS 本轮核查的是 FFV1/bgr0、485 个实际解码帧和音轨，没有新增编码前 RGB 哈希钩子，不将两者说成同一强度证明。以上是编码功能测试，不主张跨路线画质相同、速度比较成立，或 30 分钟稳定性已通过。

## 发现并修正的真实问题

1. 原生 GPU worker 原来只有 H.264 QSV，现新增显式 HEVC 与有界最终 RGB 软件编码出口。
2. CPU FFV1 的 MKV 帧数被 OpenCV 根据含音频尾巴的容器时长高估；改为 ffprobe 实际解码计数。FFV1 使用 `bgr0` 保存 RGB。
3. HEVC 裸流初始时间戳导致前两帧被裁；MKV 毫秒时间基也会污染 NTSC PTS。关闭新出口 B 帧后，以压缩包 `setts` 重建已验证的 CFR 有理数网格，不重新编码。
4. Intel FFV1 色彩标签重封装将负音频预留时间折算成视频 +43ms；增加 `-copyts` 保持视频零起点。
5. D3D11 查询完成后，非阻塞 Map 偶尔仍返回 `DXGI_ERROR_WAS_STILL_DRAWING`；仅对该状态做 15 秒内重试。其他错误不吞掉。最后全片测试记录 1 次忙重试并完整输出，验证这不是只存在于桩测试的路径。
6. FFmpeg 子进程加入原生进程自己的 Job Object；连接、写入和收尾有超时。异常包含中文摘要与 FFmpeg 原始错误，不把 EOF 伪装成成功。
7. UI 能力列表和编码探测异步到达时，选择框可能只保留 H.264；两者到达后都刷新，未就绪的已选项明确禁用，不能偷偷切回自动。FFV1 自动将已有 `.mp4` 名称改为 `.mkv`。

早期失败证据保留在 `encoder-native-smoke/`、`encoder-final-75/`、`encoder-4k-ntsc/`，分别对应时间戳、Map 瞬时资源忙、Intel FFV1 PTS 问题。它们不是最终通过数据。最终三个 EXE 一起重建后重新跑完上述 100 项矩阵与补测。

## 源码、二进制与运行时

最终编译产物与 `candidate/runtime/bin/` 对应文件 SHA256 一致：

| 文件 | SHA256 |
| --- | --- |
| `gpu-block-native.exe` | `bb64ecf00bd55db73815ed321017e6e7196ed24bfa1d33a48ad61dc63f517912` |
| `gpu-dis-native.exe` | `19bb0d9d165c000d915c0bda650603cd3cee0b5bda795c35941b54714a29fa83` |
| `amd-of-native.exe` | `8689d4c64315779689ce61dd19325e61d0248252328531a236784385859051d9` |

`tools/audit_offline_encoders.py` 自动校验矩阵覆盖、实际 codec/帧数/音轨、FFV1 哈希、三 EXE 一致性并生成 `ENCODER_ACCEPTANCE.json`。`--update-local-manifest` 只更新候选运行时本地清单，不冒充 GitHub 已发布资产。SDK DLL 和算法 shader 未改；原 EXE 备份位于 `encoder-baseline-bin/`。

日志中的 `encode_settings.onevpl_pipeline_used=false` 表示软件出口；既有 `target_usage/gop` 字段是保留的硬件配置，不应作为软件编码器实际参数使用。软件 x264/x265 的实际命令为 CRF18/medium/B0，其 GOP 沿用编码器默认；FFV1 为 level3/bgr0。

## 整体项目尚未关闭的验收项

本轮相关回归 63/63 通过，但**全库不是全绿**：发现机制运行 370 项，5 项跳过，12 项失败、0 个错误。编码补齐前 360 项、14 项失败；本轮修复其中 2 条过时编码断言，剩余 12 条属于此前五路线接入后的契约/清单对齐债务。

不能直接删掉这些测试让统计变绿。主要涉及旧 UI 仍要求仅三路线或组合仅 CPU、实时入口尚未启用、registry 身份与旧 runtime manifest 未对齐，以及旧 worker 容器契约。完整测试 ID 和原始摘要见 `unit-encoder-full.json`，上轮基线见 `unit-report-before.json`。下一步工具箱发布验收应逐条按新产品契约修正并做独立验证。

AV1/NVENC/AMF 未接入此次产品出口；B580 以外显卡未实测。软件编码可能比 QSV 慢且 FFV1 成片较大。未跑长时压力矩阵、未制作新发布包；仓库运行时清单保持 `not-published`。
