# 离线工具箱编码器

五条路线（GPU Block、GPU DIS、AMD 光流、CPU DIS、Intel 视频接口）的统一入口均接受以下编码器，适用于超分、2× 补帧、超分后补帧。算法不因编码器选择改变，不支持的编码器会明确报错。

| 选项 | 成片 | 三条原生 GPU 路线的出口 |
| --- | --- | --- |
| 自动 / `h264_qsv` | H.264，MP4 或 MKV | 默认，GPU VPP → QSV |
| `hevc_qsv` | HEVC / H.265，MP4 或 MKV | GPU VPP → QSV |
| `libx264` | H.264，MP4 或 MKV | 最终纹理回读 RGB → CPU 编码 |
| `libx265` | HEVC / H.265，MP4 或 MKV | 最终纹理回读 RGB → CPU 编码 |
| `ffv1` | FFV1，必须 MKV | 最终纹理回读 RGB → 无损编码 |

## 使用

前端切换“编码器”；选择 FFV1 时，已有的 `.mp4` 输出名称改成 `.mkv`。ComfyUI 独立离线节点从同一能力入口读取选项，不需要额外安装 PyTorch 或替换 ComfyUI 环境。

统一命令入口为 `pipeline/offline_toolbox.py` 的 JSON 请求：

```json
{"backend":"gpu-block","mode":"sr-fg","scale":1.5,"encoder":"libx265","input":"<原片绝对路径>","output":"<成片绝对路径.mp4>"}
```

传给 `run --request <请求文件> --runtime-root <便携运行时目录> --work-root <工作目录>`。默认仍为 H.264 QSV。

旧 CPU 命令 `run_xess.py`、`run_fg.py`、`run_pipeline.py` 恢复 `--encoder` 参数。省略时保留旧默认，不更改已保存工作流；FFV1 使用流式处理和 MKV。旧 `run_fg --pipeline-backend full-gpu` 不是新入口，不能用它冒充新编码出口，非 H.264 选择会提示改用统一入口。

## 传输与画质边界

- H.264/HEVC 硬件出口保留原生 GPU 链，不新增全帧 CPU 回读。
- 软件出口不先生成有损 H.264 再转码。直接读取最终 XeSS/XeFG/锐化纹理，不改光流、深度、Mask，也不重新分析视频。
- 软件出口只有一张 D3D11 staging 纹理、一帧 RGB 打包缓冲及 256 KiB 命名管道，沿用既有有界 GPU 槽。编码器内部缓存另计；不落盘全片 raw。
- 软件模式应称“GPU 增强 + 末端 CPU 编码”，不能称全链 GPU。报告分别记录末端 RGBA 回读字节和 RGB 管道字节；QSV 两项均为零。
- FFV1 对处理后的 8-bit RGB 无损，不代表恢复原片中已丢失的细节。CPU 和原生 GPU 出口使用 `bgr0`，不额外做 YUV 色度抽样；Intel 视频接口原本输出 NV12，其 FFV1 保存这份 4:2:0 结果，不冒称原生 RGB。
- GPU 路线的 FFV1 会将编码前 RGB SHA256 与最终成片解码为 RGB 后的 SHA256 比较；哈希、帧数、PTS、音轨任一不通过，不发布成片。
- 软件编码可能比 QSV 慢，且有额外回读成本。该选择不会自动切换为 CPU DIS。
- 编码器的压缩参数并不相同：新原生出口的 QSV 使用 ICQ20，x264/x265 使用 CRF18 / medium；CPU DIS 和 Intel 接口保留各自既有参数。不同有损编码器的体积与画面会不同，“算法不变”不等于有损成片逐字节相同。

## 收尾、失败与时间戳

1. FFmpeg 子进程放入独立 Windows Job Object，父进程退出时自动回收。连接、管道写入、编码收尾均有时限，并响应取消。
2. GPU fence 完成后读取 staging。`Map(DO_NOT_WAIT)` 仍可能返回资源忙，仅对 `DXGI_ERROR_WAS_STILL_DRAWING` 做有界重试；其他错误直接失败。参见 [Microsoft Map 文档](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-map)。
3. 所有新出口关闭 B 帧，封装时用 `setts` 重建已验证的 CFR 有理数时间轴，不重编码、不补帧。避免 HEVC 初始负时间戳或 MKV 毫秒精度带来的误裁帧。
4. 音轨拷贝不决定视频终点；短音轨不能裁掉最后一帧。Intel 色彩标签重封装使用 `-copyts`，保留音频预留时间与零起点视频的关系。
5. CPU 成片用 ffprobe 实际解码计数，不再用 OpenCV 根据 MKV 总时长估算帧数。

## 构建与验证

`tools/build_offline_encoders.cmd <输出目录>` 从当前工作树重建三个原生 worker。沿用已有 SDK 依赖环境变量；不改 SDK、算法 shader 或已封存实验树。最终用户只使用预编译运行时，不需要 VS、DXC 或 oneAPI 编译器。

验证工具：

- `tools/test_offline_delivery_matrix.py --encoder-matrix`：五路线 × 三模式 × 五编码器，独立检查实际 codec、解码帧数、PTS 与音轨。
- 同一工具的 `--scale 2 --modes sr-fg`：1080p → 4K、NTSC 有理数帧率补测。
- `tools/test_terminal_encoder_failures.py`：编码器失败、活动编码管道取消，确保无假完成。
- `tests/test_offline_encoder_support.py`：选项透传、旧命令兼容、MKV 计数和时间戳回归。

AV1、NVENC、AMF 不在此次五编码器支持集合。诊断器能够探测到名字不等于整条处理链支持。B580 是当前验收设备；没有将本次结果冒称 A770、AMD 或 NVIDIA 真机验证。此次仅本地开发候选，不代表 GitHub 已发布。
