# SEA-RAFT XPU 实验分支状态

**分支：`experiment/sea-raft-xpu`**  
**状态：实验归档，不建议作为主线或日常用户版本**

## 结论

本分支完整保留了 SEA-RAFT + Intel XPU 融合相关核实验，包括 StreamingCorrBlock、DPC++/SYCL fused correlation、SLM staged 路径、XPU dispatcher 集成，以及 SR/FG 的 SEA-RAFT 参数透传。

经过 B580 真机、SR、FG、720p/2K 和 30/60fps 场景验证，结论是：**SEA-RAFT 作为本项目的前置运动矢量路线不适合进入主线**。

它没有带来可稳定复现的肉眼级画质提升，却带来了约 2.8–3 倍计算时间、较低的可靠性、更高的显存压力，以及 PyTorch XPU、模型和 SYCL 扩展等额外运行时依赖。主线建议使用 DIS，并把资源可靠性判断、遮挡掩码和局部回退放在后续优化重点。

这不是对 SEA-RAFT 光流算法本身的否定，而是对“从成品视频猜测 XeSS 所需引擎运动数据”这条工程路线的否定。

## 实测对比

测试均使用同一台 Intel Arc B580、同一输入和相同的输出编码参数。时间为完整短片墙钟时间；可靠性是前置运动分析报告的统计值，不是画质评分。

| 场景 | DIS | SEA-RAFT | 结论 |
|---|---:|---:|---|
| FG：60→120fps | 157.9秒 | 457.8秒 | SEA-RAFT 慢约 2.9 倍，画质无明显优势 |
| FG：30→60fps | 80.5秒 | 224.7秒 | SEA-RAFT 慢约 2.8 倍；可靠性均值 70.9%，最低 15.8% |
| SR：1080×1920→1440×2560 | 73.0秒 | 216.4秒 | SEA-RAFT 慢约 3.0 倍；可靠性均值 83.2%，DIS 为 96.0% |

赛车场景中两种路线都会在高速运动、运动模糊、烟雾和遮挡边缘产生伪影；SEA-RAFT 的伪影形态不同，但没有解决根本问题。

## 本分支包含的实验组件

- `pipeline/sea_raft_core/corr.py`：流式相关体和 XPU 相关路径选择；
- `pipeline/sea_raft_core/xpu_corr/`：DPC++/SYCL fused correlation、SLM staged 内核、PyTorch XPU 注册与加载；
- `pipeline/sea_raft_core/raft.py`：高分辨率相关路径切换；
- `pipeline/motion_core.py`：SEA-RAFT、flow-scale 和 XPU 运行时处理；
- `pipeline/run_xess.py`、`pipeline/run_fg.py`：SEA-RAFT 参数透传；
- `tests/test_xpu_corr_op.py`、`tests/test_streaming_corr.py` 等：数值、边界和稳定性测试；
- `tools/bench_xpu_corr.py`：相关核基准工具；
- `docs/XPU_CORR_SLM_DEV_NOTES.md`：实现过程和已知坑记录。

## 如何运行实验

本分支不是开箱即用的发行包。需要 Windows、Intel Arc、oneAPI DPC++、Intel PyTorch XPU、`safetensors`、SEA-RAFT 模型和 OpenVINO。先在 `pipeline/sea_raft_core/xpu_corr/build.cmd` 所要求的环境中构建扩展，再运行：

```text
run_xess.py input.mp4 1.333333 --preset quality --flow-mode sea-raft --flow-resolution auto720
run_fg.py input.mp4 --preset quality --flow-mode sea-raft --flow-resolution auto720
```

相关核可用环境变量控制：

```text
XESS_XPU_CORR=auto       # 默认：可用时使用融合核，失败回退
XESS_XPU_CORR=off        # 强制 StreamingCorrBlock
XESS_XPU_CORR=required   # 融合核不可用时直接失败，用于实验验证
```

当前发布的实验构建在 B580 上验证过；A770 仅完成兼容构建，未完成同等真机回归。`.pyd` 不进入 Git，必须在目标机器重新构建。

## 主线处理建议

不要把本分支合并回 `main`，也不要把 SEA-RAFT 放入普通用户预设。主线应保留 DIS 极速档，并用 DIS 一致性、AI 深度、遮挡掩码和局部回退提高稳定性。这个分支保留的目的，是让后续研究者可以复现实验、检查融合核实现，并在出现新的硬件或模型条件时继续比较。

