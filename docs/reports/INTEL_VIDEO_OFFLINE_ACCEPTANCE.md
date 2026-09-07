# Intel 视频接口离线收口验收

分类：`READY_FOR_INTEGRATION`，限定本机Intel Arc B580、驱动32.0.101.8974与下列实测范围。其他显卡、真实OBS/WGC、用户视觉批准均未完成。

原生算法头2ec4e95，最终容器/PTS验证头82629f5。实现提交依次de8744d、820830a、2ec4e95、db896ec、82629f5；本文为后续报告提交，不要求报告记录自身未来commit。

## 可集成接口

`pipeline/fast_pro.py:run_fast_pro`接受SR、FI/FG、SR+FI组合。组合用同一oneVPL VPP处理会话的两个官方AI扩展，Query/Init实测成功；另有一次零像素预检会话，总会话2。没有XeSS回退、SR后成片重解码或中间raw文件，末端只编码一次。

NV12有效crop数据经过CPU pipe上传及回读；实际字节来自原生计数，驱动内部总线/资源copy不可观测。Intel内部SR/FI顺序与运动分析/消费者次数不公开，JSON使用null，不声称实现自定义共享motion。详细参数和能力限制见[接口合同](../INTEL_VIDEO_INTERFACE.md)。

## 已完成门

| 项目 | 本轮结论 |
|---|---|
| 横向人像与竖向赛车 | 8帧、48帧，SR/FI/组合全部通过 |
| 完整人像243帧 | SR243、FI486、组合486，通过全片解码/PTS/音轨载荷SHA256 |
| 完整赛车300帧 | SR300、FI600、组合600，通过全片解码/PTS/音轨载荷SHA256 |
| 高分辨率 | 1080/1440/2160输出、竖屏2160×3840 SR/组合及FI8帧实际通过；未据此声称4K完整长片视觉通过 |
| 编码器 | h264_qsv/hevc_qsv/libx264/libx265/FFV1实际短链通过；FFV1长音轨尾部额外样片通过 |
| 首尾/旋转 | 1/2帧三模式、rotation90元数据FI通过 |
| 故障 | 奇数crop、VFR、10bit、低磁盘、原生半帧、取消、下游关闭、超时均明确拒绝；无partial和错误最终片 |
| 单测 | 35项通过，0失败、0skip |
| 稳定性 | 600.093秒组合编码到NUL，14832输入/29662输出后主动取消，未落盘raw |
| 身份差分 | SR与bilinear、FI与重复帧基线均不同；不视为正式模型认证或优质证明 |

600秒门按计划主动取消，未EOF drain；其中2N-2不取代正常完整成片的2N语义。CPU RSS/GPU busy/VRAM未取得，不宣称内存泄漏门通过。

## 固定端点计时

864×480@24的人像48帧，SR1296×720，h264_qsv默认参数，3轮SR/FI/组合交错。含预检、哈希和最终完整验证的墙钟中位：2.8676/3.9218/4.5271秒；处理链墙钟中位：1.7012/2.3590/2.8958秒。后者包含CPU I/O和等待，不能称纯AI GPU时间，也不用于跨路线加速结论。

## 明确限制

零首PTS CFR、8-bit SDR与偶数NV12 crop；源/输出PTS容差按真实time_base。VFR/不连续/非零起点、HDR/高bit拒绝。直接SR输入对齐高<=1440、倍率>=1.4；旋转可覆盖目标时如实记录，修复了旧5%余量误拒2x问题，自动有损preshrink禁用。

当前驱动SR/组合拒绝`VPPVideoSignalInfo`（Query=-3）；保留原有BT709有限/未标记NV12 SR路径，全范围/BT601 SR拒绝。FI可传该扩展，Full709/limited601实际成功。未标记源不猜色彩。XeSS sharp/five-frame/anti-stripe/depth/motion-provider不适用于此入口。

## 本地产物与依赖

本轮产物根相对工具ROOT为`work/r4-route-completion-20260907/D`。`FINAL.md`为中文完整报告，`VIDEO_INDEX.md`链接原片及`videos/final-{face|racing}-{sr|fi|sr-fg}.mp4`六个完整最新成片。`ACCEPTANCE.json`、`BACKEND_CAPABILITIES.json`、`DEPENDENCIES.json`、`INPUTS.json`、`FINAL_REVALIDATION.json`、`PERFORMANCE.json`、`FAULTS.json`、`STABILITY.json`与租约日志保留可复核证据；66个容器案例64成功、2个预期颜色拒绝，无意外失败。

实际runtime API2.17，oneVPL头源码674d015bcb294bc39fa276e99a652ea045423e82。本轮`build-color/vpl-ai-vpp.exe`的SHA256为3841df789e49fea92295c52963a55f546b4d17ab391989d47d56aa43fbe996b9。配套libvpl/真正加载驱动DLL的版本、SHA与许可索引保留在依赖JSON。oneVPL dispatcher/header为MIT；实际FFmpeg7.1编码器报告GPLv3-or-later；完整共享ffprobe9.0依赖不可只复制一个exe。未复制系统驱动或模型，没有升级用户环境、修改前端或发布远端。
