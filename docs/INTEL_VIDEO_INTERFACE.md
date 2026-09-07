# Intel 视频接口：离线调用合同 v2

后端 ID `intel-vpl-ai`。保留 `pipeline/fast_pro.py` / `run_fast_pro` 兼容入口。

```text
python pipeline/fast_pro.py sr-fg input.mp4 output.mp4 --vpl engine/vpl-ai-vpp.exe --ffmpeg engine/ffmpeg.exe --ffprobe engine/ffprobe.exe --work-dir job --out-width 1296 --out-height 720 --encoder h264_qsv
```

`mode` 支持 `sr`、`fi`（别名 `fg`）、`sr-fg`（别名 `sr-fi`）。组合模式为同一原生 VPP 会话设置 `MFX_EXTBUFF_VPP_AI_SUPER_RESOLUTION` 和 `MFX_EXTBUFF_VPP_AI_FRAME_INTERPOLATION` 两个扩展。原生入口组合名为 `sr-fi`。B580/oneVPL runtime 2.17 已实际运行8→16帧；runtime 内部先后顺序和运动分析次数不公开，相关计数为 null，不宣称可复用运动场。

SR 输出 N 帧，FI/组合输出 2N 帧（尾部 drain），帧率使用有理数；只接受零起点 CFR SDR 8-bit、偶数 crop。VFR、不连续/非零首 PTS、HDR/10bit/奇数 crop 显式失败。输入方向按 coded geometry 处理并保留 rotation/SAR/DAR元数据；实际媒体门仍会验证这些字段。直接 SR 需要 aligned height <=1440、>=1.4x；竖屏可按现有规划进行 CPU transpose→AI→transpose回转及末端缩小，实际路线进 requested/planned/applied。有损 preshrink 不自动启用。

默认 `h264_qsv`，可显式选 `hevc_qsv`、`libx264`、`libx265`、`ffv1`；软件H264/H265当前固定 fast/CRF16，FFV1要求MKV/AVI。编码器不支持时失败，不自动替换。每任务会实际预检 VPP Query/Init，检查编码器可用，完整执行结果决定任务成功。能力报告必须限定实际测试的几何/编码档。

数据经过 CPU NV12 pipe→VPL GPU surface→CPU NV12 pipe→编码；不是全GPU、不是zero-copy。没有落盘raw或中间成片、无SR后重解码，每任务仅一次最终编码。运行结果记录实际原生输入/输出帧数、NV12上传/回读字节、加载runtime模块路径、EXE/FFmpeg SHA256和命令。Intel内建模型随驱动提供，无单独模型目录。

颜色实测：B580当前驱动的AI SR/组合拒绝`mfxExtVPPVideoSignalInfo`（Query=-3），FI接受该扩展。SR仅开放原有BT.709有限范围或未标记8-bit NV12路径，显式拒绝全范围/BT.601 SR；不猜测未标记源色彩。FI按来源传递BT.709/BT.601和有限/全范围，相关样片已实跑。FFmpeg精简探针仅接受未标记/BT.709 SDR，其余复杂颜色或rotation元数据要求完整ffprobe。报告区分处理会话1与零像素Query/Init预检会话1（总会话2），避免预检成本隐身。

输出先写同目录`.intel-vpl.partial`，经过全片严格解码、分辨率/帧数/有理数PTS节奏/颜色/方向/音轨数量和所有音轨压缩载荷SHA256校验，才原子发布。不覆盖已有输出，不允许覆盖原片。取消/失败清理本任务partial并保留日志；`--timeout-seconds`默认1200。CLI失败也写`work-dir/reports/fast_pro_last.json`，schema `intel-vpl-job-v2`，status `success|failed|cancelled`。

构建 `tools/build_vpl.bat`，设置 `XESS_VPL_BUILD_DIR` 为独立产物目录，`VSDEVCMD` 可指定已有 Visual Studio 开发环境，`VPL_SRC`和`VPL_LIBDIR`指向配套头/导入库/dispatcher DLL。无驱动和用户Python升级。便携包应包含对应libvpl许可，并把Intel显卡驱动列为外部依赖；不可从系统DriverStore私自打包驱动DLL。
