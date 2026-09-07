# 离线 GPU 五帧融合、抗竖纹与界面局部更新验收

日期：2026-09-07。基线：本地离线交付分支 d7000b8。仅本地更新，未发布 GitHub。

此文保留首轮验收历史。后续用户将 AMD 光流纳入增强范围，最新能力与验收见 [AMD_OFFLINE_EFFECTS_ACCEPTANCE.md](AMD_OFFLINE_EFFECTS_ACCEPTANCE.md)；下表 AMD“不开放”不再是当前产品限制。

## 交付范围

| 路线 | 五帧融合 | 抗竖纹 | 工具箱显示 |
| --- | --- | --- | --- |
| GPU Block | SR、SR→FG | SR、SR→FG | 支持时显示 |
| GPU DIS（实验） | SR、SR→FG | SR、SR→FG | 支持时显示 |
| CPU DIS | SR（原功能） | SR、SR→FG（原功能） | 保留 |
| Intel 视频接口 | 不开放 | 不开放 | 隐藏 |
| AMD 光流（实验） | 不开放 | 不开放 | 隐藏 |

两个开关默认关闭。独立 FG 没有 SR 阶段，不显示这些 SR 处理开关；末尾锐化不受此限制。
ComfyUI R4 节点继续共用能力接口，支持的 GPU 路线可以传入两个开关；没有修改 ComfyUI Python、旧插件或启动器。

## 算法与资源边界

1. 新模块 `native_sr_effects.h` 与 `native_sr_effects.hlsl` 接在 XeSS 输出之后、FG 输入之前。处理后的同一张 SR 纹理同时交给 FG 和最终编码器；锐化仍在最终输出一次。
2. 抗竖纹移植当前 CPU 方案：原片 bicubic（A=-0.75）放大为引导图，RGB 浮点灰度，3×3 Sobel-x，绝对值 /8，阈值 (x-0.5)/4，sigma=2.5、21 taps 分离高斯，强度 0.90 混合。边界 REFLECT_101，明确量化到 RGB8。没有重新启用旧版近似 k5 护边。
3. GPU 五帧是**因果窗口：当前＋前四帧**，沿用旧 GPU 方案组织方式。通过已有 current→previous 光流连续投影，进行亮度误差、大运动、越界拒绝和颜色残差限幅。没有额外重算光流，不等待未来帧。它与 CPU 的前后各两帧预融合**不是同一算法实现**，不宣称两种输出逐位等价。
4. 历史随首帧/场景切换/外部 reset 清零，最多四张旧帧。只有开启五帧才将槽数提高至至少 5。当前共同接口里 legacy `forward_flow` 才是 current→previous，不能按变量名字误用 `backward_flow`。
5. 每槽、每个处理 pass 有独立描述符区域；引导图与 scratch 按槽分配，显式 UAV→SRV/COPY 转换。复用持久源纹理，不读回历史像素。处理完有一次 D3D12 内部 GPU copy，不宣传 zero-copy。两项关闭不分配 scratch、不 dispatch。

## 数值复验

`tools/test_gpu_sr_effects.py`：两条路线各 old/off/guard/five/both 五种配置，每种 7 帧；显式诊断回读不计作生产链的传输。

| 对照（每条路线逐帧） | GPU Block 最大误差 | GPU DIS 最大误差 |
| --- | ---: | ---: |
| 新 off 对已部署旧 EXE | 0，逐字节一致 | 0，逐字节一致 |
| GPU 抗竖纹对 CPU OpenCV 参考 | 1 色阶 | 1 色阶 |
| GPU 因果五帧对独立 NumPy 参考 | 1 色阶 | 1 色阶 |
| 两项同时开启对 CPU 参考 | 1 色阶 | 1 色阶 |

抗竖纹最差帧平均绝对误差分别 0.02364/0.02341 色阶。开关确实改变图像，7 帧累计历史样本 18，测试断言均通过。
这是实现正确性核查，不是五帧融合在所有视频上提升画质的证明；动态视频可能发生副作用，仍保持用户显式选择。

## 真实视频矩阵

独立 ffprobe 解码计数、实际 codec、PTS、音轨检查：

- 720p 冒烟：2 路线 × SR/SR→FG，12→12/23 帧，4/4。
- 人脸全片：2 路线 × SR/SR→FG，243→243/485 帧，4/4。两项＋末尾锐化全部开启，H.264 QSV，音轨保留。
- 1080p→4K：2 路线 × SR/SR→FG × H.264 QSV/FFV1，12→12/23 帧，8/8。FFV1 编码前后 RGB 哈希通过；只有软件编码末端发生显式回读。
- QSV 生产测试的完整像素 CPU 上传/回读均为 0，新效果模块本身没有 CPU 像素传输。

全片单次墙钟（含处理入口，不是交错性能基准）：Block SR 6.54s、SR→FG 10.24s；GPU DIS SR 12.16s、SR→FG 13.27s。不据此宣传相对提速。

早期失败保留：`effects-reference` 因探针漏传必需深度模型而拒绝，补充参数后在 v2 通过；`effects-4k` 的测试素材因为 `-shortest` 被较短音轨裁成 11 帧。已修复素材生成器并在启动矩阵前核对实际输入帧数，`effects-4k-fixed` 是最终 12 帧通过数据。未掩盖失败或改成宽松帧数门。

## 界面

- 算法/模式/倍率切换只更新现有控件的 value、hidden、disabled，不替换整块 `page-offline.innerHTML`，不重绑整页事件。
- 自定义倍率输入保持同一个 DOM；原片路径、输出路径、焦点、滚动和任务进度不因切换重建而消失。
- 不支持的效果隐藏并清除非法勾选。`[hidden]` 明确高于 flex/grid 样式。
- 异步计划不先清空已有计划卡片；旧请求的迟到响应不能覆盖新选择，启动处理仍重新校验计划。
- Node 行为测试执行实际 app.js，覆盖控件隐藏、两条 GPU 路线可见、自定义倍率、整页未替换、导航保存路径和异步编码列表。电脑操作验证新窗口已显示 GPU Block 两个可用开关；检测到用户操作后不继续抢占窗口。

## 构建、部署、回归

- 构建脚本：`tools/build_offline_gpu_effects.cmd`（显式 E 盘输出参数，使用已有 MSVC/OpenCL/OpenVINO/SDK，不安装依赖）。
- GPU Block EXE SHA256：`2bdb2d6aec0c296e650dceaf464ea3b236d4a8af78121c52361fe8e017d1d20f`
- GPU DIS EXE SHA256：`048513eb1ee6e86edc5e90c4324450192f29eebfa89676939d75fb7793b7b88d`
- shader SHA256：`b1152483b5b38e1ba3231ea1511c4cdb59e9df91168ef1b0a245775d10028650`
- Python 定向回归 50/50；实际 JS 行为测试通过；UI 校验器 20/20；compileall 与 validate_repo 通过。没有把未复跑的全仓测试说成全绿。
- 本地更新脚本：`tools/update_offline_gpu_effects.ps1`。修改前逐文件备份，校验复制哈希并更新本地 runtime 文件清单。AMD、Intel 原生 EXE、CPU EXE、SDK DLL、Python 均未修改。
- 工作目录：`work/r4-offline-delivery-20260907/` 下的 `effects-build`、`effects-before`、`effects-smoke`、`effects-reference-v2`、`effects-full`、`effects-4k-fixed`；每组包含命令/日志/原生 JSON/结果 JSON。
- 已清理本轮 840 个可复现诊断 raw 文件，共 1,700,833,120 字节。保留测试源码、结果 JSON、日志、完整成片和更新前二进制备份。未删除其他任务资源。

## 边界

B580 本地实测；A770、AMD/NVIDIA 显卡未实测，不扩大硬件支持承诺。没有新增 30 分钟稳定性证明。五帧与抗竖纹会增加 GPU 开销，关闭即保留原链；不宣称对动态素材必然改善。最终主观画质仍由用户观看全片确认。
