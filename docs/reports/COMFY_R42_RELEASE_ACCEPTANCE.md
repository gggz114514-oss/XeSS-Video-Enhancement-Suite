# ComfyUI R4.2 三路 GPU 兼容入口验收

日期：2026-09-08。Source 1.4.2；Runtime 2026.09.08-r4.2。
这是 R4.1 的增量补丁，不包含实时工具箱、跨厂商推理实验或其他算法升级。

## 改动范围

- GPU Block、GPU DIS、AMD 光流共用「Arc A 系列兼容模式」，默认关闭，覆盖 SR / FG / SR→FG 三节点。
- CPU DIS、Intel 视频接口不使用该 NV12→D3D12 共享入口，不显示此开关。
- 不改变光流、深度、Mask、倍率、编码器和画质功能开关；旧工作流缺字段仍按关闭处理。
- GPU DIS 增补与旧 NV12 输入相同的整数 RGB→灰度适配；不将矩阵浮点亮度冒充其原灰度定义。AMD 光流继续消费公共 RGBA。
- 相对 R4.1，运行库仅替换两个编译引擎、新增一个编译着色器。GPU Block 引擎和其他 4448 个文件逐字节保留。

## 真实成片回归

维护者机器为 B580 / 驱动 32.0.101.8974。输入为问题素材派生的零起点 PTS、864×480、24fps、48 帧、有音轨、BT.709 limited SDR 视频。

| 验证 | 结果 |
| --- | --- |
| GPU DIS / AMD 光流 × SR/FG/SR→FG × QSV/x264 × 开关关/开 | 24/24，通过 |
| 上述每个开关配对解码 RGB SHA256 | 全部一致 |
| 原公开引擎 vs 新默认，两个后端 × 三模式 | 6/6，解码 RGB SHA256 一致 |
| 最终 ZIP 实际安装、完整 4451 文件哈希、重复安装幂等 | 通过 |
| 从该安装运行三路 SR→FG，五帧融合+锐化+抗竖纹全开，兼容关/开 | 6/6，每个后端的配对 RGB SHA256 一致 |
| 帧数、音频、完整解码 | SR=48；FG/SR→FG=95；均保留音频并完整解码 |

本轮共 36 次真实媒体运行。节点调用只替代 ComfyUI 返回视频对象的包装；参数处理、controller、编译引擎、编码、封装、最终验证均真实执行。不是模拟原生后端。

旧引擎对照视图保持原 R4.1 全部文件哈希，仅附加新 controller 必查、旧引擎不会加载的 RGBA→灰度 shader；未修改原安装目录。R4.1 已有的 GPU Block 12 组媒体回归及候选长跑见其独立验收报告，不能套用为新增两路的长跑结论。

## 回归与打包

177 项 Python 单测（ResourceWarning 作为错误）、节点 JS 行为测试、compileall、仓库及资产校验、diff 空白校验。
新增开关保持原 widget 顺序；GPU DIS / AMD 的开关可见性及 CPU / Intel 的隐藏重置均有测试。

ZIP：`xess-comfy-runtime-windows-x64-2026.09.08-r4.2.zip`

- 大小：369081205 字节；4451 文件。
- SHA256：`bc8e9616dcdc9ab3cd1d572443bedd5a3aaf23551c4d693e3d453b6ba4d332dc`。
- GPU DIS EXE：`126a8c524a002ab2175cfd01fb230281a1655c43cda8325e11ca03897298b61a`。
- AMD 光流 EXE：`0d7a26586e15f8f1e328bd6e88dff57aa0bc98e2f1a6dc05cbee99197e2c58eb`。
- RGBA→灰度 shader：`0a3baf87b81a609871c8a2f817292f8c470c5930167c0c78b1abd18685879139`。

私有 GPU DIS / GPU Block 源码、着色器源码、调试源码块、PDB、bundle、开发历史不进入公开提交或资产。
R4.1 Release 及其资产保留，不覆盖。

## 验证边界与失败记录

- A770 用户反馈对应原兼容入口；本轮新增 GPU DIS / AMD 光流只在 B580 做维护者回归，未冒充 A770 本机或全部驱动实测。两路仍保留实验标记。
- 48 帧短回归不是高分辨率性能报告或千帧稳定性证明。本补丁不承诺所有 A 系列的速度，不放宽 SDR、范围、矩阵及输入格式限制。
- 第一轮新 worker 被原“仅 GPU Block”保护条件拒绝；第二轮被 DIS 的 NV12-only 输入校验拒绝。已修复公共入口及整数灰度适配后重编，最终矩阵通过；失败日志保留。
- 旧版对照第一次被新 controller 的新增依赖检查拒绝，随后改用上述独立旧引擎视图；没有删除生产依赖校验。
- 草稿状态下的发布状态断言曾失败（not-published）；正式发布清单阶段重跑全部测试。未删除或放宽断言。

## 本地证据索引

- `work/check-allroutes-v3/results.json`：24 新运行 + 6 原版对照。
- `work/check-allroutes-v3.log`、`work/check-allroutes-final.log`：首轮及修正测试驱动日志。
- `work/check-r42-installed/results.json`：最终 ZIP 安装与六组画质功能回归。
- `work/package-audit-r42.json`：包级差异、逐文件哈希及源码审计。
- `work/unit-r42-final.log`、`work/validate-r42.log`：最终自动化验证。
- `work/remote-asset-verification-r42.json`：发布前上传资产的流式下载 SHA256 核验。

## 更新

关闭 ComfyUI，在节点目录 `git pull --ff-only`；重启、刷新浏览器。默认安装会按清单下载校验匹配运行库，无需手工覆盖引擎。
自行设置 `XESS_RUNTIME_ROOT` 会跳过自动部署，须取消或指向匹配 R4.2 的完整运行库。
出现整幅绿/紫色条带时开启兼容模式；原画面正常时保持关闭。
