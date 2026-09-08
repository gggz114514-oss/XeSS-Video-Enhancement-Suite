# ComfyUI R4.1 小版本验收

版本：Source 1.4.1 / Runtime 2026.09.08-r4.1。公开源码基于已清理历史的 `bd75bed`；未合并私有开发分支，不重新引入 GPU Block / GPU DIS 源码或历史。

## 发布范围

新增可选布尔字段 `arc_a_compat`，三个节点的 required 字段及原有顺序不变。前端仅 GPU Block 显示；服务端也校验，不向其他后端传入兼容参数。默认不追加原生参数，开启时追加 `--decode-share rgba-d3d11`。缺少着色器时在计划阶段明确拒绝。

运行库 4450 个文件。相对已发布 R4，只修改 `bin/gpu-block-native.exe`，新增 `shaders/common/native_rgba_share.cso` 和 `shaders/common/native_rgba_ingress.dxil`；其余包内文件逐项 SHA256 相同。

- 运行库 ZIP：369,066,597 字节。
- ZIP SHA256：`fe35350cdc56f49763f14ac452e7d0221681f1482e50dd7ea52875a6cb8591cd`。
- GPU Block worker SHA256：`d5838aeba4285c28656faa11713c9b56eca6be2e6195ca7670f7b1fbe57ae304`。
- 新增着色器审计未发现 ILDB / SRCI / PDBI 源码或调试块；包不新增源文件、PDB、补丁或 Git bundle。

## 本次正式包回归（B580）

| 项目 | 结果 |
| --- | --- |
| 实际 ZIP 安装及完整文件哈希校验 | 通过；再次 ensure 不重复安装 |
| SR / FG / SR→FG × QSV / x264 × 兼容开 / 关 | 12/12 通过 |
| 输入 48 帧，输出帧数、全片解码、音轨 | SR 48；FG/组合 95；全部保留音轨 |
| 每组兼容开 / 关解码 RGB 哈希 | 6/6 完全一致 |
| 原发布 R4 worker 对新版默认路径，三个模式 QSV | 3/3 解码 RGB 哈希一致 |
| 原生报告确认实际选中路径 | `nv12` / `rgba-d3d11` 与请求一致 |
| Python 单测（含升级异常测试及新增 8 项开关测试） | 175/175 通过 |
| JS widget 可见性与状态回归 | 通过；不重建节点，其他后端清除隐藏开关 |
| 仓库、源码边界、工作流、清单及 ZIP 哈希校验 | 通过 |

媒体测试实际调用公开节点方法 → 请求 JSON → offline_toolbox → 已安装发布包 → 成片校验。仅返回 VIDEO 的 ComfyUI 对象包装器用测试替身，不冒充本轮运行了完整 ComfyUI UI 服务。前端交互测试为 JS 状态测试。

本轮第一次短素材截取采用 stream-copy 导致非零 PTS，被现有入口正确拒绝，未进入 GPU。改用明确零起点的 48 帧测试素材后重跑全部矩阵，没有放宽产品 PTS 校验或改动用户原片。

## 复用的兼容候选证据

与发布包相同 SHA256 的 worker 此前完成 311 帧完整 SR / 621 帧完整 SR→FG 的原/兼容成片哈希一致、1000 帧、取消/下游关闭、中文路径及色彩合成测试。这些是候选阶段的独立测试，不计入本轮 12 次短媒体回归。

## 诚实边界

A770 用户反馈兼容路径有效；维护者本机没有 A770，不能宣称本轮直接完成 A770/全 A 系列验证。B580 回归不能证明所有显卡/驱动同一根因。GPU DIS、AMD 光流等其他后端不新增兼容路径；跨厂商实验及实时产品不随本次发布。

发布流程采用草稿上传、回读资产并核对 SHA256，再正式发布 Release 并更新 main；不把诊断 ZIP 当作正式运行库。旧 `.runtime/versions` 与用户的 ComfyUI/Python 环境保留。
