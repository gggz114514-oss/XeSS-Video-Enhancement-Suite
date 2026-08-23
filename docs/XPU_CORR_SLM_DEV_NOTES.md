# SEA-RAFT XPU 融合相关核 SLM 版本：开发问题记录

本文档记录 `pipeline/sea_raft_core/xpu_corr/` SLM（Shared Local Memory）staged
gather-correlate 内核开发过程中实际遇到的问题与修复方式，供后续维护者避坑。
所有问题均为本次开发中真实发生并验证过的，不是理论风险清单。

日期：2026-08-23　硬件：Intel Arc B580（bmg）　工具链：DPC++ 2026.1 icx +
PyTorch cpp_extension SyclExtension + Ninja

## 1. 原 SLM 源码从未编译过，存在三类编译错误

接手时的 `gather_correlate.sycl` 是一版从未通过编译的草稿：

1. **`group_local_memory_for_overwrite(cgh)` 误用**。该 API 是设备端语法，
   接受 `sycl::group`；对 host 端 `sycl::handler` 没有对应重载，报
   "no matching function"。修复：改用 SYCL 2020 标准 host 端写法
   `sycl::local_accessor<float,1> slab(range, cgh)`，内核里
   `slab.get_pointer()` 取指针。
2. **launcher 重写时丢了流获取**。`c10::xpu::getCurrentXPUStream()` 与
   `sycl::queue& queue = stream.queue()` 两行缺失，导致 submit 处报
   `use of undeclared identifier 'queue'`。任何基于现有 launcher 改写的
   内核都要保留这两行。
3. **无效占位**：`sycl::any_of` 占位代码无法编译；
   `sycl::is_finite(static_cast<float>(span))` 对整数 span 没有意义。
   全部移除，边界检查改为显式的 `std::isfinite` + 数值范围钳制。

## 2. loader.py 的库目录覆盖从未生效（最严重的隐性 bug）

原实现：

```python
for directory in _candidate_dirs():   # [自身目录, XESS_XPU_CORR_LIB_DIR]
    sys.path.insert(0, directory)
```

两次 `insert(0)` 之后，自身目录反而排在最前，`XESS_XPU_CORR_LIB_DIR`
指向的目录**永远轮不到**。后果：第一轮"staged 路径测试"实际上静默加载了
旧基线 .pyd，测试结果全部失真。修复为 `for directory in reversed(...)`。

教训：凡是提供"加载覆盖目录"这类调试开关，必须有对应的自动化测试断言
"覆盖确实生效"（例如校验模块 `__file__`），否则覆盖失败是无声的。
本次正是靠 pybind 新增接口在旧二进制上抛 `AttributeError` 才暴露出来。

## 3. 无 tensor 参数的算子无法走 PyTorch dispatcher

统计计数器的读取/清零最初按普通 op 注册进 `TORCH_LIBRARY`，调用时报
`NotImplementedError`：schema 里没有 tensor 参数时 dispatcher 无法把它
路由到 XPU-only 注册。修复：这两个函数改为直接经 `PYBIND11_MODULE`
暴露（`corr_stats()` / `reset_corr_stats()`），并在注释里说明原因。
带 tensor 参数的正式算子仍走 `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL
(xess_xpu, XPU, ...)`。

## 4. site-packages 的 `tests` 包遮蔽本地测试目录

仓库根目录的 `tests/` 是 namespace package，而环境 site-packages 里存在
一个常规包 `tests`。Python 的常规包优先级高于 namespace package，因此
`python -m unittest tests.test_xpu_corr_op` 会 ModuleNotFoundError 或
加载到错误目标。必须使用：

```
python -m unittest discover -s tests -p "test_xpu_corr_op.py"
```

## 5. Windows cmd 的 `%VAR%` 在复合行解析期展开

`set W=... & mkdir %W%\sub` 这类复合命令中 `%W%` 用的是解析前的旧值，
导致目录建到别处且不报错。本次所有临时文件要求放在 E 盘工作目录，此类
命令一律改用 PowerShell 显式绝对路径执行。

## 6. 环境里的 git 不在 PATH

本机可用的 git 是 ComfyUI 整合包自带版本，位于
`<ComfyUI 整合包根目录>\git\cmd\git.exe`，
需以完整路径 + `git -C <worktree>` 方式调用。

## 7. 默认 AOT 架构列表不含 dg2

cpp_extension 默认 `TORCH_XPU_ARCH_LIST` 推导出的目标是
`mtl,mtl-h,bmg,arl-h,lnl-m,ptl`，**没有 dg2（A770/A750）**。发布构建必须
显式设置 `TORCH_XPU_ARCH_LIST=bmg,dg2`，并保留通用 `spir64` JIT 作为
兜底（A770 本次仅做了兼容构建，未实测，见交付报告风险项）。

## 8. setup.py 的 MSVC flag 包装 monkeypatch 不能删

`setup.py` 中 `_wrap_sycl_host_flags` 把 icx 传给 MSVC host 编译器的
包含路径参数重新分组，否则含空格/连字符的深层安装路径（如整合包
目录名）会让 `-I` 参数断裂导致编译失败。
后续维护者不要把这个 patch 当作无用代码清理掉。

## 9. 命中率低的根因是几何必然，不是内核 bug

WG=256 时一个工作组基本横跨特征图的整行（1080p 行宽 240、1440p 行宽
320），scale=1 层的 staged patch 宽度 ≈ 整行跨度 + 光流抖动 + slack 环。
host 侧复算（span_probe.py）与设备端计数器完全一致地显示：identity/
shift 类坐标 100% 命中；平滑光流约 52–70%；真实光流 67–98%（1440p B=1
最低）；iid ±25px 随机坐标 0%（span 达 18k–23k float，必然超过 8192
预算）。结论：回退逻辑按设计工作；想提升真实场景命中率应减小 WG
（64/128），而不是放宽数值安全检查。

## 10. 固定预留 32KiB SLM 对回退路径的影响

`local_accessor` 按 `kSlmFloats=8192` 固定申请，即使某次 launch 全部落
到 direct 路径也占用 SLM，可能压低 occupancy。基准中随机坐标（0% 命中）
场景 SLM 实例化慢于纯 direct 的现象与此假设一致。若未来要消除该代价，
需要按 allow_staged 拆分 kernel 实例化的 slab 申请（当前一轮优化未做，
见交付报告"未决事项"）。

## 11. 基准方法论：环境变量是每次调用读取的

launcher 在每次 kernel 调用时 `std::getenv` 读取 `XESS_XPU_CORR_SLM`
等开关。这带来两个结论：

* 同一进程内可以逐调用翻转开关做 A/B 交错计时（v2 基准的做法），
  消除顺序计时带来的热漂移偏差；
* 反过来，靠进程启动时读环境的假设写的缓存型基准会失真。

第一版基准还有两处缺陷已在新版修正：streaming/slm 比值列把秒除以毫秒
（数量级显示错误）、slm→direct→streaming 顺序计时受热漂移影响。

## 12. 统计计数器张量故意泄漏

`xess_corr_stats_storage()` 返回进程级静态 `at::Tensor` 且故意不释放：
解释器关闭阶段静态对象析构顺序不确定，提前析构会在 CUDA/XPU 上下文
销毁后触发 device 张量析构崩溃。泄漏量恒定（2 个 int），属有意设计。
