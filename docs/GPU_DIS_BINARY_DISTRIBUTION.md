# GPU DIS / GPU Block 编译版发布边界

本次移除当前公开源码树中的 GPU DIS、GPU Block（含 Full H2、Lite、旧实验版本）C++、头文件及 HLSL/HLSLI 实现，不删除两个档位。CPU DIS、AMD 光流、Intel 视频接口、公共特效及节点源码保持公开。

## 使用与构建

- 用户仍通过节点自动安装固定版本的 R4 Runtime。GPU DIS / GPU Block 的 EXE 和已编译着色器继续提供。
- 本次不改变 runtime_manifest.json、运行库版本、哈希或 Release 二进制资产；已安装用户无需重新下载。
- GPU Block 还内嵌在原生全链主程序及旧 xess_fg.cpp 中，本次一并移除。AMD worker 依赖同一个全链主程序，因此公开 AMD 算法源码仍在，但完整 AMD worker 也需使用发布的编译版。
- build_offline_encoders.cmd / build_amd_of_native.cmd 明确报错说明私有主程序缺失，不输出假成功；build_offline_gpu_effects.cmd 仅构建公共特效 shader。
- build_vsr_fg.bat 继续构建使用外部运动数据的传统 XeSS SR/FG worker。该公开构建不再支持 --gpu-block-motion，参数会明确报错，绝不静默替换为零光流或其他算法。
- 发布中的三种 GPU worker 必须使用配套的 EXE、DLL、着色器。维护者的完整源码与构建流程保留在本地封存开发树，不随本次提交上传。
- 原先依赖私有源码的静态槽回收测试不再在公共 CI 中执行；改为验证固定二进制合同。运行库功能仍用真实视频做回归，二者不冒充等价测试。
- validate_repo.py 与运行库打包器拒绝已识别的私有源码路径；发布前仍须人工检查改名文件、补丁、归档和嵌入式源码。

## 历史边界

这是普通删除提交，不是 Git 历史清除。旧提交、旧分支、旧标签及其 GitHub Source code 归档仍可能包含先前公开过的实现；本次未重写历史、移动标签或删除 Release。已经下载的副本也不会因此消失。

独立离线工具箱仍未获发布确认，本次不发布该项目。
