# R4 GPU 算法源码移除验收

基线：origin/main 9b68ee0e7b015661af2438a85070a62f6e4c8732。
范围：GPU DIS 与 GPU Block 的源码分发，不改视频算法、节点或已发布 Runtime。

## 修改

- 删除 77 个源码文件，包含 DIS、Block Full/Lite/旧变体、共用原生全链中的内嵌实现。
- 从公开 xess_fg.cpp 移除 GPU Block 实现；传统外部运动输入接口保留，GPU Block 参数明确拒绝。
- AMD 算法源码和公共特效不改；完整 GPU worker 的公开构建改为明确说明私有依赖。
- 添加源码路径发布检查、运行库打包检查与测试。

## 验证

- Python 单测 167/167，ResourceWarning 作为错误。
- validate_repo.py、git diff --check、Node 界面状态测试通过。
- 公共 build_vsr_fg.bat 实编译 xess-vsr.exe / xess-fg.exe 成功。
- 使用原 R4 编译库，对 GPU DIS / GPU Block / AMD 三种路线分别跑 SR、FG、SR→FG，共 9/9 成功。
- 输入均 8 帧：SR 输出 8 帧；FG/组合输出 15 帧；PTS 检查通过，音轨保留，无 partial 残留。
- 运行库打包器检查 4,448 个文件，通过源码边界检查。
- runtime_manifest.json、Python 视频管线和节点代码相对基线零修改。原运行库 ZIP SHA256：
  f9012e3e2eaf5d9caa83dc6957e4bae147bd7aa06804bba7edae83df90165bda。

本轮是源码发布边界回归，不重新宣称长时间稳定性、画质提升或 A770 实测。
GPU 静态槽回收测试随私有主程序退出公共源码验收；新的固定二进制合同测试并非其等价替代。
完整旧源码仍在本地原开发树与已有 Git 历史，可恢复；未重写历史、删除旧标签或 Release。
