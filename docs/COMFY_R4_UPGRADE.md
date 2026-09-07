# ComfyUI R4 升级、安装与回退

## 发布边界

本版从 GitHub R3 `3f44b57` 建独立发布分支，只纳入 ComfyUI 三节点和五路视频工作器。离线 GUI、实时 GUI、OBS/WGC 和旧节点不发布。实验功能以实验标签保留，不宣称跨 GPU 验证通过。

## 自动升级合同

1. `git pull --ff-only` 更新源码和固定版本的 `runtime_manifest.json`。
2. ComfyUI 启动时后台检查运行时，节点注册不等待下载。首次执行前再次确保运行时可用。
3. 清单固定 Release URL、ZIP SHA256、展开大小和每个文件的 SHA256；不查询“最新 Release”来混配源码。
4. 解压到本节点 `.runtime/installing-*`，拒绝越界、Windows 设备名、重复文件、链接和未声明文件。
5. 完整校验后放入 `.runtime/versions/<版本>-<哈希前缀>`。旧 R3 `engine` 和旧 R4 版本保留，失败不覆盖。
6. 再启动使用文件大小/修改时间缓存；变动后重新哈希，`-Force` 可以强制完整校验并修复。该缓存是本地损坏检测，不是防御恶意本机管理员。

首次下载或修复需要网络；“一次 pull”不代表不需要重启、不需要下载或旧工作流不需换节点。自动安装不会运行 pip、不会设置系统 PATH，不会修改用户的 Comfy 配置。

手动包：`install_runtime.bat -AssetPath <ZIP>`；Python 未找到时加 `-Python <ComfyUI Python>`。可设置 `COMFYUI_XESS_RUNTIME` 改变安装目录；旧值以 `engine` 结尾时自动使用其父目录下的 versions，仍不改旧 engine。`XESS_RUNTIME_ROOT` 是高级用户显式指定的完整已安装 R4 运行时，设置后关闭自动安装，用户负责兼容性。

`COMFYUI_XESS_SKIP_RUNTIME_DOWNLOAD=1` 关闭自动下载；不能据此让缺失的运行时变成可用。下载失败后修复网络再执行节点会重试。

## 回退

先关闭 ComfyUI，备份工作流和自己修改过的源码。保持 `.runtime/engine` 不动，在插件目录使用 `git switch --detach 3f44b57` 回到已发布 R3，再启动。已有未提交修改时先自行备份/提交，不要使用 hard reset。

重新进入 R4：`git switch main` 后 `git pull --ff-only`。R3/R4 工作流不互相保证兼容，回退时使用对应版本的工作流。

旧版或 repair-backup 目录不会自动删除，避免误删在用资源。确认不再回退且所有 ComfyUI/视频任务都关闭后，可人工清理不再使用的**具体版本目录**；不要删除整个工程、输入视频或模型目录。

## 验收口径

源码单测、安装故障注入、真实 ComfyUI 执行和真机媒体矩阵分别记录，不互相替代；实际发布证据见 `docs/reports/COMFY_R4_RELEASE_ACCEPTANCE.md`。发布顺序为：封包验证 → 资产上传 → 清单与源码上线，避免 main 引用尚不存在的 ZIP。
