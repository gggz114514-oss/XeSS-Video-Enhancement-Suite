这是一个用xess处理视频的项目，大概思路是光流补齐xess需要的运动矢量，然后使用xess超分辨率。
视频信息不足时效果480——720p出奇的差，但是用aa档（原生档）优化视频的画面还算可以（用于优化seedvr2出的视频），具体效果请看项目里面的两个视频           
xessfg（帧生成）已经适配，具体效果请看视频
然后这个便携包打开即用
便携包
xess-portable\
├── run_xess.bat          ← 一键入口
├── run_xess.py           ← 调度脚本（自动完成全部步骤）
├── flow.py               ← DIS 光流
├── xess-vsr.exe + libxess.dll + vcruntime140*.dll
├── ffmpeg.exe            ← 内置单文件 ffmpeg
└── python\               ← 便携 Python（自带 cv2/numpy，位置无关）

用法：

text

run_xess.bat 视频.mp4 [倍率] [--quality Q] [--frames N] [--out-dir D]

run_xess.bat MiniMax_H3_seedvr2_720p_00001_.mp4 → AA 修画面（倍率默认 1.0）run_xess.bat in.mp4 2 → 2 倍放大，输出 原名_xess_2x_2592x1440.mp4，自动带原音轨倍率任意（1.5 / 2 / 2.5 / 3…），quality 缺省按倍率自动选档；中间文件用完自动清理


注意 所有显卡都可以跑！不要太老就行！
