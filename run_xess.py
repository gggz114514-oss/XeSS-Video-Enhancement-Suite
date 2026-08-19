#!/usr/bin/env python3
"""XeSS 便携包一键入口：指定视频 + 放大倍率，自动完成
probe → ffmpeg 解码 raw → DIS 光流 → XeSS 放大 → 封装 mp4（带音频）。

用法:
  run_xess.py <input.mp4> [倍率] [--quality Q] [--frames N] [--out-dir DIR]

  倍率      默认 1.0（即 XeSS AA，只修画面不放大）；支持 1.5 / 2 / 2.5 / 3 等
  --quality XeSS 档位 0-5，缺省按倍率自动选择
  --frames  只处理前 N 帧（测试用）
  --out-dir 输出目录，缺省与输入视频同目录
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, "python", "python.exe")
FFMPEG = os.path.join(ROOT, "ffmpeg.exe")
XESS = os.path.join(ROOT, "xess-vsr.exe")
FLOW = os.path.join(ROOT, "flow.py")


def die(msg):
    print(f"[run_xess] 错误: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd, **kw):
    print(f"[run_xess] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        die(f"命令失败 (exit {r.returncode}): {' '.join(cmd)}\n"
            + (r.stderr[-2000:] if r.stderr else ""))
    return r


def quality_for(scale):
    if scale <= 1.6:
        return 4          # ultra-quality (1.5x)
    if scale <= 2.2:
        return 3          # quality (1.7x)
    if scale <= 2.8:
        return 2          # balanced (2.3x)
    if scale <= 3.5:
        return 1          # performance (3.0x)
    return 0              # ultra-performance (4.0x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="输入视频（mp4 等 ffmpeg 可解码的格式）")
    ap.add_argument("scale", nargs="?", type=float, default=1.0, help="放大倍率，默认 1.0 = AA")
    ap.add_argument("--quality", type=int, default=-1, help="XeSS 档位 0-5，缺省按倍率自动")
    ap.add_argument("--frames", type=int, default=0, help="只处理前 N 帧（测试用）")
    ap.add_argument("--out-dir", default="", help="输出目录，缺省为输入视频所在目录")
    ap.add_argument("--keep", action="store_true", help="保留中间 raw/光流文件")
    args = ap.parse_args()

    for p, name in ((PY, "python\\python.exe"), (FFMPEG, "ffmpeg.exe"),
                    (XESS, "xess-vsr.exe"), (FLOW, "flow.py")):
        if not os.path.isfile(p):
            die(f"缺少 {name}，便携包不完整")
    if not os.path.isfile(args.video):
        die(f"找不到视频: {args.video}")
    if args.scale <= 0:
        die(f"倍率必须大于 0: {args.scale}")

    # 1) 探测视频参数
    meta = json.loads(subprocess.run(
        [PY, FLOW, args.video, "--probe-only"],
        check=True, capture_output=True, text=True).stdout)
    in_w, in_h, fps, n_total = meta["width"], meta["height"], meta["fps"], meta["frames"]
    if args.frames > 0:
        n_total = min(n_total, args.frames)
    print(f"[run_xess] {args.video}: {in_w}x{in_h} {fps}fps {meta['frames']}帧，"
          f"倍率 {args.scale:g}x")

    # 2) 计算输出分辨率（偶数，16 的倍数最稳），比例与输入一致
    out_w = int(round(in_w * args.scale / 16) * 16)
    out_h = int(round(in_h * args.scale / 16) * 16)
    if out_w < 2 or out_h < 2:
        die("输出分辨率非法")
    quality = args.quality if args.quality >= 0 else quality_for(args.scale)
    mode = "XeSS AA（只修复不放大）" if out_w == in_w and out_h == in_h else f"放大到 {out_w}x{out_h}"
    print(f"[run_xess] 输出 {out_w}x{out_h}，quality {quality} [{mode}]")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.video))
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_mp4 = os.path.join(out_dir, f"{base}_xess_{args.scale:g}x_{out_w}x{out_h}.mp4")

    tmp = tempfile.mkdtemp(prefix="xess_run_")
    try:
        raw = os.path.join(tmp, "frames.raw")
        mvs = os.path.join(tmp, "mvs")
        out_raw = os.path.join(tmp, "out.raw")

        # 3) 解码成 rgb24 raw（-an 丢弃原音频，封装时再从源取）
        run([FFMPEG, "-y", "-i", args.video, "-an",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{in_w}x{in_h}",
             "-vframes", str(n_total), raw])

        # 4) DIS 光流
        run([PY, FLOW, raw, "--raw", "--in-w", str(in_w), "--in-h", str(in_h),
             "--frames", str(n_total), "--out", mvs])

        # 5) XeSS 放大
        run([XESS, "--frames", raw, "--mv", mvs,
             "--in-w", str(in_w), "--in-h", str(in_h),
             "--out-w", str(out_w), "--out-h", str(out_h),
             "--frames-count", str(n_total), "--quality", str(quality),
             "--out", out_raw])

        # 6) 封装 mp4（音频从源视频拷，源无音轨也兼容）
        os.makedirs(out_dir, exist_ok=True)
        run([FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", out_raw,
             "-i", args.video, "-map", "0:v", "-map", "1:a?",
             "-c:v", "libx264", "-preset", "slow", "-crf", "16",
             "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", out_mp4])
    finally:
        if args.keep:
            print(f"[run_xess] 中间文件保留在 {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"[run_xess] 完成 -> {out_mp4}")


if __name__ == "__main__":
    main()
