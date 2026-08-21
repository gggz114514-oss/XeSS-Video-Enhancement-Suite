#!/usr/bin/env python3
"""DIS 光流 -> XeSS 运动矢量

对视频逐帧计算前向光流 f(t->t+1)（单位：输入分辨率像素），
输出每帧一个 float32 [fx, fy] 交错排列的 bin（H*W*2），供 xess-vsr.exe 读取。
第 0 帧无前文 -> 零矢量（XeSS 对首帧本就无历史可复用）。

支持两种输入：
  视频:    flow.py input.mp4 --out mvs
  raw:     flow.py --raw frames.raw --in-w W --in-h H --frames N --out mvs

用法:
  venv\\Scripts\\python.exe flow.py input.mp4 --out mvs [--max-frames N] [--sign -1.0]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


def run_dis(cap_or_read, n, w, h, out, sign, label):
    """cap_or_read 每次调用返回下一帧 BGR（或 None）；逐帧算 DIS 并写 bin。"""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev_bgr = cap_or_read(0)
    if prev_bgr is None:
        sys.exit("[flow] 首帧读取失败")
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    np.zeros((h, w, 2), dtype=np.float32).tofile(os.path.join(out, "mv_000000.bin"))
    done = 1
    for idx in range(1, n):
        bgr = cap_or_read(idx)
        if bgr is None:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        flow = dis.calc(prev_gray, gray, None)          # 前向流 t->t+1，像素
        (flow * sign).astype(np.float32).tofile(os.path.join(out, f"mv_{idx:06d}.bin"))
        prev_gray = gray
        done += 1
        if idx % 25 == 0:
            print(f"[flow] {idx}/{n}", flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="输入 mp4（一采视频）")
    ap.add_argument("--raw", action="store_true", help="输入是 raw rgb24 流（需 --in-w/--in-h/--frames）")
    ap.add_argument("--in-w", type=int, default=0, help="raw 模式宽度")
    ap.add_argument("--in-h", type=int, default=0, help="raw 模式高度")
    ap.add_argument("--frames", type=int, default=0, help="raw 模式帧数")
    ap.add_argument("--out", default="mvs", help="运动矢量输出目录")
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（冒烟测试用）")
    ap.add_argument("--sign", type=float, default=-1.0,
                    help="MV 符号：-1 = 前向流取反（XeSS 常见约定：矢量指向上一帧位置），+1 = 原样")
    ap.add_argument("--probe-only", action="store_true",
                    help="只输出 {width,height,fps,frames} JSON 到 stdout，不算光流")
    args = ap.parse_args()

    if args.probe_only:
        if args.raw:
            if not (args.in_w > 0 and args.in_h > 0 and args.video):
                sys.exit("[flow] raw 探测需要 video=raw文件路径 与 --in-w/--in-h")
            n_total = os.path.getsize(args.video) // (args.in_w * args.in_h * 3)
            if args.frames > 0:
                n_total = min(n_total, args.frames)
            if args.max_frames > 0:
                n_total = min(n_total, args.max_frames)
            print(json.dumps({"width": args.in_w, "height": args.in_h,
                              "fps": 0.0, "frames": n_total}))
        else:
            if not args.video:
                sys.exit("[flow] 探测需要 video 参数（或 --raw 模式）")
            cap = cv2.VideoCapture(args.video)
            if not cap.isOpened():
                sys.exit(f"[flow] 打不开 {args.video}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if args.max_frames and args.max_frames > 0:
                n = min(n, args.max_frames)
            print(json.dumps({
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0.0, 4),
                "frames": n}))
            cap.release()
        return

    os.makedirs(args.out, exist_ok=True)

    if args.raw:
        if not (args.in_w > 0 and args.in_h > 0 and args.video):
            sys.exit("[flow] raw 模式需要 video=raw文件路径 与 --in-w/--in-h")
        w, h = args.in_w, args.in_h
        frame_bytes = w * h * 3
        n_total = os.path.getsize(args.video) // frame_bytes
        if args.frames > 0:
            n_total = min(n_total, args.frames)
        if args.max_frames > 0:
            n_total = min(n_total, args.max_frames)
        # Memory-map raw input so finished-resolution 2K/4K videos do not need
        # several gigabytes of resident RAM merely to calculate optical flow.
        data = np.memmap(args.video, dtype=np.uint8, mode="r",
                         shape=(n_total, h, w, 3))
        fps = 0.0

        def read_raw(i):
            return data[i] if i < n_total else None

        n_done = run_dis(read_raw, n_total, w, h, args.out, args.sign, "raw")
    else:
        if not args.video:
            sys.exit("[flow] 需要 video 参数（或 --raw 模式）")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"[flow] 打不开 {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.max_frames and args.max_frames > 0:
            n = min(n, args.max_frames)
        print(f"[flow] {args.video}: {w}x{h} {n} 帧 fps={fps:.3f}")

        def read_video(i):
            ok, bgr = cap.read()
            return bgr if ok else None

        n_done = run_dis(read_video, n, w, h, args.out, args.sign, "video")
        cap.release()

    meta = {"width": w, "height": h, "fps": round(fps, 4), "frames": n_done, "sign": args.sign}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[flow] 完成 {n_done} 帧 -> {args.out}")


if __name__ == "__main__":
    main()
