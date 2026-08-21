from __future__ import annotations

import argparse
import pathlib
import sys
from fractions import Fraction

import cv2
import numpy as np
import torch


NODE_DIR = pathlib.Path(__file__).resolve().parent
COMFY_ROOT = NODE_DIR.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(NODE_DIR))

from xess_nodes import (  # noqa: E402
    XeSSFrameGeneration,
    XeSSSuperResolution,
    XeSSVideoFrameGeneration,
    XeSSVideoFrameGenerationExpert,
    XeSSVideoSuperResolution,
    XeSSVideoSuperResolutionExpert,
)


def frames_from_video(path: str, count: int) -> torch.Tensor:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < count:
        raise RuntimeError(f"decoded {len(frames)}/{count} frames")
    return torch.from_numpy(np.stack(frames)).float().div_(255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-memory smoke test for ComfyUI-XeSS")
    parser.add_argument("input")
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--work-dir", default="auto")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--mfsr", action="store_true")
    parser.add_argument("--sr-only", action="store_true")
    parser.add_argument("--native-video", action="store_true")
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--allow-overlay", action="store_true")
    parser.add_argument("--transport", choices=("stream", "shared"), default="stream")
    parser.add_argument("--sr-preset", choices=("fast", "balanced", "quality"), default="fast")
    parser.add_argument("--fg-preset", choices=("fast", "balanced", "quality"), default="fast")
    parser.add_argument("--fusion", type=float, default=0.0)
    parser.add_argument("--ui-mask", action="store_true")
    args = parser.parse_args()
    source = frames_from_video(args.input, args.frames)
    preset_cn = {
        "fast": "快速（速度优先）", "balanced": "均衡（质量优先）", "quality": "高质量（最慢）",
    }
    transport_cn = {"stream": "内存管道", "shared": "共享内存"}
    sr_options = dict(
        preset=preset_cn[args.sr_preset], scale=1.5, quality="自动（按倍率选择）",
        flow_mode="跟随处理档位", mv_path="跟随处理档位",
        responsive_mask=True, responsive_strength=0.8,
        depth_temporal=0.25, flow_consistency=1.5, mv_dilate=1, depth_edge=0.04,
        temporal_fusion=args.fusion, mfsr_enabled=args.mfsr, mfsr_strength=1.8,
        mfsr_detail_boost=0.9, mfsr_max_injection=22.0,
        sharpen_mode="关闭", sharpen_static=-1.0, sharpen_motion=-1.0,
        transport=transport_cn[args.transport], device=-1, free_vram=False, max_output_gb=4.0,
        engine_path=args.engine, work_dir=args.work_dir, verbose=args.verbose,
    )
    sr_video = None
    if args.native_video:
        from comfy_api.latest import InputImpl, Types
        input_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=source, audio=None, frame_rate=Fraction(24, 1)),
            bit_depth=8,
        )
        if args.simple:
            mode = "极致画质（最高挡）" if args.sr_preset == "quality" else "极速模式（最低挡）"
            sr_video, width, height, sr_info = XeSSVideoSuperResolution().upscale_video(
                input_video, mode=mode, scale=1.5
            )
        else:
            sr_video, width, height, sr_info = XeSSVideoSuperResolutionExpert().upscale_video(
                input_video, **sr_options
            )
        sr_components = sr_video.get_components()
        sr_images = sr_components.images
        assert sr_components.frame_rate == Fraction(24, 1)
    else:
        sr_images, width, height, sr_info = XeSSSuperResolution().upscale(
            source, **sr_options
        )
    print(sr_info)
    assert tuple(sr_images.shape) == (args.frames, height, width, 3)
    if args.sr_only:
        print("ComfyUI-XeSS SR-only self-test passed")
        return
    ui_mask = None
    if args.ui_mask:
        ui_mask = torch.zeros((1, height, width), dtype=torch.float32)
        ui_mask[:, height * 3 // 4:height * 7 // 8, width // 8:width * 7 // 8] = 1.0
    fg_options = dict(
        preset=preset_cn[args.fg_preset], flow_mode="跟随处理档位",
        depth_mode="AI 深度（推荐）", motion_window="跟随处理档位", depth_temporal=0.25,
        flow_consistency=1.5, mv_dilate=1, depth_edge=0.04,
        temporal_motion_strength=0.65, temporal_depth_strength=0.18,
        sharpen_mode="关闭", sharpen_static=-1.0, sharpen_motion=-1.0,
        allow_overlay=args.allow_overlay, transport=transport_cn[args.transport], device=-1, free_vram=False,
        max_output_gb=4.0, engine_path=args.engine, work_dir=args.work_dir,
        verbose=args.verbose, ui_mask=ui_mask,
    )
    if args.native_video:
        if args.simple:
            mode = "极致画质（最高挡）" if args.fg_preset == "quality" else "极速模式（最低挡）"
            fg_video, output_fps, output_count, fg_info = XeSSVideoFrameGeneration().interpolate_video(
                sr_video, mode=mode, ui_mask=ui_mask
            )
        else:
            fg_video, output_fps, output_count, fg_info = (
                XeSSVideoFrameGenerationExpert().interpolate_video(sr_video, **fg_options)
            )
        fg_components = fg_video.get_components()
        fg_images = fg_components.images
        assert fg_components.frame_rate == Fraction(48, 1)
    else:
        fg_images, output_fps, output_count, fg_info = XeSSFrameGeneration().interpolate(
            sr_images, source_fps=24.0, **fg_options
        )
    print(fg_info)
    assert tuple(fg_images.shape) == (args.frames * 2 - 1, height, width, 3)
    assert output_fps == 48.0 and output_count == args.frames * 2 - 1
    print("ComfyUI-XeSS self-test passed")


if __name__ == "__main__":
    main()
