"""Explicit terminal encoding selection; never changes motion/SR/FG algorithms."""
import os

ENCODERS = ("h264_qsv", "hevc_qsv", "libx264", "libx265", "ffv1")


def configure_cli_encoder(args):
    selection = getattr(args, "encoder", None)
    if selection is not None:
        os.environ["XESS_OUTPUT_ENCODER"] = "h264_qsv" if selection == "auto" else selection
    selected = os.environ.get("XESS_OUTPUT_ENCODER")
    if selected and selected not in ENCODERS:
        raise ValueError("不支持编码器：" + selected)
    if getattr(args, "pipeline_backend", "classic") == "full-gpu" and selected not in (None, "h264_qsv"):
        raise ValueError("旧 full-gpu 命令只支持 H.264；其他编码器请使用新版离线工具箱入口。")
    if selected == "ffv1":
        os.environ["XESS_OUTPUT_CONTAINER"] = ".mkv"
        if getattr(args, "io_mode", "stream") == "chunked":
            raise ValueError("FFV1 当前请使用 --io-mode stream，不支持旧 MP4 分段链。")
        if getattr(args, "io_mode", "stream") == "auto":
            args.io_mode = "stream"


def video_options(args):
    # Legacy commands retain their previous default. The toolbox explicitly
    # supplies its selection to this isolated worker environment.
    encoder = os.environ.get("XESS_OUTPUT_ENCODER", "libx264")
    if encoder not in ENCODERS:
        raise ValueError("不支持的成片编码器：" + encoder)
    if encoder.endswith("_qsv"):
        return ["-c:v", encoder, "-global_quality", "20", "-preset", "medium",
                "-bf", "0", "-async_depth", "1", "-pix_fmt", "nv12"]
    if encoder == "ffv1":
        # FFmpeg 7.1 FFV1 has no 8-bit gbrp format. bgr0 preserves RGB bytes.
        return ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0"]
    return ["-c:v", encoder, "-preset", args.encoder_preset,
            "-crf", str(args.crf), "-pix_fmt", "yuv420p"]
