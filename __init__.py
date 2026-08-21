from .xess_nodes import (
    XeSSFrameGeneration,
    XeSSSuperResolution,
    XeSSVideoFrameGeneration,
    XeSSVideoFrameGenerationExpert,
    XeSSVideoSuperResolution,
    XeSSVideoSuperResolutionExpert,
)


NODE_CLASS_MAPPINGS = {
    "XeSSSuperResolution": XeSSSuperResolution,
    "XeSSFrameGeneration": XeSSFrameGeneration,
    "XeSSVideoSuperResolution": XeSSVideoSuperResolution,
    "XeSSVideoFrameGeneration": XeSSVideoFrameGeneration,
    "XeSSVideoSuperResolutionExpert": XeSSVideoSuperResolutionExpert,
    "XeSSVideoFrameGenerationExpert": XeSSVideoFrameGenerationExpert,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XeSSSuperResolution": "XeSS 图像超分（专家）",
    "XeSSFrameGeneration": "XeSS 图像插帧（专家）",
    "XeSSVideoSuperResolution": "XeSS 视频超分（两挡自动）",
    "XeSSVideoFrameGeneration": "XeSS 视频插帧（两挡自动）",
    "XeSSVideoSuperResolutionExpert": "XeSS 视频超分（专家）",
    "XeSSVideoFrameGenerationExpert": "XeSS 视频插帧（专家）",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
