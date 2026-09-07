"""Release/source-tree boundary: GPU DIS / GPU Block ship as compiled runtime."""
from pathlib import PurePosixPath

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hlsl", ".hlsli", ".cl", ".inl", ".sycl"}


def is_gpu_dis_source(path) -> bool:
    path = PurePosixPath(str(path).replace("\\", "/").casefold())
    if path.suffix not in SOURCE_SUFFIXES:
        return False
    return (
        any(part in {"gpu_dis", "gpu-dis", "strict_dis", "strict-dis"} for part in path.parts[:-1])
        or path.name.startswith(("dis_", "gpu_dis", "native_strict_dis", "strict_dis", "surface_gpu_dis"))
    )


def is_gpu_block_source(path) -> bool:
    path = PurePosixPath(str(path).replace("\\", "/").casefold())
    if path.suffix not in SOURCE_SUFFIXES:
        return False
    return (
        path.name in {"vpl_gpu_full_fg.cpp", "vpl_gpu_full_chain_probe.cpp", "xve_gpu_block_core.h"}
        or any(part in {"gpu_block", "gpu-block"} for part in path.parts[:-1])
        or path.name.startswith(("gpu_block", "native_gpu_block", "surface_gpu_motion",
                                 "surface_gpu_pyramid", "surface_gpu_lite", "motion_"))
        or path.name in {"surface_gpu_velocity.hlsl", "surface_gpu_mask.hlsl"}
    )


def is_private_gpu_source(path) -> bool:
    return is_gpu_dis_source(path) or is_gpu_block_source(path)
