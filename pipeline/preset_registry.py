# -*- coding: utf-8 -*-
"""Phase 8 preset registry: the single source of truth for public tiers.

Public nodes (SR/FG) only expose: tier, target resolution, and a few
explicit-problem switches (banding guard, memory strategy).  Everything else
(DIS internals, MV path, depth numbers, masks, dilate, transport, threads,
device, engine path, work dir, 5-frame strength, sharpen values) is resolved
here and shared verbatim by the standalone runner and the ComfyUI node, so
numbers cannot drift between entry points.

Fast Pro entries only exist after Phase 2 passes all five gates; until then
the registry refuses them (the UI shows "本机不可用" instead of a fake tier).
"""
import os

SCALE_QUALITY = {  # copy of run_xess.quality_for semantics, kept in sync
    (0.0, 1.05): 6,
    (1.05, 1.6): 4,
    (1.6, 2.2): 3,
    (2.2, 2.8): 2,
    (2.8, 3.5): 1,
    (3.5, 99.0): 0,
}


def quality_for_scale(scale):
    """XeSS quality setting for a given upscale factor (0..6)."""
    for (lo, hi), q in SCALE_QUALITY.items():
        if lo < scale <= hi:
            return q
    return 6


SR_TIERS = {
    "极速": {
        "backend": "xess", "quality": "auto", "mv_path": "highres",
        "mv_upsample": "bilinear", "responsive_max": 0.8, "mask": "stream",
        "sharpen_mode": "fixed", "sharpen_static": 0.25, "sharpen_motion": 0.25,
        "guard_strength": 0.75, "threads": 4, "io": "shared",
        "note": "XeSS/XeFG + DIS Fast; default fast tier",
    },
    "极致画质": {
        "backend": "xess", "quality": "auto-high", "mv_path": "highres",
        "mv_upsample": "bilinear", "responsive_max": 0.8, "mask": "stream",
        "sharpen_mode": "adaptive", "sharpen_static": 0.38, "sharpen_motion": 0.20,
        "guard_strength": 0.75, "threads": 4, "io": "shared",
        "note": "XeSS/XeFG higher quality; buffered-realtime or offline preview",
    },
}
FG_TIERS = {
    "极速": {
        "backend": "xess", "mv_path": "lowres-depth", "depth": "ai",
        "motion_window": 2, "temporal_motion": 0.65, "temporal_depth": 0.18,
        "dilate": 1, "depth_edge": 0.04, "consistency": 1.5,
        "sharpen_mode": "fixed", "sharpen": 0.12,
        "note": "XeSS FG + DIS + AI depth; default fast tier",
    },
    "极致画质": {
        "backend": "xess", "mv_path": "lowres-depth", "depth": "ai",
        "motion_window": 5, "temporal_motion": 0.65, "temporal_depth": 0.18,
        "dilate": 1, "depth_edge": 0.04, "consistency": 1.5,
        "sharpen_mode": "adaptive", "sharpen": 0.18,
        "note": "XeSS FG 5-frame window; buffered-realtime or offline preview",
    },
}
# Fast Pro survives the Phase 2 gates ONLY as a speed-first preview backend:
# capability/perf/stability passed, quality did not (SR PSNR -10dB vs XeSS,
# FI SSIM -0.07 / edge -3dB).  It is opt-in, never a default, never shown as
# a quality tier, and never silently substitutes for XeSS.
FAST_PRO = {
    "backend": "intel-vpl-ai",
    "role": "preview-only",
    "sr_scale_min": 1.5,          # 1.33x is MFX_ERR_UNSUPPORTED
    "ui_note": "画质取舍：适合实时预览/快速查看，不适合成片",
    "internal_params_hidden": True,
}

BANDING_GUARDS = {
    "自动": {"guard_strength": 0.75, "note": "已验证默认组合"},
    "加强": {"guard_strength": 0.90, "note": "固定组合，历史片验证"},
    "关闭": {"guard_strength": 0.0, "note": "帮助确认软化是否来自保护层"},
}

MEMORY_STRATEGIES = {
    "自动": {"queue_slots": 5, "preview_scale": 1.0, "max_ram_pct": 0.60,
             "max_vram_pct": 0.70},
    "保守": {"queue_slots": 4, "preview_scale": 0.5, "depth_cadence": 4,
             "max_ram_pct": 0.45, "max_vram_pct": 0.55},
}


class RegistryError(KeyError):
    pass


def resolve_sr(tier, scale, banding="自动", memory="自动"):
    if tier not in SR_TIERS:
        raise RegistryError(f"unknown SR tier {tier!r}")
    if banding not in BANDING_GUARDS:
        raise RegistryError(f"unknown banding setting {banding!r}")
    if memory not in MEMORY_STRATEGIES:
        raise RegistryError(f"unknown memory strategy {memory!r}")
    base = dict(SR_TIERS[tier])
    base["quality"] = (quality_for_scale(scale) if base["quality"] == "auto"
                       else max(0, min(6, quality_for_scale(scale) + 1)))
    base["guard_strength"] = BANDING_GUARDS[banding]["guard_strength"]
    base.update(MEMORY_STRATEGIES[memory])
    return base


def resolve_fg(tier, memory="自动"):
    if tier not in FG_TIERS:
        raise RegistryError(f"unknown FG tier {tier!r}")
    if memory not in MEMORY_STRATEGIES:
        raise RegistryError(f"unknown memory strategy {memory!r}")
    base = dict(FG_TIERS[tier])
    base.update(MEMORY_STRATEGIES[memory])
    return base


def fast_pro_available():
    return FAST_PRO is not None


def resolve_fast_pro_preview():
    """Preview-only profile; refuses if the gates were later re-revoked."""
    if FAST_PRO is None:
        raise RegistryError("Fast Pro 未通过门槛，不可用")
    return dict(FAST_PRO)
