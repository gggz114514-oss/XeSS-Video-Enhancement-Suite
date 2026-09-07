# -*- coding: utf-8 -*-
"""Fail-closed Fast Pro SR geometry planner.

The B580 matrix proves that direct oneVPL AI SR accepts an orientation-
independent surface when the *aligned input height* is at most 1440 and the
uniform requested scale is at least 1.4x.  FI has a separate capability
contract and must not inherit these SR limits.

For portrait input whose coded height is above 1440, the product route can
transpose the frame, run AI SR on the supported landscape geometry, transpose
back, and downscale only when the measured intermediate scale covers the
requested target.  The planner never returns a normal resize as an AI route:
if no measured AI route fits, callers must select XeSS explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SR_INPUT_ALIGNED_HEIGHT_MAX = 1440
SR_SCALE_MIN = 1.4
AI_SAFETY_MARGIN = 1.05
_MEASURED_AI_SCALES = (1.5, 2.0, 1.6, 1.4)


def align16(value: int) -> int:
    """Return the native VPP's 16-row aligned surface height."""
    return (int(value) + 15) // 16 * 16


def is_sr_direct_supported(in_width: int, in_height: int, out_width: int,
                           out_height: int) -> bool:
    """Return whether the measured direct AI SR geometry is supported."""
    if min(in_width, in_height, out_width, out_height) <= 0:
        return False
    scale_h = out_height / in_height
    scale_w = out_width / in_width
    if abs(scale_h - scale_w) > 0.01 * max(scale_h, scale_w):
        return False
    return (
        align16(in_height) <= SR_INPUT_ALIGNED_HEIGHT_MAX
        and scale_h >= SR_SCALE_MIN - 1e-9
    )


@dataclass(frozen=True)
class RoutePlan:
    """The requested route and the exact dimensions each stage will use."""

    route: str  # "direct" | "rotate" | "preshrink" | "unsupported"
    reason: str
    detail: dict[str, Any]

    @property
    def supported(self) -> bool:
        return self.route in {"direct", "rotate", "preshrink"}


def _even(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def _rotate_plan(in_w: int, in_h: int, out_w: int,
                 out_h: int) -> RoutePlan | None:
    """Plan transpose -> measured AI SR -> transpose back -> final downscale."""
    if align16(in_w) > SR_INPUT_ALIGNED_HEIGHT_MAX:
        return None
    min_scale = max(out_w / in_w, out_h / in_h)
    # Rotation is an exact pixel permutation. Requiring 5% excess scale
    # rejected an otherwise valid 2x request although the 2x AI output covers
    # every requested pixel. Check actual even dimensions below instead.
    required = max(SR_SCALE_MIN, min_scale)
    ai_scale = next((candidate for candidate in _MEASURED_AI_SCALES
                     if candidate >= required), None)
    if ai_scale is None:
        return None
    mid_w = _even(in_w * ai_scale)
    mid_h = _even(in_h * ai_scale)
    if mid_w < out_w or mid_h < out_h:
        return None
    return RoutePlan(
        route="rotate",
        reason=(
            f"direct SR rejected (aligned input height {align16(in_h)} > "
            f"{SR_INPUT_ALIGNED_HEIGHT_MAX} or scale < {SR_SCALE_MIN}); "
            f"rotate path runs AI SR {in_h}x{in_w} -> {mid_h}x{mid_w} "
            f"at {ai_scale}x, then rotates back and downscales"
        ),
        detail={
            "ai_scale": ai_scale,
            "rotate_in": [in_h, in_w],
            "ai_out": [_even(in_h * ai_scale), _even(in_w * ai_scale)],
            "rotate_back": [mid_w, mid_h],
            "final_scale": [out_w, out_h],
            "final_filter": "lanczos",
            "extra_cpu_stages": ["transpose 90cw", "transpose 90ccw",
                                 "downscale"],
        },
    )


def _preshrink_plan(in_w: int, in_h: int, out_w: int,
                    out_h: int) -> RoutePlan | None:
    """Plan the lossy compatibility floor: plain half-size then AI SR 2x."""
    if out_w % 2 or out_h % 2:
        return None
    pre_w, pre_h = out_w // 2, out_h // 2
    if pre_w <= 0 or pre_h <= 0 or pre_w >= in_w or pre_h >= in_h:
        return None
    if align16(pre_h) > SR_INPUT_ALIGNED_HEIGHT_MAX:
        return None
    return RoutePlan(
        route="preshrink",
        reason=(
            f"direct SR rejected; preshrink {in_w}x{in_h} -> "
            f"{pre_w}x{pre_h} then AI SR 2x -> {out_w}x{out_h}; "
            "this route discards input detail"
        ),
        detail={
            "preshrink": pre_h / in_h,
            "pre_dims": [pre_w, pre_h],
            "ai_scale": 2.0,
            "final_scale": [out_w, out_h],
        },
    )


def plan_sr_route(in_width: int, in_height: int, out_width: int,
                  out_height: int) -> RoutePlan:
    """Select direct, lossless-pixel rotate, lossy preshrink, or unsupported."""
    if is_sr_direct_supported(in_width, in_height, out_width, out_height):
        return RoutePlan(
            route="direct",
            reason="within measured AI SR limits",
            detail={"scale": out_height / in_height},
        )
    rotate = _rotate_plan(in_width, in_height, out_width, out_height)
    preshrink = _preshrink_plan(in_width, in_height, out_width, out_height)
    if rotate is not None:
        return rotate
    if preshrink is not None:
        return preshrink
    return RoutePlan(
        route="unsupported",
        reason=(
            f"no measured AI SR route reaches {out_width}x{out_height} from "
            f"{in_width}x{in_height} (aligned input height limit "
            f"{SR_INPUT_ALIGNED_HEIGHT_MAX}, scale >= {SR_SCALE_MIN}); "
            "use the XeSS quality path for this geometry"
        ),
        detail={},
    )
