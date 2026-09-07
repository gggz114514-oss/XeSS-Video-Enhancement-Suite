# -*- coding: utf-8 -*-
"""Product runtime hooks wiring the realtime components into run_xess/run_fg.

  * preflight_budget()  - plan section 16 gate with Chinese messaging
  * tier_settings_sr()  - plan section 14 preset resolution (single source)
  * preview_tap_command()- builds the transparent preview tap process args
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from memory_budget import BudgetExceeded, MemoryBudget  # noqa: E402
from preset_registry import RegistryError, resolve_sr  # noqa: E402

TIER_ALIASES = {"极速": "极速", "fast": "极速", "极致画质": "极致画质",
                "quality": "极致画质"}


def _budget_item(requested, planned, applied, adjustable, note=None):
    """One budget knob reported as requested/planned/applied; knobs outside
    the shrink chain are marked fixed so nothing claims to have adjusted
    what it cannot (task book V3 §5.2)."""
    item = {"requested": requested, "planned": planned, "applied": applied,
            "adjustable": bool(adjustable), "fixed": not adjustable}
    if note:
        item["note"] = note
    return item


def preflight_budget(work_dir, strategy, in_w, in_h, out_w, out_h,
                     frames, fps, depth_on, post_threads=None,
                     extra_fixed=None):
    """Return (ok, report dict).  Shrinks queue/preview per plan 16.9 and
    reports every knob as requested/planned/applied; fixed knobs (depth
    cadence until Phase 4/8, post threads, output ring…) are tagged as
    fixed instead of pretending they were adjusted."""
    budget = MemoryBudget(work_dir, strategy=strategy)
    requested = budget.starting_point()
    items = {
        "input_ring_slots": _budget_item(
            requested["input_ring_slots"], None, None, adjustable=True),
        "preview_scale": _budget_item(
            requested["preview_scale"], None, None, adjustable=True),
        # depth cadence is not implemented in execution yet; estimates and
        # the report use cadence=1 and tag it fixed rather than fake-applied
        "depth_cadence": _budget_item(
            requested["depth_cadence"], None, None, adjustable=False,
            note="Phase 8 cadence 未生效，执行按 cadence=1"),
    }
    for name, value in (extra_fixed or {}).items():
        items[name] = _budget_item(value, value, value, adjustable=False,
                                   note="固定参数，不在预算收缩链中")
    if post_threads is not None:
        items["post_threads"] = _budget_item(
            post_threads, post_threads, post_threads, adjustable=False,
            note="调用方固定")
    try:
        adj = budget.shrink(in_w=in_w, in_h=in_h, out_w=out_w, out_h=out_h,
                            frames=frames, fps=fps, depth_on=depth_on)
        est = budget.estimate_total(in_w, in_h, out_w, out_h, frames, fps=fps,
                                    queue_slots=adj["queue_slots"],
                                    preview_scale=adj["preview_scale"],
                                    depth_cadence=adj["depth_cadence"],
                                    depth_on=depth_on)
        check = budget.check(est)
        items["input_ring_slots"]["planned"] = adj["queue_slots"]
        items["input_ring_slots"]["applied"] = adj["queue_slots"]
        items["preview_scale"]["planned"] = adj["preview_scale"]
        items["preview_scale"]["applied"] = adj["preview_scale"]
        items["depth_cadence"]["planned"] = adj["depth_cadence"]
        items["depth_cadence"]["applied"] = None
        return True, {
            "ok": check["ok"], "est_mib": est["est_mib"],
            "strategy": strategy,
            "queue_slots": adj["queue_slots"],
            "preview_scale": adj["preview_scale"],
            "depth_cadence": adj["depth_cadence"],
            "items": items,
            "message": f"资源预算充足（预计 {est['est_mib']:.0f} MiB）"
                       if check["ok"] else check["hint"],
        }
    except BudgetExceeded as e:
        return False, {"ok": False, "strategy": strategy, "items": items,
                       "message": str(e)}


def tier_settings_sr(tier, scale):
    """Map a public tier name to concrete SR settings; '' keeps legacy CLI."""
    if not tier:
        return None
    name = TIER_ALIASES.get(tier)
    if not name:
        raise RegistryError(f"未知档位 {tier!r}")
    p = resolve_sr(name, scale=scale)
    # translate registry fields to run_xess settings keys
    return {
        "mv_path": p["mv_path"],
        "responsive": True,
        "sharpen_mode": p["sharpen_mode"] if p["sharpen_mode"] != "off" else "off",
        "sharpen": None,
        "static": p["sharpen_static"],
        "motion": p["sharpen_motion"],
        "guard_strength": p["guard_strength"],
        "post_threads": p["threads"],
        "_quality_hint": p["quality"],
    }


def preview_tap_command(python_exe, width, height, fps, frames, session_dir,
                        ffmpeg, backend="xess", preview_scale=1.0):
    """Transparent tap args; inserted before the encoder in the chain.
    preview_scale < 1 shrinks what the PREVIEW stores only (the final file
    keeps full resolution), honouring the memory strategy."""
    cmd = [python_exe, os.path.join(HERE, "preview_tap.py"),
           "--width", str(width), "--height", str(height),
           "--fps", str(fps), "--frames", str(frames),
           "--session-dir", session_dir, "--ffmpeg", ffmpeg,
           "--backend", backend]
    if preview_scale and float(preview_scale) < 1.0:
        cmd += ["--store-scale", f"{float(preview_scale):.3f}"]
    return cmd
