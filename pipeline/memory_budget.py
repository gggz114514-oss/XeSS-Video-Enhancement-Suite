# -*- coding: utf-8 -*-
"""Phase 10 automatic RAM/VRAM/disk budget controller.

Plan section 16: estimate every resource the pipeline will touch, pick auto /
conservative caps, and shrink in a fixed order instead of running into OOM:
  in-flight slots -> preview resolution -> memory buffers to disk segments
  -> realtime analysis cost -> refuse to start.

Budget sources:
  * system RAM  : GlobalMemoryStatusEx  (memory_probe.ram_mib)
  * VRAM        : DXGI QueryVideoMemoryInfo when available, else GPU Adapter
                  Memory counters (memory_probe.vram) - real usage only
  * disk        : shutil.disk_usage on the work drive; system-drive flag
"""
import os
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "..", "..", "pipeline"))

import memory_probe  # noqa: E402


def shutil_disk(cur):
    return "system-drive" if cur.get("disk_system") else "work-drive"


class BudgetExceeded(Exception):
    pass


class MemoryBudget:
    def __init__(self, work_dir, strategy="自动", pw=0.60, pv=0.70,
                 disk_safety_mib=5 * 1024, system_disk_safety_mib=25 * 1024):
        self.work_dir = work_dir
        self.strategy = strategy
        self.ram_ratio = pw
        self.vram_ratio = pv
        self.disk_safety_mib = disk_safety_mib
        self.system_disk_safety_mib = system_disk_safety_mib
        if strategy == "保守":
            self.slots0, self.scale0, self.cadence0 = 4, 0.5, 1
            self.ram_ratio = min(pw, 0.45)
            self.vram_ratio = min(pv, 0.55)
        else:
            self.slots0, self.scale0, self.cadence0 = 5, 1.0, 1
        # cadence0 stays 1 until Phase 4/8 depth cadence is really wired into
        # execution; estimates must not pretend a cadence that is not applied.

    def starting_point(self):
        """The strategy's requested starting configuration (pre-shrink)."""
        return {"input_ring_slots": self.slots0, "preview_scale": self.scale0,
                "depth_cadence": self.cadence0}

    # ---- estimators (plan 16.4) -------------------------------------------
    @staticmethod
    def estimate_frame(w, h, channels=3):
        return w * h * channels

    @staticmethod
    def estimate_textures(in_w, in_h, out_w, out_h, copies=2):
        # rgba8 upload + rgba32f? XeSS internal textures: conservatively 2x
        return 2 * (in_w * in_h * 4 + out_w * out_h * 4)

    @staticmethod
    def estimate_depth(w, h, cadence=1, in_flight=1):
        """Depth cost with cadence semantics: depth recomputed every Nth
        frame, so roughly in_flight/N hot maps stay in memory (>=1)."""
        per_map = w * h * 4
        hot_maps = max(1, max(1, in_flight) // max(1, int(cadence)))
        return per_map * hot_maps

    @staticmethod
    def estimate_ring(slots, slot_size):
        return slots * slot_size

    @staticmethod
    def estimate_preview(media_s, fps, w, h, thumb_jpeg_kb=60, seg_rate=0.1):
        thumbs = max(1, int(media_s / 1.0)) * thumb_jpeg_kb * 1024
        segments_raw = media_s * seg_rate * w * h * 3  # compressed ~10x smaller
        return thumbs + segments_raw

    def estimate_total(self, in_w, in_h, out_w, out_h, frames, fps=30,
                       queue_slots=5, preview_scale=1.0, depth_cadence=1,
                       depth_on=False):
        in_bytes = self.estimate_frame(in_w, in_h)
        out_bytes = self.estimate_frame(out_w, out_h)
        texture_bytes = self.estimate_textures(in_w, in_h, out_w, out_h)
        ring_bytes = self.estimate_ring(3, in_bytes * 2)  # analysis ring
        depth_bytes = (self.estimate_depth(in_w, in_h, cadence=depth_cadence,
                                           in_flight=max(2, queue_slots // 2))
                       if depth_on else 0)
        media_s = frames / fps
        preview_bytes = self.estimate_preview(
            media_s, fps, int(out_w * preview_scale), int(out_h * preview_scale))
        raw_total = in_bytes * queue_slots + texture_bytes + ring_bytes \
            + depth_bytes * queue_slots + out_bytes * queue_slots
        return {
            "frame_in": in_bytes, "frame_out": out_bytes,
            "textures": texture_bytes, "ring": ring_bytes,
            "depth": depth_bytes * queue_slots,
            "in_flight": (in_bytes + out_bytes) * queue_slots,
            "preview": preview_bytes, "raw_total": raw_total,
            "est_mib": (raw_total + preview_bytes) / (1 << 20),
        }

    # ---- caps ---------------------------------------------------------------
    def current(self):
        snap = memory_probe.snapshot(self.work_dir)
        ram = snap["ram"]
        vram = snap["vram"]
        disk = snap["disk"]
        out = {
            "ram_avail_mib": ram["avail_phys_mib"],
            "ram_total_mib": ram["total_phys_mib"],
            "ram_load_pct": ram["load_pct"],
            "disk_free_mib": disk["free_mib"] if disk else None,
            "disk_system": disk["is_system_drive"] if disk else None,
        }
        usages = vram.get("usage_mib") or []
        out["vram_usage_mib"] = max(usages) if usages else None
        out["vram_budget_mib"] = vram.get("budget_mib")
        out["vram_nominal_mib"] = vram.get("nominal_total_mib")
        out["vram_source"] = vram.get("source")
        return out

    def caps(self, current=None):
        cur = current or self.current()
        ram_cap = cur["ram_avail_mib"] * self.ram_ratio
        # plan 16.5: DXGI Budget - CurrentUsage is the honest free VRAM;
        # when only the perf-counter fallback is available the approximation
        # (nominal size minus live usage) comes through budget_mib as well.
        vram_free = cur.get("vram_budget_mib")
        if not vram_free and cur.get("vram_nominal_mib"):
            used = cur.get("vram_usage_mib") or 0.0
            vram_free = max(0.0, cur["vram_nominal_mib"] - used)
        disk_safety = (self.system_disk_safety_mib if cur.get("disk_system")
                       else self.disk_safety_mib)
        disk_cap = max(0.0, (cur.get("disk_free_mib") or 0) - disk_safety)
        return {
            "ram_cap_mib": ram_cap,
            "vram_budget_mib": cur.get("vram_budget_mib"),
            "vram_free_mib": vram_free,
            "vram_cap_mib": (vram_free or 0.0) * self.vram_ratio,
            "disk_cap_mib": disk_cap,
            "disk_safety_mib": disk_safety,
            "conservative": self.strategy == "保守",
            "sources": {"ram": "GlobalMemoryStatusEx",
                        "vram": cur.get("vram_source"),
                        "disk": shutil_disk(cur)},
        }

    def check(self, estimate):
        cur = self.current()
        caps = self.caps(cur)
        est_mib = estimate["est_mib"]
        problems = []
        if est_mib > caps["ram_cap_mib"]:
            problems.append(f"预计占用 {est_mib:.0f} MiB 超过 RAM 预算 "
                            f"{caps['ram_cap_mib']:.0f} MiB")
        if caps["vram_cap_mib"] and est_mib > caps["vram_cap_mib"] * 4:
            problems.append(f"预计显存压力超过当前使用量 4 倍 "
                            f"({caps['vram_cap_mib']:.0f} MiB)")
        if caps["disk_cap_mib"] and estimate.get("preview") \
                and estimate["preview"] / (1 << 20) > caps["disk_cap_mib"]:
            problems.append("预览缓存超过磁盘安全线")
        return {
            "ok": not problems, "est_mib": est_mib,
            "caps": caps, "problems": problems,
            "hint": "；".join(problems) if problems else "预算充足",
        }

    def shrink(self, in_w, in_h, out_w, out_h, frames, fps,
               depth_on=False):
        """Plan 16.9 ordered shrinkage; returns adjusted estimator params."""
        q, prev, cad = self.slots0, self.scale0, self.cadence0
        for _ in range(8):
            est = self.estimate_total(in_w, in_h, out_w, out_h, frames, fps,
                                      queue_slots=q, preview_scale=prev,
                                      depth_cadence=cad, depth_on=depth_on)
            r = self.check(est)
            if r["ok"]:
                return {"queue_slots": q, "preview_scale": prev,
                        "depth_cadence": cad, "est_mib": est["est_mib"]}
            if q > 3:
                q -= 1
            elif prev > 0.25:
                prev = round(prev * 0.5, 2)
            elif cad < 4:
                cad += 2
            else:
                raise BudgetExceeded("资源预算不足，无法启动（已依次收缩"
                                     "队列/预览/深度节奏）")
        raise BudgetExceeded("资源预算不足")


def cli_main():
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    default_work = os.environ.get(
        "XESS_RT_WORK",
        os.path.normpath(os.path.join(here, "..", "..", "..", "..", "work")))
    budget = MemoryBudget(default_work, strategy="自动")
    print(json.dumps(budget.current(), ensure_ascii=False, indent=2))
    est = budget.estimate_total(1920, 1080, 3840, 2160, 240, fps=30,
                                depth_on=False)
    print(json.dumps(budget.check(est), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli_main()
