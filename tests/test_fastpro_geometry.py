# -*- coding: utf-8 -*-
"""Contract tests for measured Fast Pro geometry and vertical routing."""

from __future__ import annotations

import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))
sys.path.insert(0, os.fspath(ROOT / "pipeline"))

from fastpro_geometry import (  # noqa: E402
    SR_INPUT_ALIGNED_HEIGHT_MAX,
    SR_SCALE_MIN,
    align16,
    is_sr_direct_supported,
    plan_sr_route,
)


class Align16Test(unittest.TestCase):
    def test_align16(self):
        self.assertEqual(align16(1080), 1088)
        self.assertEqual(align16(1920), 1920)
        self.assertEqual(align16(1440), 1440)
        self.assertEqual(align16(1), 16)


class DirectSupportTest(unittest.TestCase):
    def test_measured_pass_cases(self):
        self.assertTrue(is_sr_direct_supported(1920, 1080, 2880, 1620))
        self.assertTrue(is_sr_direct_supported(1920, 1080, 3840, 2160))
        self.assertTrue(is_sr_direct_supported(1080, 1440, 1620, 2160))
        self.assertTrue(is_sr_direct_supported(1080, 1440, 2160, 2880))
        self.assertTrue(is_sr_direct_supported(960, 1440, 1440, 2160))
        self.assertTrue(is_sr_direct_supported(720, 1280, 1440, 2560))
        self.assertTrue(is_sr_direct_supported(2560, 1440, 3840, 2160))
        self.assertTrue(is_sr_direct_supported(2160, 960, 3240, 1440))

    def test_measured_fail_cases(self):
        self.assertFalse(is_sr_direct_supported(1920, 1080, 2560, 1440))
        self.assertFalse(is_sr_direct_supported(1080, 1920, 1440, 2560))
        self.assertFalse(is_sr_direct_supported(960, 540, 1280, 720))
        self.assertFalse(is_sr_direct_supported(1280, 720, 1728, 972))
        self.assertFalse(is_sr_direct_supported(2000, 1000, 2780, 1390))
        self.assertFalse(is_sr_direct_supported(1080, 1920, 1620, 2880))
        self.assertFalse(is_sr_direct_supported(1080, 1920, 2160, 3840))
        self.assertFalse(is_sr_direct_supported(1080, 1456, 1620, 2184))
        self.assertFalse(is_sr_direct_supported(2560, 1536, 3840, 2304))
        self.assertFalse(is_sr_direct_supported(1152, 1800, 1728, 2700))
        self.assertFalse(is_sr_direct_supported(960, 2160, 1440, 3240))
        self.assertTrue(is_sr_direct_supported(1440, 1440, 2160, 2160))

    def test_boundaries_are_exact(self):
        self.assertTrue(is_sr_direct_supported(1080, 1440, 1512, 2016))
        self.assertFalse(is_sr_direct_supported(1000, 1000, 1390, 1390))
        self.assertEqual(SR_INPUT_ALIGNED_HEIGHT_MAX, 1440)
        self.assertEqual(SR_SCALE_MIN, 1.4)

    def test_non_uniform_or_degenerate_rejected(self):
        self.assertFalse(is_sr_direct_supported(1920, 1080, 2880, 1080))
        self.assertFalse(is_sr_direct_supported(0, 1080, 2880, 1620))


class PlannerTest(unittest.TestCase):
    def test_portrait_1080x1920_to_1440x2560_prefers_rotate(self):
        plan = plan_sr_route(1080, 1920, 1440, 2560)
        self.assertEqual(plan.route, "rotate")
        self.assertEqual(plan.detail["rotate_in"], [1920, 1080])
        self.assertEqual(plan.detail["ai_scale"], 1.5)
        self.assertEqual(plan.detail["ai_out"], [2880, 1620])
        self.assertEqual(plan.detail["rotate_back"], [1620, 2880])
        self.assertEqual(plan.detail["final_scale"], [1440, 2560])
        self.assertTrue(plan.supported)

    def test_landscape_uses_direct(self):
        plan = plan_sr_route(1920, 1080, 2880, 1620)
        self.assertEqual(plan.route, "direct")

    def test_rotate_wins_over_preshrink_when_both_exist(self):
        plan = plan_sr_route(1080, 1920, 1440, 2560)
        self.assertEqual(plan.route, "rotate")
        plan2 = plan_sr_route(1440, 2560, 1440, 2560)
        self.assertEqual(plan2.route, "rotate")

    def test_unsupported_when_no_ai_route_fits(self):
        plan = plan_sr_route(960, 2160, 2880, 6480)
        self.assertEqual(plan.route, "unsupported")

    def test_unsupported_guides_to_xess_not_orientation(self):
        plan = plan_sr_route(2960, 2960, 5920, 5920)
        self.assertEqual(plan.route, "unsupported")
        self.assertIn("XeSS", plan.reason)
        self.assertNotIn("竖屏不支持", plan.reason)


class ProductWiringTest(unittest.TestCase):
    def _info(self, w=1080, h=1920):
        from fast_pro import MediaInfo
        return MediaInfo(path="in.mp4", width=w, height=h, frames=8,
                         fps_num=60000, fps_den=1001, duration_s=8 / 59.94,
                         is_vfr=False, audio_streams=1, pix_fmt="yuv420p",
                         color_range=None, color_space=None,
                         color_transfer=None, color_primaries=None,
                         rotation=None)

    def test_rotate_plan_commands(self):
        from fast_pro import build_commands
        plan = plan_sr_route(1080, 1920, 1440, 2560)
        decode, vpp, encode = build_commands(
            "sr", self._info(), 1440, 2560, ffmpeg="ffmpeg", vpl="vpl",
            source="in.mp4", partial="out.mp4", plan=plan)
        self.assertIn("transpose=1,format=nv12", decode)
        self.assertIn("-s", decode)
        self.assertIn("1920x1080", decode)
        vpp_text = " ".join(vpp)
        self.assertIn("--in-width 1920", vpp_text)
        self.assertIn("--in-height 1080", vpp_text)
        self.assertIn("--out-width 2880", vpp_text)
        self.assertIn("--out-height 1620", vpp_text)
        self.assertIn("transpose=2,scale=1440:2560:flags=lanczos,format=nv12",
                      " ".join(encode))
        self.assertIn("-video_size", encode)
        self.assertIn("2880x1620", " ".join(encode))

    def test_direct_plan_keeps_identity_chain(self):
        from fast_pro import build_commands
        decode, vpp, encode = build_commands(
            "sr", self._info(1920, 1080), 2880, 1620, ffmpeg="ffmpeg",
            vpl="vpl", source="in.mp4", partial="out.mp4", plan=None)
        self.assertIn("format=nv12", decode)
        self.assertNotIn("transpose", decode)
        self.assertIn("--out-width 2880", " ".join(vpp))
        self.assertIn("--out-height 1620", " ".join(vpp))
        self.assertNotIn("transpose", " ".join(encode))

    def test_unsupported_plan_fails_closed(self):
        from fast_pro import FastProError, run_fast_pro
        info = self._info(960, 2160)
        with mock.patch("fast_pro.probe_media", return_value=info):
            with self.assertRaises(FastProError) as ctx:
                run_fast_pro(mode="sr", source_path="in.mp4",
                             output_path="out.mp4", ffprobe="",
                             ffmpeg="ffmpeg", vpl="vpl", work_dir="work",
                             out_width=2880, out_height=6480)
        self.assertIn("XeSS", str(ctx.exception))
        self.assertNotIn("竖屏", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
