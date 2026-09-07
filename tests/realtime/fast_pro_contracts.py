"""GPU-free contract tests for Fast Pro B0-B2."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from pathlib import Path

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(TESTS, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fast_pro import (  # noqa: E402
    FAST_PRO_SR_MAX_ALIGNED_INPUT_HEIGHT, FastProError, MediaInfo, build_commands,
    expected_output_frames, parse_rate, probe_media, reject_unsupported_media,
    reject_unsupported_sr_geometry, run_stream_chain, validate_output,
)
from fast_pro_capability import build_report, parse_probe  # noqa: E402


class TestRatesAndSemantics(unittest.TestCase):
    def test_common_fractional_rates_stay_exact(self):
        self.assertEqual(parse_rate("24000/1001"), Fraction(24000, 1001))
        self.assertEqual(parse_rate("29.97"), Fraction(2997, 100))

    def test_fi_is_2n_and_xefg_is_separate_2n_minus_1(self):
        self.assertEqual(expected_output_frames("sr", 7), 7)
        self.assertEqual(expected_output_frames("fi", 7), 14)
        # This assertion documents the boundary rather than making the
        # executor accept XeFG's count by accident.
        self.assertNotEqual(expected_output_frames("fi", 7), 2 * 7 - 1)

    def test_rejects_hdr_and_p010(self):
        common = dict(path="x", width=1920, height=1080, frames=10,
                      fps_num=30, fps_den=1, duration_s=1 / 3,
                      is_vfr=False, audio_streams=0, color_range=None,
                      color_space="bt2020nc", color_transfer="smpte2084",
                      color_primaries="bt2020", rotation=None)
        with self.assertRaises(FastProError):
            reject_unsupported_media(MediaInfo(pix_fmt="p010le", **common))

    def test_sr_geometry_is_measured_and_fail_closed(self):
        info = MediaInfo("landscape.mp4", 864, 480, 8, 24, 1, 1 / 3,
                         False, 0, "yuv420p", None, "bt709", "bt709",
                         "bt709", None)
        reject_unsupported_sr_geometry(info, 1296, 720)
        with self.assertRaises(FastProError):
            reject_unsupported_sr_geometry(info, 1080, 600)
        tall = MediaInfo("tall.mp4", 1080, FAST_PRO_SR_MAX_ALIGNED_INPUT_HEIGHT,
                         8, 24, 1, 1 / 3, False, 0, "yuv420p", None,
                         "bt709", "bt709", "bt709", None)
        reject_unsupported_sr_geometry(tall, 1512, 2016)
        too_tall = MediaInfo("too-tall.mp4", 1080, 1442, 8, 24, 1, 1 / 3,
                             False, 0, "yuv420p", None, "bt709", "bt709",
                             "bt709", None)
        with self.assertRaises(FastProError):
            reject_unsupported_sr_geometry(too_tall, 1512, 2019)

    def test_commands_have_nv12_and_no_rgb_or_software_fallback(self):
        info = MediaInfo("in.mp4", 1280, 720, 12, 24000, 1001, 0.5, False,
                         1, "yuv420p", None, "bt709", "bt709", "bt709", None)
        decode, vpp, encode = build_commands(
            "fi", info, 1280, 720, ffmpeg="ffmpeg", vpl="vpl-ai-vpp.exe",
            source="in.mp4", partial="out.partial.mp4")
        combined = " ".join(decode + vpp + encode).lower()
        self.assertIn("nv12", combined)
        self.assertNotIn("rgb24", combined)
        self.assertIn("h264_qsv", combined)
        _, _, software = build_commands("sr", info, 1920, 1080, ffmpeg="ffmpeg",
                           vpl="vpl", source="in", partial="out.mp4", encoder="libx264")
        self.assertIn("libx264", software)
        self.assertIn("yuv420p", software)
        self.assertNotIn("h264_qsv", software)
        with self.assertRaises(FastProError):
            build_commands("sr", info, 1920, 1080, ffmpeg="ffmpeg",
                           vpl="vpl", source="in", partial="out.mp4", encoder="ffv1")


class TestProbeAndCapability(unittest.TestCase):
    def test_probe_parser_requires_query_and_init(self):
        text = """
Runtime API version                          2.17
Implementation flags                         0x00000302
AI SR default       [video] Query           MFX_ERR_NONE (0)
AI SR default       [video] Init            MFX_ERR_NONE (0)
AI FI default       [video] Query           MFX_ERR_UNSUPPORTED (-3)
AI FI default       [video] Init            MFX_ERR_NONE (0)
"""
        parsed = parse_probe(text)
        self.assertEqual(parsed["runtime_api_version"], "2.17")
        self.assertFalse(parsed["query_init_all_supported"])
        sr = parsed["features"][0]
        self.assertEqual(sr["api_extension"], "MFX_EXTBUFF_VPP_AI_SUPER_RESOLUTION")
        self.assertFalse(parsed["probe_complete"])
        self.assertTrue(sr["query_init_supported"])

    def test_probe_parser_rejects_truncated_positive_probe(self):
        text = """
Runtime API version                          2.17
Implementation flags                         0x00000302
AI SR default       [video] Query           MFX_ERR_NONE (0)
AI SR default       [video] Init            MFX_ERR_NONE (0)
"""
        parsed = parse_probe(text)
        self.assertFalse(parsed["probe_complete"])
        self.assertFalse(parsed["query_init_all_supported"])

    def test_capability_report_marks_a770_unmeasured_and_identity_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.exe"
            probe.write_bytes(b"probe")
            report = build_report(
                "AI SR default [video] Query MFX_ERR_NONE (0)\n"
                "AI SR default [video] Init MFX_ERR_NONE (0)\n",
                probe_path=str(probe), gpu="Intel Arc B580",
                driver="32.0.test", onevpl="2.17")
        self.assertTrue(report["B580"]["tested"])
        self.assertFalse(report["A770"]["tested"])
        self.assertEqual(report["A770"]["status"], "not_measured")
        self.assertEqual(report["B580"]["ai_identity"]["classification"], "not_run")
        self.assertFalse(report["probe"]["probe_complete"])
        self.assertFalse(report["probe"]["query_init_all_supported"])
        self.assertFalse(report["B580"]["ai_identity"]["ordinary_resize_or_frame_copy_silently_allowed"])


class TestManagedPipe(unittest.TestCase):
    def _child(self, code):
        return [sys.executable, "-c", code]

    def test_stderr_is_drained_and_backend_marker_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            decode = self._child("import sys; sys.stdout.buffer.write(b'x'*4096)")
            vpp = self._child(
                "import sys; d=sys.stdin.buffer.read(); "
                "sys.stderr.write('backend=intel-vpl-ai\\n'); "
                "sys.stdout.buffer.write(d)")
            encode = self._child("import sys; d=sys.stdin.buffer.read(); "
                                 "sys.stderr.write('encoder-ok\\n')")
            result = run_stream_chain(decode, vpp, encode, log_dir=tmp)
            self.assertTrue(result["backend_marker"])
            self.assertEqual(result["output_bytes"], 4096)
            self.assertIsNotNone(result["first_chunk_s"])
            self.assertIn("encoder-ok", result["stderr"]["encode"])

    def test_cancel_terminates_chain_promptly(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = threading.Event()
            decode = self._child("import time; time.sleep(30)")
            vpp = self._child("import sys; sys.stdin.buffer.read()")
            encode = self._child("import sys; sys.stdin.buffer.read()")
            timer = threading.Timer(0.2, event.set)
            timer.start()
            started = time.monotonic()
            try:
                with self.assertRaises(FastProError):
                    run_stream_chain(decode, vpp, encode, log_dir=tmp,
                                     cancel_event=event)
            finally:
                timer.cancel()
            self.assertLess(time.monotonic() - started, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
