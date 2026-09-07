"""Exercise real sidecar serialization, bridge streaming and source timing."""
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from media_timeline import probe_cfr
from motion_core import DisFlow
from prepare_common import shared_source_pts
from shared_fg_prepare import run
from shared_motion_io import HEADER, SidecarPacket, read_sidecar, write_sidecar
from stream_protocol import Flags, read_packet


class SharedCpuTransportTests(unittest.TestCase):
    def test_sidecar_writes_contiguous_buffers_without_array_bytes_copies(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("shared_motion_io.Path.open") as opened, \
             patch("shared_motion_io.os.fsync"), patch("shared_motion_io.os.replace"):
            packet = self.packet(0)
            write_sidecar(directory, packet)
            payloads = [call.args[0] for call in opened.return_value.__enter__.return_value.write.call_args_list]
            self.assertIsInstance(payloads[0], bytes)
            self.assertEqual(len(payloads[0]), HEADER.size)
            self.assertEqual(len(payloads), 5)
            self.assertTrue(all(isinstance(value, memoryview) for value in payloads[1:]))
            self.assertEqual(sum(value.nbytes for value in payloads[1:]), 32 * 24 * 17)

    def packet(self, index, confidence=True, pts=None):
        flow = np.full((24, 32, 2), .1, np.float32)
        flow[..., 1] = -.2
        if index == 3:
            flow += .03
        return SidecarPacket(index, 32, 24, max(0, index - 1), index if pts is None else pts,
                             24, int(Flags.RESET if index == 0 else Flags.NONE), flow,
                             np.full((24, 32), .4, np.float32),
                             np.full((24, 32), 9, np.uint8),
                             np.full((24, 32), .9, np.float32) if confidence else None)

    def test_sidecar_preserves_nonconstant_arrays_and_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.packet(1)
            original.confidence[2, 3] = .23
            path = write_sidecar(directory, original)
            restored = read_sidecar(path)
            for name in ("flow", "depth", "mask", "confidence"):
                np.testing.assert_array_equal(getattr(original, name), getattr(restored, name))
            self.assertEqual(path.stat().st_size, HEADER.size + 32 * 24 * 17)

    def test_sidecar_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_sidecar(directory, self.packet(1))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                read_sidecar(path)

    def bridge(self, window=2, missing_confidence=False, bad_pts=False, short=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for index in range(8):
            write_sidecar(directory.name, self.packet(index, not missing_confidence,
                          0 if bad_pts and index == 2 else None), slots=8)
        frames = bytes(8 * 48 * 40 * 3 - (1 if short else 0))
        output = io.BytesIO()
        args = SimpleNamespace(width=48, height=40, frames=8, timeout=.05,
                               motion_dir=directory.name, motion_window=window,
                               temporal_motion_strength=.65, temporal_depth_strength=.18,
                               report="")
        with patch("shared_fg_prepare.sys.stdin", SimpleNamespace(buffer=io.BytesIO(frames))), \
             patch("shared_fg_prepare.sys.stdout", SimpleNamespace(buffer=output)), \
             patch("shared_fg_prepare.sys.stderr", io.StringIO()):
            run(args)
        counts = json.loads(Path(directory.name, "consumer_counters.json").read_text())
        return output.getvalue(), counts

    def test_bridge_counts_actual_consumers_and_axis_geometry(self):
        raw, counts = self.bridge()
        self.assertEqual(counts["source_pair_analysis_count"], 0)
        self.assertEqual(counts["directional_dispatch_count"], 0)
        self.assertEqual(counts["sr_consume_count"], 7)
        self.assertEqual(counts["fg_consume_count"], 7)
        packet = read_packet(io.BytesIO(raw))
        flow = np.frombuffer(packet.motion, np.float32).reshape(40, 48, 2)
        np.testing.assert_allclose(flow[..., 0], .15, rtol=1e-6)
        np.testing.assert_allclose(flow[..., 1], -1/3, rtol=1e-6)
        self.assertEqual(packet.mask, b"")

    def test_five_frame_bridge_drains_all_frames_and_uses_source_confidence(self):
        _, counts = self.bridge(window=5)
        self.assertEqual(counts["fg_packets_emitted"], 8)
        self.assertEqual(counts["motion_window_applied"], 5)
        self.assertEqual(counts["lookahead_frames"], 2)
        self.assertTrue(counts["confidence_from_source"])

    def test_five_frame_refuses_missing_confidence(self):
        with self.assertRaisesRegex(RuntimeError, "requires source confidence"):
            self.bridge(window=5, missing_confidence=True)

    def test_bridge_rejects_pts_gap_and_early_color_eof(self):
        with self.assertRaisesRegex(RuntimeError, "non-adjacent PTS"):
            self.bridge(bad_pts=True)
        with self.assertRaises(EOFError):
            self.bridge(short=True)

    def test_actual_dis_direction_calls_are_counted(self):
        rng = np.random.default_rng(6)
        previous = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        for bidi in (False, True):
            engine = DisFlow(bidi)
            _, forward, _ = engine.infer(previous, np.roll(previous, 2, axis=1))
            self.assertEqual(engine.directional_dispatch_count, 2 if bidi else 1)
            self.assertEqual(forward is not None, bidi)

    def test_rational_pts_probe_sorts_decode_order_and_rejects_vfr(self):
        metadata = {"streams": [{"width": 1080, "height": 1920, "avg_frame_rate": "60000/1001",
                                  "r_frame_rate": "60000/1001", "time_base": "1/60000"}],
                    "packets": [{"pts": n} for n in (0, 2002, 1001, 3003)]}
        with patch("media_timeline.Path.is_file", return_value=True), \
             patch("media_timeline.subprocess.check_output", return_value=json.dumps(metadata)):
            result = probe_cfr("input.mp4", "ffmpeg.exe")
        self.assertEqual(Fraction(result["fps_rational"]), Fraction(60000, 1001))
        self.assertEqual(result["source_pts_ticks"], [0, 1001, 2002, 3003])
        metadata["packets"][-1]["pts"] = 4004
        with patch("media_timeline.Path.is_file", return_value=True), \
             patch("media_timeline.subprocess.check_output", return_value=json.dumps(metadata)), \
             self.assertRaisesRegex(RuntimeError, "VFR source is unsupported"):
            probe_cfr("input.mp4", "ffmpeg.exe")

    def test_cfr_rate_uses_actual_pts_when_last_packet_shortens_average(self):
        metadata = {"streams": [{"width": 864, "height": 480, "avg_frame_rate": "1464/121",
                                  "r_frame_rate": "12/1", "time_base": "1/12288"}],
                    "packets": [{"pts": index * 1024} for index in range(122)]}
        with patch("media_timeline.Path.is_file", return_value=True), \
             patch("media_timeline.subprocess.check_output", return_value=json.dumps(metadata)):
            result = probe_cfr("input.mp4", "ffmpeg.exe", max_frames=8)
        self.assertEqual(Fraction(result["fps_rational"]), Fraction(12))
        self.assertEqual(result["frames"], 8)

    def test_shared_pts_retains_nonzero_origin_and_original_tick_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "pts.json")
            path.write_text(json.dumps({"source_pts_ticks": [-7, 35, 76],
                                        "source_time_base": "1/1000"}))
            points, denominator = shared_source_pts(SimpleNamespace(
                source_pts_file=str(path), frames=3, pts_num=1, pts_den=24))
            self.assertEqual(points, [-7, 35, 76])
            self.assertEqual(denominator, 1000)
            with self.assertRaisesRegex(ValueError, "invalid shared source PTS"):
                shared_source_pts(SimpleNamespace(source_pts_file=str(path), frames=2))


if __name__ == "__main__":
    unittest.main()
