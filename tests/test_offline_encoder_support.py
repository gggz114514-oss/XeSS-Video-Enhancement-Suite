import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import media_validation
import offline_toolbox as box
from offline_encoders import configure_cli_encoder


class EncoderSupportTests(unittest.TestCase):
    def test_cli_default_does_not_change_legacy_encoder(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_cli_encoder(SimpleNamespace(encoder=None))
            self.assertNotIn("XESS_OUTPUT_ENCODER", os.environ)

    def test_cli_auto_and_ffv1_container(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_cli_encoder(SimpleNamespace(encoder="auto"))
            self.assertEqual(os.environ["XESS_OUTPUT_ENCODER"], "h264_qsv")
            args = SimpleNamespace(encoder="ffv1", io_mode="auto")
            configure_cli_encoder(args)
            self.assertEqual(os.environ["XESS_OUTPUT_CONTAINER"], ".mkv")
            self.assertEqual(args.io_mode, "stream")

    def test_unsupported_legacy_chunked_ffv1_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "stream"):
                configure_cli_encoder(SimpleNamespace(encoder="ffv1", io_mode="chunked"))

    def test_old_native_command_not_silently_relabelled(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "新版"):
                configure_cli_encoder(SimpleNamespace(encoder="libx265", pipeline_backend="full-gpu"))

    def test_metadata_only_remux_preserves_negative_audio_origin(self):
        import fast_pro
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "a.mkv"
            source.touch()
            with patch.object(fast_pro, "color_flags", return_value=["-color_range", "tv"]), \
                 patch.object(fast_pro, "_run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run, \
                 patch.object(fast_pro.os, "replace"):
                fast_pro._rewrite_color_metadata("ffmpeg", str(source), None)
            command = run.call_args.args[0]
            self.assertIn("-copyts", command)
            self.assertEqual(command[command.index("-avoid_negative_ts")+1], "disabled")

    def test_all_five_routes_list_all_five_encoders(self):
        for route in box.capabilities()["backends"]:
            self.assertEqual(route["encoders"], box.ALL_ENCODERS)

    def test_every_native_codec_gets_explicit_terminal_selection(self):
        for route in ("gpu-block", "gpu-dis", "amd-of"):
            for encoder in box.ALL_ENCODERS:
                with self.subTest(route=route, encoder=encoder):
                    p = dict(applied=dict(backend=route, encoder=encoder, mode="sr-fg",
                                         scale=1.5, sharpen=True, depth="ai"),
                             source_frames=48, codec="h264", source_fps="24",
                             output_width=1296, output_height=720)
                    job = Path("test-job")
                    cmd = box.native_command(p, box.runtime_paths(job), job, job / "cancel")
                    self.assertEqual(cmd[cmd.index("--terminal-encoder")+1], encoder)
                    self.assertEqual(cmd[cmd.index("--motion-backend")+1], route)
                    endpoint = box.native_encoded_path(p, job)
                    self.assertEqual(cmd[cmd.index("--qsv-out")+1], endpoint)
                    self.assertNotIn("raw", endpoint.name)
                    if encoder == "hevc_qsv":
                        self.assertEqual(endpoint.suffix, ".hevc")
                    elif encoder in ("ffv1", "libx264", "libx265"):
                        self.assertEqual(endpoint.suffix, ".mkv")

    def test_mkv_uses_actual_decoded_count_not_opencv_duration_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "ffprobe.exe"
            probe.touch()
            response = subprocess.CompletedProcess([], 0, json.dumps({"streams": [
                dict(width=1296, height=720, avg_frame_rate="24/1", nb_read_frames="48")]}), "")
            with patch.dict(os.environ, {"XESS_FFPROBE": str(probe)}), patch.object(media_validation, "_run", return_value=response) as run:
                actual = media_validation.probe_video("python", "flow.py", "padded-audio.mkv")
                self.assertEqual(actual["frames"], 48)
                self.assertIn("-count_frames", run.call_args.args[0])
                self.assertNotIn("flow.py", run.call_args.args[0])

    def test_unknown_actual_frame_count_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "ffprobe.exe"
            probe.touch()
            response = subprocess.CompletedProcess([], 0, json.dumps({"streams": [
                dict(width=320, height=180, avg_frame_rate="24/1", nb_read_frames="N/A")]}), "")
            with patch.dict(os.environ, {"XESS_FFPROBE": str(probe)}), patch.object(media_validation, "_run", return_value=response):
                with self.assertRaises(media_validation.MediaValidationError):
                    media_validation.probe_video("python", "flow.py", "bad.mkv")

    def test_rational_rate_not_rounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "ffprobe.exe"
            probe.touch()
            response = subprocess.CompletedProcess([], 0, json.dumps({"streams": [
                dict(width=320, height=180, avg_frame_rate="0/0", r_frame_rate="60000/1001", nb_read_frames="300")]}), "")
            with patch.dict(os.environ, {"XESS_FFPROBE": str(probe)}), patch.object(media_validation, "_run", return_value=response):
                result = media_validation.probe_video("python", "flow.py", "ntsc.mkv")
                self.assertAlmostEqual(result["fps"], 60000/1001)


if __name__ == "__main__":
    unittest.main()
