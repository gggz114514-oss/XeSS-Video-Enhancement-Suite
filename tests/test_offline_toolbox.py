import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import offline_toolbox as box
from offline_encoders import video_options


class OfflineContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "input.mp4"
        self.source.write_bytes(b"fixture")
        self.request = dict(input=str(self.source), output=str(self.root / "out.mp4"))

    def test_five_routes_all_three_modes(self):
        routes = box.capabilities()["backends"]
        self.assertEqual(len(routes), 5)
        self.assertTrue(all(r["modes"] == ["sr", "fg", "sr-fg"] for r in routes))

    def test_default_is_lite_block_not_cpu(self):
        r = box.validate_request(self.request)
        self.assertEqual(r["backend"], "gpu-block")
        self.assertEqual(r["encoder"], "h264_qsv")
        self.assertFalse(r["anti_stripe"])

    def test_no_silent_effect(self):
        for key in ("sharpen", "five_frame", "anti_stripe"):
            with self.subTest(key=key), self.assertRaises(box.OfflineError):
                box.validate_request(dict(self.request, backend="intel-vpl-ai", **{key: True}))

    def test_gpu_effects_available_only_on_sr_routes(self):
        for route in ("gpu-block", "gpu-dis", "amd-of"):
            for mode in ("sr", "sr-fg"):
                r = box.validate_request(dict(self.request, backend=route, mode=mode,
                                             five_frame=True, anti_stripe=True))
                self.assertTrue(r["five_frame"] and r["anti_stripe"])
            for key in ("five_frame", "anti_stripe"):
                with self.assertRaises(box.OfflineError):
                    box.validate_request(dict(self.request, backend=route, mode="fg", **{key: True}))

    def test_gpu_effects_reach_actual_native_command(self):
        for route in ("gpu-block", "gpu-dis", "amd-of"):
            r = box.validate_request(dict(self.request, backend=route, mode="sr-fg",
                                         five_frame=True, anti_stripe=True))
            p = dict(applied=r, codec="h264", source_frames=12, source_fps="24", output_width=1296, output_height=720)
            cmd = box.native_command(p, box.runtime_paths(self.root), self.root, "cancel")
            self.assertEqual(cmd[cmd.index("--gpu-five-frame")+1], "on")
            self.assertEqual(cmd[cmd.index("--gpu-anti-stripe")+1], "on")
            self.assertEqual(cmd[cmd.index("--gpu-post")+1], "off")

    def test_amd_sharpen_is_terminal_once_in_each_mode(self):
        for mode in ("sr", "fg", "sr-fg"):
            r = box.validate_request(dict(self.request, backend="amd-of", mode=mode, sharpen=True))
            p = dict(applied=r, codec="h264", source_frames=12, source_fps="24", output_width=1296, output_height=720)
            cmd = box.native_command(p, box.runtime_paths(self.root), self.root, "cancel")
            self.assertEqual(cmd.count("--gpu-post"), 1)
            self.assertEqual(cmd[cmd.index("--gpu-post")+1], "on")
            self.assertEqual(cmd[cmd.index("--motion-backend")+1], "amd-of")

    def test_amd_fusion_reserves_five_slots(self):
        r = box.validate_request(dict(self.request, backend="amd-of", five_frame=True))
        p = dict(applied=r, codec="h264", source_frames=12, source_fps="24", output_width=1296, output_height=720)
        cmd = box.native_command(p, box.runtime_paths(self.root), self.root, "cancel")
        self.assertEqual(cmd[cmd.index("--slots")+1], "5")

    def test_native_effects_reuse_correct_flow_and_have_distinct_descriptors(self):
        src = Path(__file__).resolve().parents[1] / "src/realtime"
        header = (src / "native_sr_effects.h").read_text(encoding="utf-8")
        self.assertIn("(index*4+pass)*16", header)
        self.assertIn(".forward_flow.Get()", header)
        self.assertNotIn(".backward_flow.Get()", header)
        self.assertIn("if(reset){history_count=0;", header)
        self.assertIn("history_count=std::min(4u", header)

    def test_no_silent_software_encoder_fallback(self):
        for route in ("gpu-block", "gpu-dis", "amd-of"):
            request = box.validate_request(dict(self.request, backend=route, encoder="libx264"))
            self.assertEqual(request["backend"], route)
            self.assertEqual(request["encoder"], "libx264")
        with self.assertRaises(box.OfflineError):
            box.validate_request(dict(self.request, encoder="nonexistent"))

    def test_cpu_and_intel_software_encoders(self):
        for route in ("cpu-dis", "intel-vpl-ai"):
            for encoder in ("h264_qsv", "hevc_qsv", "libx264", "libx265"):
                self.assertEqual(box.validate_request(dict(self.request, backend=route, encoder=encoder))["encoder"], encoder)

    def test_nan_inf_and_bool_rejected(self):
        for scale in (float("nan"), float("inf"), True, .9, 4.1):
            with self.subTest(scale=scale), self.assertRaises(box.OfflineError):
                box.validate_request(dict(self.request, scale=scale))

    def test_boolean_not_truthy_string(self):
        with self.assertRaises(box.OfflineError):
            box.validate_request(dict(self.request, sharpen="false"))

    def test_source_never_overwritten(self):
        with self.assertRaises(box.OfflineError):
            box.validate_request(dict(self.request, output=str(self.source)))
        self.assertEqual(self.source.read_bytes(), b"fixture")

    def test_existing_output_never_overwritten(self):
        Path(self.request["output"]).write_text("owned")
        with self.assertRaises(box.OfflineError):
            box.validate_request(self.request)

    def test_ffv1_requires_mkv(self):
        with self.assertRaises(box.OfflineError):
            box.validate_request(dict(self.request, backend="cpu-dis", encoder="ffv1"))
        r = box.validate_request(dict(self.request, backend="cpu-dis", encoder="ffv1",
                                      output=str(self.root / "lossless.mkv")))
        self.assertEqual(r["encoder"], "ffv1")

    def test_missing_runtime_reported(self):
        with self.assertRaisesRegex(box.OfflineError, "缺少文件"):
            box.plan(self.request, self.root / "absent")

    def test_cancelled_before_backend_start(self):
        cancelled = self.root / "cancel"
        cancelled.touch()
        with self.assertRaises(box.Cancelled):
            box.run_process([sys.executable, "-c", "raise Exception('must not run')"],
                            cwd=self.root, environment=os.environ.copy(), log=self.root / "log",
                            cancelled=cancelled.exists)

    def test_stderr_is_drained_to_log(self):
        log = self.root / "big.log"
        box.run_process([sys.executable, "-c", "import sys; sys.stderr.write('x'*300000)"],
                        cwd=self.root, environment=os.environ.copy(), log=log, cancelled=lambda: False, timeout=15)
        self.assertEqual(log.stat().st_size, 300000)

    def test_worker_failure_includes_root_error(self):
        with self.assertRaisesRegex(box.OfflineError, "missing_actual_dependency"):
            box.run_process([sys.executable, "-c", "raise RuntimeError('missing_actual_dependency')"],
                            cwd=self.root, environment=os.environ.copy(), log=self.root / "error.log",
                            cancelled=lambda: False, timeout=15)

    def test_status_failure_never_complete(self):
        result = box.run_job(self.request, self.root / "absent", self.root / "work")
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "failed")
        self.assertFalse(Path(self.request["output"]).exists())

    def test_native_uses_once_source_and_lite(self):
        p = dict(applied=box.validate_request(self.request), codec="h264", source_frames=243,
                 source_fps="24000/1001", output_width=1296, output_height=720)
        cmd = list(map(str, box.native_command(p, box.runtime_paths(self.root), self.root, self.root / "cancel")))
        self.assertEqual(cmd[cmd.index("--motion-scale")+1], ".5")
        self.assertEqual(cmd[cmd.index("--motion-repair")+1], "refine")
        self.assertNotIn("--depth-dir", cmd)
        self.assertNotIn("/", cmd[cmd.index("--fps")+1])

    def test_encoder_does_not_change_legacy_default(self):
        class Args:
            encoder_preset = "slow"
            crf = 16
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn("libx264", video_options(Args()))
        with patch.dict(os.environ, {"XESS_OUTPUT_ENCODER": "ffv1"}):
            self.assertIn("bgr0", video_options(Args()))

    def test_atomic_json_no_partial(self):
        path = self.root / "status.json"
        box.atomic_json(path, {"status": "complete"})
        self.assertEqual(json.loads(path.read_text())["status"], "complete")
        self.assertFalse(list(self.root.glob("*.partial")))


if __name__ == "__main__":
    unittest.main()
