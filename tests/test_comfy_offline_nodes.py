from __future__ import annotations

import pathlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import comfy_offline_nodes as nodes  # noqa: E402


CAPS = {
    "ok": True,
    "schema_version": "xess.offline.capabilities.v1",
    "backends": [
        {"id": "gpu-block", "status": "supported", "modes": ["sr", "fg", "sr-fg"],
         "encoders": ["auto", "h264_qsv"],
         "effects": {"sharpen": ["sr", "fg", "sr-fg"], "five_frame": ["sr", "sr-fg"], "anti_stripe": ["sr", "sr-fg"]},
         "depth": ["ai", "constant"]},
        {"id": "cpu-dis", "status": "supported", "modes": ["sr", "fg", "sr-fg"],
         "encoders": ["auto", "libx264", "libx265", "ffv1"],
         "effects": {"sharpen": ["sr", "fg", "sr-fg"],
                      "five_frame": ["sr"], "anti_stripe": ["sr", "sr-fg"]},
         "depth": ["ai", "constant"]},
        {"id": "intel-vpl-ai", "status": "supported", "modes": ["sr", "fg", "sr-fg"],
         "encoders": ["auto", "h264_qsv", "hevc_qsv", "libx264", "libx265", "ffv1"],
         "effects": {"sharpen": [], "five_frame": [], "anti_stripe": []}, "depth": []},
        {"id": "gpu-dis", "status": "experimental", "modes": ["sr", "fg", "sr-fg"],
         "encoders": ["auto", "h264_qsv"],
         "effects": {"sharpen": ["sr", "fg", "sr-fg"], "five_frame": ["sr", "sr-fg"], "anti_stripe": ["sr", "sr-fg"]},
         "depth": ["ai", "constant"]},
        {"id": "amd-of", "status": "experimental", "modes": ["sr", "fg", "sr-fg"],
         "encoders": ["auto", "h264_qsv"],
         "effects": {"sharpen": ["sr", "fg", "sr-fg"], "five_frame": ["sr", "sr-fg"], "anti_stripe": ["sr", "sr-fg"]},
         "depth": ["ai", "constant"]},
    ],
}


class ComfyOfflineNodeContractTests(unittest.TestCase):
    def setUp(self):
        # Runtime download/install has independent fault-injection tests.
        import runtime_manager
        installer = mock.patch.object(runtime_manager, 'ensure_runtime')
        installer.start()
        self.addCleanup(installer.stop)
        test_parent = pathlib.Path(os.environ.get("XESS_COMFY_TEST_ROOT", str(ROOT / "work")))
        test_parent.mkdir(parents=True, exist_ok=True)
        self.temp_root = pathlib.Path(
            tempfile.mkdtemp(prefix="comfy-node-test-",
                             dir=str(test_parent))
        )
        self.source = self.temp_root / "input.mp4"
        self.source.write_bytes(b"not a real video; subprocess contract only")

    def tearDown(self):
        # Keep test directories for post-run inspection; this mirrors the
        # runtime's recoverable bounded workspace policy.
        pass

    def test_capability_driven_inputs_and_request_schema(self):
        with mock.patch.object(nodes, "_capabilities", return_value=CAPS):
            sr = nodes.XeSSR4OfflineSuperResolution.INPUT_TYPES()["required"]
            fg = nodes.XeSSR4OfflineFrameGeneration.INPUT_TYPES()["required"]
        self.assertIn("GPU Block（快速）", sr["backend"][0])
        self.assertIn("Intel 视频接口", fg["backend"][0])
        for choices in (sr["backend"][0], fg["backend"][0]):
            self.assertIn("GPU DIS（实验）", choices)
            self.assertIn("AMD 光流（实验）", choices)
        self.assertIn("2×", sr["scale"][0])
        request = nodes._request(
            mode="sr-fg", source=self.source,
            output=self.temp_root / "out.mp4", backend="cpu-dis", scale=1.5,
            encoder="ffv1", depth="ai", sharpen=False,
            five_frame=False, anti_stripe=True)
        self.assertEqual(request["schema_version"], nodes.SCHEMA_VERSION)
        self.assertEqual(request["mode"], "sr-fg")
        self.assertEqual(request["input"], str(self.source.resolve()))
        self.assertTrue(request["anti_stripe"])

    def test_default_backend_is_explicit_gpu_block(self):
        with mock.patch.object(nodes, "_capabilities", return_value=CAPS):
            spec = nodes.XeSSR4OfflineSuperResolution.INPUT_TYPES()["required"]["backend"]
        self.assertEqual(spec[1]["default"], "GPU Block（快速）")

    def test_missing_gpu_runtime_keeps_explicit_gpu_default_fail_closed(self):
        unavailable = {**CAPS, "backends": [dict(item, available=False) for item in CAPS["backends"]]}
        with mock.patch.object(nodes, "_capabilities", return_value=unavailable):
            spec = nodes.XeSSR4OfflineSuperResolution.INPUT_TYPES()["required"]["backend"]
        self.assertEqual(spec[1]["default"], "GPU Block（快速）")
        self.assertEqual(spec[0][0], "GPU Block（快速）")

    def test_worker_output_tail_is_bounded(self):
        tail = nodes._BoundedTail(limit=64)
        tail.append("a" * 1000)
        tail.append("final-json")
        self.assertLessEqual(len(tail.text()), 64)
        self.assertTrue(tail.text().endswith("final-json"))

    def test_default_work_root_is_not_system_temp(self):
        with mock.patch.dict(nodes.os.environ, {"XESS_OFFLINE_WORK_ROOT": ""}, clear=False), \
             mock.patch.object(nodes, "NODE_DIR", self.temp_root / "package"):
            root = nodes._work_root()
        self.assertEqual(root, (self.temp_root / "package" / "work").resolve())

    def test_plan_failure_is_explicit_and_does_not_run(self):
        plan_failure = nodes._ProcessResult(2, {"ok": False, "status": "failed",
                                                 "error": "Intel 视频接口不支持末尾锐化"}, "")
        with mock.patch.object(nodes, "_invoke_toolbox", return_value=plan_failure) as invoke:
            with self.assertRaisesRegex(nodes.OfflineNodeError, "Intel 视频接口不支持"):
                nodes.XeSSR4OfflineSuperResolution().upscale_video(
                    str(self.source), backend="Intel 视频接口", scale="1.5×",
                    sharpen=True)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(invoke.call_args.args[0], "plan")

    def test_validate_inputs_rejects_unsupported_effect_before_run(self):
        with mock.patch.object(nodes, "_capabilities", return_value=CAPS):
            message = nodes.XeSSR4OfflineSuperResolution.VALIDATE_INPUTS(
                backend="Intel 视频接口", encoder="自动", sharpen=True)
        self.assertIsInstance(message, str)
        self.assertIn("不支持", message)

    def test_amd_sr_nodes_accept_effects_without_fallback(self):
        with mock.patch.object(nodes, "_capabilities", return_value=CAPS):
            for cls in (nodes.XeSSR4OfflineSuperResolution, nodes.XeSSR4OfflineSuperResolutionFrameGeneration):
                self.assertIs(cls.VALIDATE_INPUTS(backend="AMD 光流（实验）", encoder="自动",
                              anti_stripe=True, five_frame=True, sharpen=True), True)
            self.assertIs(nodes.XeSSR4OfflineFrameGeneration.VALIDATE_INPUTS(
                backend="AMD 光流（实验）", encoder="自动", sharpen=True), True)
            self.assertIs(nodes.XeSSR4OfflineFrameGeneration.VALIDATE_INPUTS(
                backend="AMD 光流（实验）", encoder="自动", anti_stripe=True), True)

    def test_fg_exposes_only_relevant_inputs(self):
        with mock.patch.object(nodes, "_capabilities", return_value=CAPS):
            spec = nodes.XeSSR4OfflineFrameGeneration.INPUT_TYPES()
            self.assertEqual(list(spec["required"]), ["video", "backend", "encoder", "depth", "sharpen"])
            for group in spec.values():
                self.assertNotIn("five_frame", group)
                self.assertNotIn("anti_stripe", group)
            for cls in (nodes.XeSSR4OfflineSuperResolution, nodes.XeSSR4OfflineSuperResolutionFrameGeneration):
                required = cls.INPUT_TYPES()["required"]
                self.assertIn("five_frame", required)
                self.assertIn("anti_stripe", required)

    def test_legacy_fg_fields_never_reach_worker(self):
        for backend in nodes.BACKEND_LABELS.values():
            with self.subTest(backend=backend), mock.patch.object(nodes, "_capabilities", return_value=CAPS):
                self.assertIs(nodes.XeSSR4OfflineFrameGeneration.VALIDATE_INPUTS(
                    backend=backend, encoder="自动", five_frame=True, anti_stripe=True), True)
        node = nodes.XeSSR4OfflineFrameGeneration()
        with mock.patch.object(node, "_execute", return_value="ok") as execute, \
             mock.patch.object(sys, "stderr", new_callable=io.StringIO) as log:
            self.assertEqual(node.interpolate_video(str(self.source), sharpen=True,
                             five_frame=True, anti_stripe=True), "ok")
            self.assertFalse(execute.call_args.kwargs["five_frame"])
            self.assertFalse(execute.call_args.kwargs["anti_stripe"])
            self.assertTrue(execute.call_args.kwargs["sharpen"])
            self.assertIn("旧工作流迁移", log.getvalue())

    def test_fg_legacy_migration_keeps_other_validation(self):
        with mock.patch.object(nodes, "_capabilities", return_value=CAPS):
            for options in (dict(sharpen=True), dict(encoder="unsupported")):
                result = nodes.XeSSR4OfflineFrameGeneration.VALIDATE_INPUTS(
                    backend="Intel 视频接口", five_frame=True, anti_stripe=True, **options)
                self.assertIsInstance(result, str)

    def test_fg_example_matches_widget_schema(self):
        workflow = json.loads((ROOT / "workflows/r4_offline_comfy_video.json").read_text(encoding="utf-8"))
        fg = next(n for n in workflow["nodes"] if n["type"] == "XeSSR4OfflineFrameGeneration")
        with mock.patch.object(nodes, "_capabilities", return_value=CAPS):
            keys = list(nodes.XeSSR4OfflineFrameGeneration.INPUT_TYPES()["required"])
        self.assertEqual([item["name"] for item in fg["inputs"]], keys)
        self.assertEqual(len(fg["widgets_values"]), len(keys)-1)

    def test_mock_effects_match_production_contract(self):
        from pipeline.offline_toolbox import capabilities
        actual = {entry["id"]: entry["effects"] for entry in capabilities()["backends"]}
        self.assertEqual({entry["id"]: entry["effects"] for entry in CAPS["backends"]}, actual)

    def test_complete_run_requires_real_output_file(self):
        output = self.temp_root / "output.mp4"
        output.write_bytes(b"encoded output")
        plan = nodes._ProcessResult(0, {"ok": True, "plan": {"backend": "cpu-dis"}}, "")
        run = nodes._ProcessResult(0, {"ok": True, "status": "complete",
                                       "output": str(output), "message": "完成"}, "")
        with mock.patch.object(nodes, "_invoke_toolbox", side_effect=[plan, run]), \
             mock.patch.object(nodes, "_make_output_video", return_value="VIDEO"):
            result = nodes.XeSSR4OfflineFrameGeneration().interpolate_video(
                str(self.source), backend="CPU DIS（稳定）")
        self.assertEqual(len(result), len(nodes.XeSSR4OfflineFrameGeneration.RETURN_TYPES))
        video, path = result
        self.assertEqual(video, "VIDEO")
        self.assertEqual(path, str(output.resolve()))

    def test_missing_output_is_not_reported_as_success(self):
        plan = nodes._ProcessResult(0, {"ok": True, "plan": {}}, "")
        run = nodes._ProcessResult(0, {"ok": True, "status": "complete",
                                       "output": str(self.temp_root / "missing.mp4")}, "")
        with mock.patch.object(nodes, "_invoke_toolbox", side_effect=[plan, run]), \
             self.assertRaisesRegex(nodes.OfflineNodeError, "成片不存在"):
            nodes.XeSSR4OfflineFrameGeneration().interpolate_video(
                str(self.source), backend="CPU DIS（稳定）")

    def test_missing_runtime_is_chinese_and_fail_closed(self):
        with mock.patch.object(nodes, "_runtime_root", return_value=None), \
             self.assertRaisesRegex(nodes.OfflineNodeError, "未找到运行时根目录"):
            nodes.XeSSR4OfflineFrameGeneration().interpolate_video(
                str(self.source), backend="CPU DIS（稳定）")

    def test_cancel_file_terminates_and_drains_subprocess(self):
        cancel = self.temp_root / "cancel.signal"
        result_holder = []
        worker = threading.Thread(target=lambda: result_holder.append(nodes._run_command(
            [sys.executable, "-c", "import sys,time; print('progress', flush=True); "
             "sys.stderr.write('diagnostic\\n'); sys.stderr.flush(); time.sleep(30)"],
            timeout=10, cancel_file=cancel, allow_interrupt=False)))
        worker.start()
        time.sleep(0.25)
        cancel.write_text("cancel", encoding="utf-8")
        worker.join(timeout=8)
        self.assertFalse(worker.is_alive())
        result = result_holder[0]
        self.assertTrue(result.cancelled)
        self.assertIsNone(result.payload)
        self.assertIn("diagnostic", result.stderr)


if __name__ == "__main__":
    unittest.main()
