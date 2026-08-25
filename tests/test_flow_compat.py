"""Retired SEA-RAFT options must keep old workflows running on native Fast DIS."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))
sys.path.insert(0, os.fspath(ROOT / "pipeline"))


def _install_torch_stub_if_needed() -> None:
    """Let the dependency-light CI run import xess_nodes without torch.

    xess_nodes only touches torch inside function bodies; a minimal symbol
    module is enough to satisfy the top-level import.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = type("Tensor", (), {})
        torch_stub.is_tensor = lambda value: False
        sys.modules["torch"] = torch_stub


_install_torch_stub_if_needed()

import xess_nodes  # noqa: E402
import run_fg  # noqa: E402
import run_xess  # noqa: E402
import prepare_common  # noqa: E402


class NodeFlowCompatTests(unittest.TestCase):
    def setUp(self) -> None:
        xess_nodes._MIGRATION_PRINTED = False

    def test_presets_no_longer_reference_sea_raft(self) -> None:
        for presets in (xess_nodes.SR_PRESETS, xess_nodes.FG_PRESETS):
            for preset in presets.values():
                self.assertEqual(preset["flow"], "dis")
                self.assertFalse(preset["bidirectional"])

    def test_legacy_flow_values_migrate_to_dis_fast_with_notice(self) -> None:
        preset = {"flow": "dis", "bidirectional": False}
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stderr):
            flow, bidirectional = xess_nodes._resolve_flow(preset, "sea-raft")
        self.assertEqual((flow, bidirectional), ("dis", False))
        self.assertIn("SEA-RAFT", stderr.getvalue())
        # The migration notice is printed once per process.
        with contextlib.redirect_stdout(io.StringIO()):
            again, _ = xess_nodes._resolve_flow(preset, "sea-raft-single")
        self.assertEqual(again, "dis")

    def test_canonical_maps_chinese_legacy_labels_for_old_workflows(self) -> None:
        self.assertEqual(xess_nodes._canonical("SEA-RAFT 双向", xess_nodes.FLOW_VALUES), "sea-raft")
        self.assertEqual(xess_nodes._canonical("sea-raft-single", xess_nodes.FLOW_VALUES),
                         "sea-raft-single")
        self.assertEqual(xess_nodes._canonical("DIS 极速", xess_nodes.FLOW_VALUES), "dis-fast")

    def test_ui_choices_no_longer_offer_sea_raft(self) -> None:
        self.assertFalse(any("SEA-RAFT" in choice for choice in xess_nodes.FLOW_CHOICES))


def _driver_args(module, *, preset="quality", flow_mode="auto"):
    namespace = argparse.Namespace(preset=preset, flow_mode=flow_mode)
    if module is run_xess:
        namespace.mv_path = "auto"
        namespace.responsive_mask = "auto"
        namespace.sharpen = None
        namespace.sharpen_mode = "auto"
        namespace.sharpen_static = None
        namespace.sharpen_motion = None
        namespace.io_mode = "stream"
    else:
        namespace.final_sharpen = None
        namespace.sharpen_mode = "auto"
        namespace.sharpen_static = None
        namespace.sharpen_motion = None
        namespace.motion_window = "auto"
        namespace.io_mode = "stream"
    return namespace


class DriverFlowCompatTests(unittest.TestCase):
    def setUp(self) -> None:
        run_xess._MIGRATION_PRINTED = False
        run_fg._MIGRATION_PRINTED = False

    def test_run_xess_preset_quality_is_dis_one_way_highres(self) -> None:
        settings = run_xess.resolve_settings(_driver_args(run_xess))
        self.assertEqual(settings["flow"], "dis")
        self.assertFalse(settings["bidirectional"])
        self.assertEqual(settings["mv_path"], "highres")

    def test_run_xess_legacy_flow_mode_maps_to_dis_fast_once(self) -> None:
        first = io.StringIO()
        with contextlib.redirect_stdout(first):
            settings = run_xess.resolve_settings(_driver_args(run_xess, flow_mode="sea-raft"))
        self.assertEqual((settings["flow"], settings["bidirectional"]), ("dis", False))
        self.assertIn("Fast DIS", first.getvalue())
        # The migration notice is printed once per process.
        second = io.StringIO()
        with contextlib.redirect_stdout(second):
            run_xess.resolve_settings(_driver_args(run_xess, flow_mode="sea-raft-single"))
        self.assertEqual(second.getvalue(), "")

    def test_run_xss_dis_occlusion_stays_expert_bidirectional(self) -> None:
        settings = run_xess.resolve_settings(_driver_args(run_xess, flow_mode="dis-occlusion"))
        self.assertEqual((settings["flow"], settings["bidirectional"]), ("dis", True))

    def test_run_fg_legacy_flow_mode_maps_to_dis(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            settings = run_fg.resolve_settings(_driver_args(run_fg, flow_mode="sea-raft"))
        self.assertEqual((settings["flow"], settings["bidirectional"]), ("dis", False))
        self.assertIn("SEA-RAFT has been retired", stdout.getvalue())

    def test_prepare_engine_choice_maps_to_dis_once(self) -> None:
        prepare_common._MIGRATION_PRINTED = False
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            resolved = prepare_common.resolve_engine("sea-raft")
        self.assertEqual(resolved, "dis")
        self.assertIn("Fast DIS", stderr.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(prepare_common.resolve_engine("sea-raft"), "dis")

    def test_prepare_legacy_engine_forces_bidirectional_off(self) -> None:
        prepare_common._MIGRATION_PRINTED = False
        args = argparse.Namespace(engine="sea-raft", bidirectional=True)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            prepare_common.normalize_legacy_engine(args)
        self.assertEqual((args.engine, args.bidirectional), ("dis", False))
        self.assertIn("forced off", stderr.getvalue())
        # Expert native DIS keeps explicit bidirectional.
        expert = argparse.Namespace(engine="dis", bidirectional=True)
        prepare_common._MIGRATION_PRINTED = False
        with contextlib.redirect_stderr(io.StringIO()):
            prepare_common.normalize_legacy_engine(expert)
        self.assertEqual((expert.engine, expert.bidirectional), ("dis", True))


if __name__ == "__main__":
    unittest.main()
