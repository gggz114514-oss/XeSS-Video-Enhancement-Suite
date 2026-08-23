from __future__ import annotations

import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "pipeline"))

import run_fg  # noqa: E402
import run_xess  # noqa: E402


def _args(**overrides):
    values = dict(
        depth_temporal=0.25, flow_consistency=1.5, mv_dilate=1,
        depth_edge=0.04, responsive_max=0.8, flow_scale=None,
        flow_resolution="auto720", force_depth=False, depth_model="depth.xml",
        depth_device="GPU", device=-1, verbose=False, depth="ai",
        overlay_mask="", temporal_motion_strength=0.65,
        temporal_depth_strength=0.18,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


SR_SETTINGS = {
    "flow": "sea-raft", "bidirectional": True, "mv_path": "lowres-depth",
    "responsive": True,
}
FG_SETTINGS = {
    "flow": "sea-raft", "bidirectional": True, "motion_window": 5,
}


class FlowPolicyCommandTests(unittest.TestCase):
    def _commands(self, args):
        sr, _ = run_xess.prep_command(
            args, SR_SETTINGS, "python.exe", 2592, 1440, 3, stream=True)
        fg = run_fg.prep_command(
            args, FG_SETTINGS, "python.exe", 2592, 1440, 3, stream=True)
        return sr, fg

    def test_auto_2k_passes_half_scale(self):
        for command in self._commands(_args()):
            self.assertIn("--flow-scale", command)
            index = command.index("--flow-scale")
            self.assertEqual(float(command[index + 1]), 0.5)

    def test_native_mode_is_explicit(self):
        for command in self._commands(_args(flow_resolution="native")):
            self.assertNotIn("--flow-scale", command)
            self.assertIn("--flow-resolution", command)
            self.assertEqual(command[command.index("--flow-resolution") + 1], "native")

    def test_explicit_one_scale_overrides_auto(self):
        for command in self._commands(_args(flow_scale=1.0)):
            self.assertIn("--flow-scale", command)
            index = command.index("--flow-scale")
            self.assertEqual(float(command[index + 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
