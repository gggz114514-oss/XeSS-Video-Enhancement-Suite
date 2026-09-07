"""Pre-FG protection must use the original source and never sharpen twice."""
import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from run_pipeline import (add_edge_guard_arguments, pre_fg_guard_command,
                          resolve_edge_guard_strength)


class Guide:
    def arguments(self):
        return ["--shm-name", "test-guide", "--shm-slots", "8", "--shm-slot-size", "123"]


class PipelineEdgeGuardTests(unittest.TestCase):
    def parse_guard(self, *flags):
        parser = argparse.ArgumentParser()
        add_edge_guard_arguments(parser)
        args = parser.parse_args(flags)
        args.edge_guard_strength = resolve_edge_guard_strength(
            args.anti_stripe, args.edge_guard_strength)
        return args

    def test_default_off_creates_no_post_command(self):
        args = self.parse_guard()
        self.assertEqual(args.edge_guard_strength, 0)
        self.assertIsNone(pre_fg_guard_command(args, 20, 10, 30, 15, 9, None))

    def test_on_uses_user_approved_strength(self):
        args = self.parse_guard("--anti-stripe", "on")
        self.assertEqual(args.edge_guard_strength, .90)
        args.video = "source.mp4"
        cmd = pre_fg_guard_command(args, 20, 10, 30, 15, 9, Guide())
        self.assertEqual(cmd[cmd.index("--guard-strength") + 1], "0.9")

    def test_explicit_off_overrides_stored_strength(self):
        args = self.parse_guard("--anti-stripe", "off", "--edge-guard-strength", ".9")
        self.assertEqual(args.edge_guard_strength, 0)
        self.assertIsNone(pre_fg_guard_command(args, 20, 10, 30, 15, 9, None))

    def test_legacy_explicit_strength_is_preserved(self):
        for value in (0, .75, .90, 1):
            with self.subTest(value=value):
                self.assertEqual(self.parse_guard(
                    "--edge-guard-strength", str(value)).edge_guard_strength, value)

    def test_invalid_or_nonfinite_strength_rejected(self):
        for value in (-.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_edge_guard_strength("on", value)

    def test_on_zero_conflict_rejected(self):
        with self.assertRaises(ValueError):
            self.parse_guard("--anti-stripe", "on", "--edge-guard-strength", "0")

    def test_off_has_no_stage_or_guide(self):
        self.assertIsNone(pre_fg_guard_command(
            SimpleNamespace(edge_guard_strength=0), 20, 10, 30, 15, 9, None))

    def test_guard_never_inherits_terminal_sharpen(self):
        args = SimpleNamespace(edge_guard_strength=.75, video="source.mp4",
                               sharpen_mode="adaptive", final_sharpen=.7)
        cmd = pre_fg_guard_command(args, 20, 10, 30, 15, 9, Guide())
        self.assertEqual(cmd[cmd.index("--sharpen-mode") + 1], "off")
        self.assertNotIn("--static", cmd)
        self.assertNotIn("--motion", cmd)
        self.assertEqual(cmd[cmd.index("--video") + 1], "source.mp4")
        self.assertEqual(cmd[cmd.index("--frames") + 1], "9")
        self.assertEqual(cmd[cmd.index("--guide-shm-name") + 1], "test-guide")
        self.assertNotIn("--shm-name", cmd)

    def test_output_and_source_dimensions_are_separate(self):
        args = SimpleNamespace(edge_guard_strength=.5, video="source.mp4")
        cmd = pre_fg_guard_command(args, 17, 13, 32, 24, 2, Guide())
        for key, expected in (("--width", "32"), ("--height", "24"),
                              ("--in-w", "17"), ("--in-h", "13"),
                              ("--guard-strength", "0.5")):
            self.assertEqual(cmd[cmd.index(key) + 1], expected)


if __name__ == "__main__":
    unittest.main()
