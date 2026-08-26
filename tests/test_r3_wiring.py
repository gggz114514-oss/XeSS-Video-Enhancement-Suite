"""The r3 speedups must be reachable through the normal SR launcher."""

from __future__ import annotations

import argparse
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import run_xess  # noqa: E402


class _FakeRing:
    name = "Local\\test-ring"
    slots = 6
    slot_size = 123456

    def arguments(self):
        return ["--shm-name", self.name, "--shm-slots", str(self.slots),
                "--shm-slot-size", str(self.slot_size)]


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        post_threads=4, edge_guard_strength=0.75, sharpen_static=None,
        sharpen_motion=None, device=-1, verbose=False, responsive_max=0.8,
        video="input.mp4",
    )


class R3LauncherWiringTests(unittest.TestCase):
    def test_auto_uses_shared_ring_at_720_output(self) -> None:
        self.assertEqual(run_xess.resolve_io_mode("auto", 480, 720), "shared")
        self.assertEqual(run_xess.resolve_io_mode("auto", 479, 719), "stream")
        self.assertEqual(run_xess.resolve_io_mode("file", 480, 720), "file")

    def test_post_command_exposes_threads_and_output_ring(self) -> None:
        settings = {"sharpen": 0.25, "sharpen_mode": "fixed",
                    "static": 0.30, "motion": 0.16}
        command = run_xess.post_command(
            _args(), settings, 864, 480, 1296, 720, 8, ring=_FakeRing())
        self.assertIn("--threads", command)
        self.assertEqual(command[command.index("--threads") + 1], "4")
        self.assertIn("--shm-name", command)
        self.assertIn("Local\\test-ring", command)

    def test_worker_command_carries_output_ring(self) -> None:
        settings = {"flow": "dis", "mv_path": "highres", "responsive": True}
        command = run_xess.xess_command(
            _args(), settings, 864, 480, 1296, 720, 8, 4,
            stream=True, ring=_FakeRing(), out_ring=_FakeRing())
        self.assertIn("--shm-name", command)
        self.assertIn("--out-shm-name", command)
        self.assertIn("--out-shm-slot-size", command)


if __name__ == "__main__":
    unittest.main()
