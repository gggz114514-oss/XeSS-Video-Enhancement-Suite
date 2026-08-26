from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "pipeline"))

import stage_timer  # noqa: E402


class TimingRequestedTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(stage_timer.timing_requested())
            self.assertFalse(stage_timer.StageTimer().on)

    def test_flag_or_env_enables(self) -> None:
        self.assertTrue(stage_timer.timing_requested(True))
        with mock.patch.dict(os.environ, {stage_timer.TIMING_ENV: "1"}, clear=True):
            self.assertTrue(stage_timer.timing_requested())
        for off in ("", "0", "false", "OFF"):
            env = {stage_timer.TIMING_ENV: off}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertFalse(stage_timer.timing_requested())


class StageTimerTests(unittest.TestCase):
    def test_disabled_timer_records_nothing(self) -> None:
        timer = stage_timer.StageTimer(False)
        with timer.span("sharpen"):
            pass
        timer.observe("worker_read_wait", 0.5)
        stream = io.StringIO()
        self.assertEqual(timer.report("prepare-sr", stream=stream), {})
        self.assertEqual(stream.getvalue(), "")
        self.assertEqual(timer.totals, {})

    def test_enabled_timer_reports_one_json_line(self) -> None:
        timer = stage_timer.StageTimer(True)
        with timer.span("sharpen"):
            pass
        with timer.span("sharpen"):
            pass
        timer.observe("worker_read_wait", 0.25)
        stream = io.StringIO()
        payload = timer.report("prepare-sr", stream=stream)
        line = stream.getvalue()
        self.assertGreaterEqual(payload["sharpen_s"], 0.0)
        self.assertEqual(payload["calls"], {"sharpen": 2, "worker_read_wait": 1})
        self.assertIn("[timing] component=prepare-sr", line)
        parsed = json.loads(line.split("component=prepare-sr ", 1)[1])
        self.assertIn("worker_read_wait_s", parsed)


if __name__ == "__main__":
    unittest.main()
