"""Frame timestamp filtering for the Intel oneVPL container adapter."""

from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

import fast_pro  # noqa: E402


class FastProPtsTests(unittest.TestCase):
    def test_only_explicit_video_frames_are_counted(self) -> None:
        data = {
            "frames": [
                {"codec_type": "audio", "best_effort_timestamp_time": "0.000"},
                {"codec_type": "video", "best_effort_timestamp_time": "0.000"},
                {"codec_type": "audio", "best_effort_timestamp_time": "0.021"},
                {"codec_type": "video", "best_effort_timestamp_time": "0.042"},
                {"media_type": "video", "best_effort_timestamp_time": "0.084"},
                {"best_effort_timestamp_time": "99.0"},
            ]
        }
        with patch.object(fast_pro, "_probe_json", return_value=data):
            self.assertEqual(fast_pro._frame_pts("ffprobe", "output.mp4"), [0.0, 0.042, 0.084])

    def test_missing_codec_type_is_fail_closed_for_pts(self) -> None:
        data = {"frames": [{"best_effort_timestamp_time": "0.0"}]}
        with patch.object(fast_pro, "_probe_json", return_value=data):
            self.assertEqual(fast_pro._frame_pts("ffprobe", "output.mp4"), [])

    def test_media_type_fills_null_codec_type(self) -> None:
        data = {"frames": [
            {"codec_type": None, "media_type": "video",
             "best_effort_timestamp_time": "0.0"},
            {"codec_type": "audio", "best_effort_timestamp_time": "0.01"},
        ]}
        with patch.object(fast_pro, "_probe_json", return_value=data):
            self.assertEqual(fast_pro._frame_pts("ffprobe", "output.mp4"), [0.0])


if __name__ == "__main__":
    unittest.main()
