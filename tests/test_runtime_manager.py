"""Runtime state files are shared by Python and Windows PowerShell."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import runtime_manager


class RuntimeManagerJsonTests(unittest.TestCase):
    def test_load_json_accepts_windows_powershell_utf8_bom(self) -> None:
        expected = {"runtime_version": "test-r2", "ready": True}
        with tempfile.TemporaryDirectory() as temp:
            state = pathlib.Path(temp) / ".runtime-state.json"
            state.write_bytes(b"\xef\xbb\xbf" + json.dumps(expected).encode("utf-8"))
            self.assertEqual(runtime_manager._load_json(state), expected)

    def test_load_json_keeps_plain_utf8_compatible(self) -> None:
        expected = {"message": "运行库正常"}
        with tempfile.TemporaryDirectory() as temp:
            state = pathlib.Path(temp) / "state.json"
            state.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(runtime_manager._load_json(state), expected)


if __name__ == "__main__":
    unittest.main()
