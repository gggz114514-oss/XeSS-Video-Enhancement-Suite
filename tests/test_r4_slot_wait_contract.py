"""Published binary contracts; private lifecycle source checks stay private.
These checks are not long-run GPU tests. See the runtime acceptance reports.
"""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
MANIFEST = json.loads((ROOT / "runtime_manifest.json").read_text(encoding="utf-8"))


class NativeRuntimeContractTests(unittest.TestCase):
    def test_three_workers_remain_pinned(self):
        for name in ("gpu-block-native.exe", "gpu-dis-native.exe", "amd-of-native.exe"):
            path = "bin/" + name
            self.assertIn(path, MANIFEST["required_files"])
            self.assertEqual(len(MANIFEST["file_hashes"][path]), 64)

    def test_runtime_remains_published_r4(self):
        self.assertEqual(MANIFEST["release_status"], "published")
        self.assertEqual(MANIFEST["runtime_version"], "2026.09.07-r4")

    def test_gpu_source_builds_fail_explicitly(self):
        for name in ("build_offline_encoders.cmd", "build_amd_of_native.cmd"):
            script = (ROOT / "tools" / name).read_text(encoding="utf-8")
            self.assertIn("private sources", script)
            self.assertIn("exit /b 2", script)
            self.assertNotIn("cl /", script)

    def test_classic_source_rejects_private_block_option(self):
        source = (ROOT / "src/xess_fg.cpp").read_text(encoding="utf-8")
        branch = source.split('!strcmp(key, "--gpu-block-motion")', 1)[1].split('else if', 1)[0]
        self.assertIn("published R4 runtime", branch)
        self.assertIn("return false", branch)
