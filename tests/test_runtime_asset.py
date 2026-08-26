"""Fixed runtime packaging must include only supported model families."""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "tools"))

import build_runtime_asset  # noqa: E402


class RuntimeAssetCollectionTests(unittest.TestCase):
    def test_collect_excludes_retired_sea_raft_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = pathlib.Path(temp)
            for relative in build_runtime_asset.REQUIRED_FILES:
                path = source / pathlib.Path(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"required")
            extra_python = source / "python" / "Lib" / "site-packages" / "keep.txt"
            extra_python.parent.mkdir(parents=True, exist_ok=True)
            extra_python.write_text("keep", encoding="utf-8")
            retired = source / "models" / "sea-raft" / "sea_raft_s_full.safetensors"
            retired.parent.mkdir(parents=True, exist_ok=True)
            retired.write_bytes(b"retired")

            relatives = {
                relative.as_posix()
                for _, relative in build_runtime_asset.collect(source, None)
            }

            self.assertIn("python/python.exe", relatives)
            self.assertIn("python/Lib/site-packages/keep.txt", relatives)
            self.assertIn(
                "models/depth-anything-v2-small/depth_anything_v2_small.xml",
                relatives,
            )
            self.assertNotIn(
                "models/sea-raft/sea_raft_s_full.safetensors", relatives
            )


if __name__ == "__main__":
    unittest.main()
