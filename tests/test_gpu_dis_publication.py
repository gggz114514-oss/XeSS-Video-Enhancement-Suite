import json
from pathlib import Path
import tempfile
import unittest

from tools.publication_policy import is_gpu_dis_source, is_gpu_block_source, is_private_gpu_source
from tools.build_comfy_runtime import collect

ROOT = Path(__file__).resolve().parents[1]


class GpuDisPublicationTests(unittest.TestCase):
    def test_source_paths_are_blocked(self):
        for name in ("src/realtime/gpu_dis_fg_impl.cpp",
                     "src/gpu/native_strict_dis_provider.h",
                     "src/realtime/providers/strict_dis_main.cpp",
                     "shaders/dis_perfect_patch.hlsl",
                     "shaders/dis_cpu_rounding.hlsli",
                     "shaders/gpu-dis/renamed.hlsl",
                     "shaders/gpu_dis/surface_nv12_color.hlsl",
                     "SHADERS\\GPU-DIS\\IMPL.HLSL"):
            with self.subTest(name=name):
                self.assertTrue(is_gpu_dis_source(name))

    def test_public_sources_and_compiled_dis_remain_allowed(self):
        for name in ("pipeline/motion_core.py", "src/xess_vsr.cpp",
                     "shaders/native_sr_effects.hlsl",
                     "src/realtime/native_gpu_motion_provider.h",
                     "bin/gpu-dis-native.exe", "shaders/dis/dis_perfect_patch.dxil"):
            with self.subTest(name=name):
                self.assertFalse(is_private_gpu_source(name))

    def test_current_public_source_tree_is_clean(self):
        leaked = [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*")
                  if p.is_file() and is_private_gpu_source(p.relative_to(ROOT))]
        self.assertEqual(leaked, [])
        classic = (ROOT / "src/xess_fg.cpp").read_text(encoding="utf-8")
        self.assertNotIn("record_gpu_block_motion", classic)
        self.assertNotIn("surface_gpu_motion_tile", classic)

    def test_runtime_contract_keeps_gpu_dis(self):
        manifest = json.loads((ROOT / "runtime_manifest.json").read_text(encoding="utf-8"))
        for name in ("bin/gpu-dis-native.exe", "bin/gpu-block-native.exe", "shaders/dis/dis_native_gray.dxil"):
            self.assertIn(name, manifest["required_files"])
            self.assertIn(name, manifest["file_hashes"])

    def test_packager_rejects_dis_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shaders/gpu-dis"
            path.mkdir(parents=True)
            (path / "impl.hlsl").write_text("// private", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "GPU DIS"):
                collect(tmp)

    def test_block_sources_are_blocked(self):
        for name in ("src/realtime/vpl_gpu_full_fg.cpp", "src/core/include/xve_gpu_block_core.h",
                     "shaders/surface_gpu_motion_tile.hlsl", "shaders/surface_gpu_lite_upsample.hlsl",
                     "shaders/motion_pyramid_coarse.hlsl", "src/gpu-block/renamed.cpp"):
            with self.subTest(name=name):
                self.assertTrue(is_gpu_block_source(name))

    def test_packager_rejects_block_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shaders"
            path.mkdir()
            (path / "surface_gpu_motion_tile.hlsl").write_text("// private", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "GPU Block"):
                collect(tmp)
