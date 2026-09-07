"""Static contracts for the native entry built for the three GPU routes.

Source guards, not a substitute for long-run/device fault tests.
The retired full_chain_probe main is not the published worker entry.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
SOURCE = (ROOT / "src/realtime/vpl_gpu_full_fg.cpp").read_text(encoding="utf-8")


class NativeLifecycleContractTests(unittest.TestCase):
    def test_decode_warnings_are_not_errors(self):
        branch = SOURCE.split("// Positive statuses are warnings.", 1)[1].split("mfxSurfaceHeader request", 1)[0]
        self.assertIn("if (!surface) continue;", branch)
        self.assertIn("Synchronize(surface, 15000)", branch)
        self.assertIn("surface->FrameInterface->Release(surface)", branch)

    def test_decoder_pool_has_move_only_owner(self):
        lease = SOURCE.split("struct DecodeSurfaceLease", 1)[1].split("struct FullGpuDecoder", 1)[0]
        self.assertIn("DecodeSurfaceLease(const DecodeSurfaceLease&) = delete", lease)
        self.assertIn("~DecodeSurfaceLease() { reset(); }", lease)
        self.assertIn("surface->FrameInterface->Release(surface)", lease)
        self.assertIn("lease_out->surface = surface;", SOURCE)

    def test_encoder_waits_and_returns_copy_slot(self):
        worker = SOURCE.split("void worker_loop()", 1)[1].split("Runtime& runtime_", 1)[0]
        self.assertLess(worker.index("wait_queue_fence"), worker.index("encoder_.encode_slot"))
        self.assertIn("free_slots_.push_back(job.copy_slot)", worker)
        self.assertIn("condition_.notify_all()", worker)
        self.assertIn('fail_locked("encode_slot_wait_timeout")', SOURCE)
        self.assertIn("std::atomic<bool> failed_{false}", SOURCE)

    def test_three_published_routes_use_same_native_entry(self):
        for path in ("providers/strict_dis_main.cpp", "amd_of_native_main.cpp"):
            src = (ROOT / "src/realtime" / path).read_text(encoding="utf-8")
            self.assertIn("vpl_gpu_full_fg.cpp", src)
        build = (ROOT/"tools/build_amd_of_native.cmd").read_text()
        self.assertIn("amd_of_native_main vpl_gpu_full_fg", build)
