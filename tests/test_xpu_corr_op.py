from __future__ import annotations

import importlib
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "pipeline" / "sea_raft_core"))

import torch  # noqa: E402

from corr import CorrBlock, StreamingCorrBlock  # noqa: E402
from utils.utils import coords_grid  # noqa: E402

ARGS = SimpleNamespace(corr_levels=4, corr_radius=4)
TAPS = ARGS.corr_levels * (2 * ARGS.corr_radius + 1) ** 2


def _extension_usable() -> bool:
    if not torch.xpu.is_available():
        return False
    try:
        from xpu_corr import loader
    except ImportError:
        return False
    return loader.is_available()


EXT_AVAILABLE = _extension_usable()


def _random_pair(seed: int, batch: int = 1, channels: int = 64,
                 height: int = 24, width: int = 28):
    generator = torch.Generator().manual_seed(seed)
    fmap1 = torch.randn(batch, channels, height, width, generator=generator)
    fmap2 = torch.randn(batch, channels, height, width, generator=generator)
    return fmap1, fmap2, generator


def _coords(batch: int, height: int, width: int,
            generator: torch.Generator, max_offset: float) -> torch.Tensor:
    identity = coords_grid(batch, height, width, "cpu")
    offsets = torch.rand(batch, 2, height, width, generator=generator) * 2 - 1
    return identity + offsets * max_offset


def _run_fused(fmap1: torch.Tensor, fmap2: torch.Tensor,
               coords: torch.Tensor,
               dilation: torch.Tensor | None = None) -> torch.Tensor:
    from xpu_corr import loader

    reference_block = StreamingCorrBlock(fmap1, fmap2, ARGS)
    if dilation is None:
        dilation = torch.ones(coords.shape[0], 1, *coords.shape[2:])
    return loader.gather_correlate_pyramid(
        fmap1.to("xpu").contiguous(),
        [level.to("xpu").contiguous() for level in reference_block.fmap2_levels],
        coords.to("xpu").contiguous(),
        dilation.to("xpu").contiguous(),
        list(reference_block.level_scales),
        ARGS.corr_radius,
    ).cpu()


@unittest.skipUnless(torch.xpu.is_available(), "requires torch.xpu")
class SmokeAddProbe(unittest.TestCase):
    def test_add_one_stays_on_xpu(self) -> None:
        from xpu_corr import loader

        self.assertTrue(loader.is_available(), loader.status_text())
        source = torch.zeros(16, device="xpu")
        result = loader.smoke_add(source)
        torch.xpu.synchronize()
        self.assertTrue(result.is_xpu)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(float((result - 1.0).abs().max()), 0.0)


@unittest.skipUnless(EXT_AVAILABLE, "requires torch.xpu + xess_xpu_corr build")
class FusedGatherCorrelateParity(unittest.TestCase):
    TOL = dict(atol=2e-4, rtol=2e-4)

    def _assert_matches_streaming(self, fmap1, fmap2, coords,
                                  dilation=None) -> torch.Tensor:
        reference = StreamingCorrBlock(fmap1, fmap2, ARGS)(coords,
                                                           dilation=dilation)
        fused = _run_fused(fmap1, fmap2, coords, dilation)
        self.assertEqual(fused.dtype, torch.float32)
        self.assertEqual(fused.shape, reference.shape)
        self.assertTrue(bool(torch.isfinite(fused).all()))
        torch.testing.assert_close(fused, reference, **self.TOL)
        return fused

    def test_matches_streaming_b1_c256(self) -> None:
        fmap1, fmap2, gen = _random_pair(7, batch=1, channels=256,
                                         height=13, width=17)
        coords = _coords(1, 13, 17, gen, max_offset=8.0)
        self._assert_matches_streaming(fmap1, fmap2, coords)

    def test_matches_streaming_b2_random_flow(self) -> None:
        for seed in (11, 12):
            with self.subTest(seed=seed):
                fmap1, fmap2, gen = _random_pair(seed, batch=2, height=15,
                                                 width=19)
                coords = _coords(2, 15, 19, gen, max_offset=30.0)
                self._assert_matches_streaming(fmap1, fmap2, coords)

    def test_matches_dense_small(self) -> None:
        fmap1, fmap2, gen = _random_pair(21, batch=1, channels=16)
        coords = _coords(1, 24, 28, gen, max_offset=5.0)
        dense = CorrBlock(fmap1, fmap2, ARGS)(coords)
        fused = _run_fused(fmap1, fmap2, coords)
        self.assertLess(float((dense - fused).abs().max()), 1e-3)

    def test_odd_feature_sizes(self) -> None:
        for height, width in ((13, 27), (9, 31)):
            with self.subTest(size=(height, width)):
                fmap1, fmap2, gen = _random_pair(
                    31 + height, batch=1, height=height, width=width)
                coords = _coords(1, height, width, gen, max_offset=10.0)
                self._assert_matches_streaming(fmap1, fmap2, coords)

    def test_matches_streaming_with_random_dilation(self) -> None:
        fmap1, fmap2, gen = _random_pair(41, batch=2, height=15, width=19)
        coords = _coords(2, 15, 19, gen, max_offset=12.0)
        dilation = 0.25 + torch.rand(2, 1, 15, 19, generator=gen)
        self._assert_matches_streaming(fmap1, fmap2, coords, dilation=dilation)

    def test_partial_support_on_every_edge_and_corner(self) -> None:
        # Coordinates sit exactly on the frame border so each bilinear sample
        # keeps one valid neighbour while the outside neighbour must be zeroed
        # independently; corners exercise both axes at once.
        fmap1, fmap2, gen = _random_pair(43, batch=1)
        height, width = fmap1.shape[2:]
        xs = torch.tensor([0.0, width - 1.0, 0.0, width - 1.0,
                           (width - 1) / 2])
        ys = torch.tensor([0.0, 0.0, height - 1.0, height - 1.0,
                           (height - 1) / 2])
        coords = torch.zeros(1, 2, height, width)
        for index in range(5):
            coords[0, 0, 0, index] = xs[index]
            coords[0, 1, 0, index] = ys[index]
        # Fill the remaining rows deterministically around the borders.
        coords[0, 0, :, 5:] = torch.rand(height, width - 5,
                                         generator=gen) * (width - 1)
        coords[0, 1, :, 5:] = torch.rand(height, width - 5,
                                         generator=gen) * (height - 1)
        fused = self._assert_matches_streaming(fmap1, fmap2, coords)
        self.assertGreater(float(fused.abs().max()), 0.0)

    def test_moderately_out_of_bounds_uses_zero_padding(self) -> None:
        fmap1, fmap2, gen = _random_pair(47, batch=2, height=15, width=19)
        coords = _coords(2, 15, 19, gen, max_offset=90.0)
        self._assert_matches_streaming(fmap1, fmap2, coords)

    def test_far_out_of_bounds_collapses_to_zero(self) -> None:
        fmap1, fmap2, _ = _random_pair(53, batch=1)
        height, width = fmap1.shape[2:]
        far = torch.full((1, 2, height, width), -120.0)
        reference = StreamingCorrBlock(fmap1, fmap2, ARGS)(far)
        fused = _run_fused(fmap1, fmap2, far)
        torch.testing.assert_close(fused, reference, **self.TOL)
        self.assertLess(float(fused.abs().max()), 1e-5)

    def test_extremely_far_out_of_bounds_stays_finite(self) -> None:
        fmap1, fmap2, _ = _random_pair(59, batch=1)
        height, width = fmap1.shape[2:]
        far = torch.full((1, 2, height, width), 1.0e6)
        reference = StreamingCorrBlock(fmap1, fmap2, ARGS)(far)
        fused = _run_fused(fmap1, fmap2, far)
        self.assertTrue(bool(torch.isfinite(fused).all()))
        torch.testing.assert_close(fused, reference, atol=1e-6, rtol=1e-4)

    def test_output_channel_order_is_level_outer_tap_inner(self) -> None:
        # A single level scale collapses the pyramid contribution check:
        # with scales [1, 2, 4, 8] the first 81 channels must come from the
        # level-0 map.  Compare against streaming slice-for-slice.
        fmap1, fmap2, gen = _random_pair(61, batch=1, channels=32,
                                         height=11, width=13)
        coords = _coords(1, 11, 13, gen, max_offset=3.0)
        reference = StreamingCorrBlock(fmap1, fmap2, ARGS)(coords)
        fused = _run_fused(fmap1, fmap2, coords)
        side = 2 * ARGS.corr_radius + 1
        for level in range(ARGS.corr_levels):
            lo = level * side * side
            hi = (level + 1) * side * side
            torch.testing.assert_close(
                fused[:, lo:hi], reference[:, lo:hi], **self.TOL)


@unittest.skipUnless(torch.xpu.is_available(), "requires torch.xpu")
class StagedPathStats(unittest.TestCase):
    """Prove which execution path served each case via the stat counters.

    The counters only advance while XESS_XPU_CORR_STATS=1 and are read
    through the pybind helpers, so these tests cannot silently pass on a
    kernel that fell back without saying so.
    """

    ENV_KEYS = ("XESS_XPU_CORR_STATS", "XESS_XPU_CORR_SLM",
                "XESS_XPU_CORR_SLM_MAX")

    def setUp(self) -> None:
        if not EXT_AVAILABLE:
            self.skipTest("extension build unavailable")
        from xpu_corr import loader

        self.loader = loader
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)

    def _run_case(self, coords, slm_max=None):
        batch, _, height, width = coords.shape
        fmap1, fmap2, _ = _random_pair(97, batch=batch, height=height,
                                       width=width)
        reference = StreamingCorrBlock(fmap1, fmap2, ARGS)(coords)
        if slm_max is not None:
            os.environ["XESS_XPU_CORR_SLM_MAX"] = str(slm_max)
        os.environ["XESS_XPU_CORR_STATS"] = "1"
        self.loader.reset_corr_stats()
        fused = _run_fused(fmap1, fmap2, coords)
        torch.testing.assert_close(fused, reference, atol=2e-4, rtol=2e-4)
        return self.loader.corr_stats()

    def test_identity_coords_fully_staged(self) -> None:
        # Identity grid with dilation 1: the patch always fits the slab, so
        # every work group must be served from local memory.
        coords = coords_grid(1, 48, 52, "cpu")
        staged, fallback = self._run_case(coords)
        self.assertGreater(staged, 0)
        self.assertEqual(fallback, 0)

    def test_shrinking_slm_budget_forces_fallback(self) -> None:
        # A 4-float budget can never hold a 9x9-tap patch, so all work groups
        # must take the direct path while results stay bit-comparable.
        coords = coords_grid(1, 48, 52, "cpu")
        staged, fallback = self._run_case(coords, slm_max=4)
        self.assertEqual(staged, 0)
        self.assertGreater(fallback, 0)

    def test_slm_off_reports_direct_groups(self) -> None:
        coords = coords_grid(1, 24, 28, "cpu")
        os.environ["XESS_XPU_CORR_SLM"] = "off"
        os.environ["XESS_XPU_CORR_STATS"] = "1"
        fmap1, fmap2, _ = _random_pair(101)
        reference = StreamingCorrBlock(fmap1, fmap2, ARGS)(coords)
        self.loader.reset_corr_stats()
        fused = _run_fused(fmap1, fmap2, coords)
        torch.testing.assert_close(fused, reference, atol=2e-4, rtol=2e-4)
        staged, fallback = self.loader.corr_stats()
        self.assertEqual(staged, 0)
        self.assertGreater(fallback, 0)

    def test_non_finite_inputs_stay_defined_without_crashing(self) -> None:
        # grid_sample would propagate NaN; the fused kernel defines non-finite
        # coordinates as zero contribution instead of undefined behaviour.
        # The contract under test is "finite output, no device fault".
        fmap1, fmap2, gen = _random_pair(103)
        height, width = fmap1.shape[2:]
        coords = _coords(1, height, width, gen, max_offset=4.0)
        coords[0, 0, 0, 0] = float("nan")
        coords[0, 1, 0, 1] = float("inf")
        os.environ["XESS_XPU_CORR_STATS"] = "1"
        self.loader.reset_corr_stats()
        fused = _run_fused(fmap1, fmap2, coords)
        self.assertTrue(bool(torch.isfinite(fused).all()))
        # Work groups without wild lanes may legitimately stay staged; the
        # kernel just has to account for every group it launched.
        self.assertGreater(sum(self.loader.corr_stats()), 0)


@unittest.skipUnless(torch.xpu.is_available(), "requires torch.xpu")
class DispatcherGuards(unittest.TestCase):
    def test_cpu_tensors_fail_loudly_without_fallback(self) -> None:
        if not EXT_AVAILABLE:
            self.skipTest("extension build unavailable")
        import torch  # noqa: F811

        fmap1 = torch.randn(1, 8, 8, 8)
        fmap2 = torch.randn(1, 8, 8, 8)
        coords = coords_grid(1, 8, 8, "cpu")
        dilation = torch.ones(1, 1, 8, 8)
        with self.assertRaises(RuntimeError):
            torch.ops.xess_xpu.gather_correlate_pyramid(
                fmap1, [fmap2], coords, dilation, [1, 2, 4, 8], 4)


class EnvironmentSwitchPolicy(unittest.TestCase):
    """Selection-policy behaviour that does not need a working device."""

    def _fresh_loader(self):
        for name in ("xpu_corr", "xpu_corr.loader", "xess_xpu_corr"):
            sys.modules.pop(name, None)
        spec_parent = importlib.import_module("xpu_corr")
        return importlib.reload(importlib.import_module("xpu_corr.loader"))

    def tearDown(self) -> None:
        os.environ.pop("XESS_XPU_CORR", None)
        os.environ.pop("XESS_XPU_CORR_LIB_DIR", None)
        # Restore the real loader state for subsequent tests.
        self._fresh_loader()

    def test_off_never_loads_even_when_binary_exists(self) -> None:
        os.environ["XESS_XPU_CORR"] = "off"
        loader = self._fresh_loader()
        self.assertFalse(loader.is_available())
        self.assertIn("off", loader.status_text())

    def test_required_raises_for_missing_binary(self) -> None:
        if EXT_AVAILABLE:
            self.skipTest("real binary present; cannot simulate absence "
                          "without polluting sys.modules")
        os.environ["XESS_XPU_CORR"] = "required"
        os.environ["XESS_XPU_CORR_LIB_DIR"] = str(
            ROOT / "build" / "does-not-exist")
        loader = self._fresh_loader()
        with self.assertRaises(RuntimeError):
            loader.is_available()

    def test_invalid_mode_is_treated_as_auto(self) -> None:
        os.environ["XESS_XPU_CORR"] = "sometimes"
        loader = self._fresh_loader()
        # Must not raise whatever the machine has.
        loader.is_available()


if __name__ == "__main__":
    unittest.main()
