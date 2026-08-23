from __future__ import annotations

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


def _random_pair(seed: int, batch: int = 2, channels: int = 16,
                 height: int = 24, width: int = 28):
    generator = torch.Generator().manual_seed(seed)
    fmap1 = torch.randn(batch, channels, height, width, generator=generator)
    fmap2 = torch.randn(batch, channels, height, width, generator=generator)
    return fmap1, fmap2, generator


def _coords(batch: int, height: int, width: int, generator: torch.Generator,
            max_offset: float) -> torch.Tensor:
    identity = coords_grid(batch, height, width, "cpu")
    offsets = (torch.rand(batch, 2, height, width, generator=generator) * 2.0 - 1.0)
    return identity + offsets * max_offset


class StreamingCorrParity(unittest.TestCase):
    def test_tiny_feature_maps_construct_and_query_safely(self) -> None:
        # Roughly 100px native input produces a 13px feature edge.  A 1px
        # feature map additionally exercises the pyramid's sampleable floor.
        for size in (13, 1):
            with self.subTest(size=size):
                generator = torch.Generator().manual_seed(23 + size)
                fmap1 = torch.randn(1, 8, size, size, generator=generator)
                fmap2 = torch.randn(1, 8, size, size, generator=generator)
                coords = coords_grid(1, size, size, "cpu")
                dense_block = CorrBlock(fmap1, fmap2, ARGS)
                stream_block = StreamingCorrBlock(fmap1, fmap2, ARGS)
                self.assertEqual(len(dense_block.corr_pyramid), ARGS.corr_levels)
                self.assertEqual(len(stream_block.fmap2_levels), ARGS.corr_levels)
                self.assertEqual(dense_block.level_scales, [1, 2, 4, 8])
                self.assertEqual(stream_block.level_scales, [1, 2, 4, 8])
                dense = dense_block(coords)
                stream = stream_block(coords)
                expected = (1, ARGS.corr_levels * (2 * ARGS.corr_radius + 1) ** 2,
                            size, size)
                self.assertEqual(dense.shape, expected)
                self.assertEqual(stream.shape, expected)
                self.assertTrue(bool(torch.isfinite(dense).all()))
                self.assertTrue(bool(torch.isfinite(stream).all()))
                self.assertLess(float((dense - stream).abs().max()), 1e-3)

    def test_matches_dense_for_random_flow(self) -> None:
        for seed in (7, 8):
            with self.subTest(seed=seed):
                fmap1, fmap2, generator = _random_pair(seed)
                coords = _coords(2, 24, 28, generator, max_offset=30.0)
                dense = CorrBlock(fmap1, fmap2, ARGS)(coords)
                stream = StreamingCorrBlock(fmap1, fmap2, ARGS)(coords)
                self.assertLess(float((dense - stream).abs().max()), 1e-3)

    def test_matches_dense_with_random_dilation(self) -> None:
        fmap1, fmap2, generator = _random_pair(11)
        coords = _coords(2, 24, 28, generator, max_offset=12.0)
        dilation = 0.5 + torch.rand(2, 1, 24, 28, generator=generator)
        dense = CorrBlock(fmap1, fmap2, ARGS)(coords, dilation=dilation)
        stream = StreamingCorrBlock(fmap1, fmap2, ARGS)(coords, dilation=dilation)
        self.assertLess(float((dense - stream).abs().max()), 1e-3)

    def test_out_of_range_coordinates_use_zero_padding(self) -> None:
        fmap1, fmap2, generator = _random_pair(13)
        coords = _coords(2, 24, 28, generator, max_offset=90.0)
        dense = CorrBlock(fmap1, fmap2, ARGS)(coords)
        stream = StreamingCorrBlock(fmap1, fmap2, ARGS)(coords)
        self.assertLess(float((dense - stream).abs().max()), 1e-3)

    def test_far_out_of_bounds_collapse_to_zero_padding(self) -> None:
        fmap1, fmap2, _ = _random_pair(13)
        batch, _, height, width = fmap1.shape
        far = torch.full((batch, 2, height, width), -120.0)
        dense = CorrBlock(fmap1, fmap2, ARGS)(far)
        stream = StreamingCorrBlock(fmap1, fmap2, ARGS)(far)
        self.assertLess(float((dense - stream).abs().max()), 1e-3)
        self.assertLess(float(dense.abs().max()), 1e-5)

    def test_output_shape_and_dtype_match_dense(self) -> None:
        fmap1, fmap2, generator = _random_pair(17, batch=1)
        coords = _coords(1, 24, 28, generator, max_offset=4.0)
        dense = CorrBlock(fmap1, fmap2, ARGS)(coords)
        stream = StreamingCorrBlock(fmap1, fmap2, ARGS)(coords)
        self.assertEqual(stream.shape, dense.shape)
        self.assertEqual(stream.dtype, torch.float32)

    @unittest.skipUnless(torch.xpu.is_available(), "requires torch.xpu")
    def test_xpu_matches_cpu(self) -> None:
        fmap1, fmap2, generator = _random_pair(19)
        coords = _coords(2, 24, 28, generator, max_offset=25.0)
        stream_xpu = StreamingCorrBlock(
            fmap1.to("xpu"), fmap2.to("xpu"), ARGS)(coords.to("xpu"))
        stream_cpu = StreamingCorrBlock(fmap1, fmap2, ARGS)(coords)
        self.assertLess(
            float((stream_xpu.cpu() - stream_cpu).abs().max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
