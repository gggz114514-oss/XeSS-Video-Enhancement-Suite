from __future__ import annotations

import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "pipeline"))

import numpy as np  # noqa: E402

from motion_core import (automatic_flow_scale, scaled_flow_dimensions,
                         scaled_flow_size, upsample_optical_flow)  # noqa: E402


class UpsampleOpticalFlowTests(unittest.TestCase):
    def test_constant_field_doubles_magnitude(self) -> None:
        flow = np.full((720, 1296, 2), (3.0, -2.0), np.float32)
        up = upsample_optical_flow(flow, 1440, 2592)
        self.assertEqual(up.shape, (1440, 2592, 2))
        self.assertLess(float(np.abs(up[..., 0] - 6.0).max()), 1e-3)
        self.assertLess(float(np.abs(up[..., 1] + 4.0).max()), 1e-3)

    def test_odd_grid_restores_actual_ratio(self) -> None:
        # 17 target pixels over 16 source pixels: magnitudes must follow the
        # actual 17/16 ratio instead of a requested fraction such as 0.5.
        flow = np.full((16, 16, 2), (3.0, 5.0), np.float32)
        up = upsample_optical_flow(flow, 17, 17)
        self.assertLess(float(np.abs(up[..., 0] - 3.1875).max()), 1e-4)
        self.assertLess(float(np.abs(up[..., 1] - 5.3125).max()), 1e-4)

    def test_same_grid_preserves_magnitudes_exactly(self) -> None:
        # A near-1.0 scale rounds back to the source grid; dividing by the
        # requested scale would inflate magnitudes on an untouched field.
        rng = np.random.default_rng(11)
        flow = rng.standard_normal((100, 100, 2)).astype(np.float32) * 7.0
        same = upsample_optical_flow(flow, 100, 100)
        self.assertLess(float(np.abs(same - flow).max()), 1e-4)

    def test_output_is_float32(self) -> None:
        flow = np.zeros((60, 80, 2), np.float64)
        up = upsample_optical_flow(flow, 120, 160)
        self.assertEqual(up.dtype, np.float32)


class ScaledFlowSizeTests(unittest.TestCase):
    def test_reduced_inference_respects_quality_floor(self) -> None:
        # The floor retains useful spatial detail under very aggressive scales;
        # correlation-pyramid safety is tested independently.
        self.assertEqual(scaled_flow_size(200, 0.5), 128)
        self.assertEqual(scaled_flow_size(2592, 0.01), 128)

    def test_at_or_below_floor_passes_through(self) -> None:
        self.assertEqual(scaled_flow_size(100, 0.01), 100)
        self.assertEqual(scaled_flow_size(128, 0.5), 128)

    def test_normal_scaling_rounds(self) -> None:
        self.assertEqual(scaled_flow_size(2592, 0.5), 1296)
        self.assertEqual(scaled_flow_size(1440, 0.5), 720)
        self.assertEqual(scaled_flow_size(1440, 0.999), 1439)

    def test_dimensions_preserve_aspect_under_short_edge_floor(self) -> None:
        self.assertEqual(scaled_flow_dimensions(3840, 200, 1.0 / 3.0),
                         (2458, 128))


class AutomaticFlowScaleTests(unittest.TestCase):
    def test_landscape_inputs_reduce_short_edge_to_720(self) -> None:
        self.assertAlmostEqual(automatic_flow_scale(1920, 1080), 2.0 / 3.0)
        self.assertAlmostEqual(automatic_flow_scale(2592, 1440), 0.5)
        self.assertAlmostEqual(automatic_flow_scale(3840, 2160), 1.0 / 3.0)

    def test_portrait_inputs_fit_rotated_720p_box(self) -> None:
        self.assertAlmostEqual(automatic_flow_scale(1440, 2560), 0.5)

    def test_inputs_at_or_below_720p_stay_native(self) -> None:
        self.assertEqual(automatic_flow_scale(1280, 720), 1.0)
        self.assertEqual(automatic_flow_scale(864, 480), 1.0)


if __name__ == "__main__":
    unittest.main()
