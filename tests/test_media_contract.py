"""Unit tests for the session-only R4 media/timeline contract."""

from __future__ import annotations

import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from media_contract import (MEDIA_CONTRACT_VERSION, MediaContractError,
                            MediaSpec, expected_output_frames,
                            expected_output_rate,
                            plan_output_contract, require_compatible_report,
                            scaled_dimensions)  # noqa: E402


class MediaContractTests(unittest.TestCase):
    def test_probe_mapping_honours_frame_limit(self) -> None:
        spec = MediaSpec.from_mapping(
            {"width": "864", "height": 480, "fps": "23.976", "frames": 243},
            frame_limit=48, minimum_frames=2)
        self.assertEqual((spec.width, spec.height, spec.frames), (864, 480, 48))
        self.assertAlmostEqual(spec.fps, 23.976)

    def test_probe_mapping_fails_closed(self) -> None:
        for value in (
            {"width": 0, "height": 480, "fps": 24, "frames": 8},
            {"width": 864, "height": 480, "fps": float("nan"), "frames": 8},
            {"width": 864, "height": 480, "fps": 24, "frames": 1},
            {"width": 864, "height": 480, "frames": 8},
        ):
            with self.subTest(value=value), self.assertRaises(MediaContractError):
                MediaSpec.from_mapping(value, minimum_frames=2)

    def test_scaled_dimensions_preserve_existing_nearest_16_rule(self) -> None:
        self.assertEqual(scaled_dimensions(864, 480, 1.5), (1296, 720))
        self.assertEqual(scaled_dimensions(1920, 1080, 4 / 3), (2560, 1440))
        self.assertEqual(scaled_dimensions(1, 1, 0.1), (16, 16))

    def test_scaled_dimensions_reject_invalid_values(self) -> None:
        for scale in (0, -1, float("inf"), float("nan")):
            with self.subTest(scale=scale), self.assertRaises(MediaContractError):
                scaled_dimensions(864, 480, scale)

    def test_sr_keeps_count_rate_and_declares_n(self) -> None:
        contract = plan_output_contract(
            "sr", "gpu-block", MediaSpec(864, 480, 24.0, 8),
            width=1296, height=720)
        self.assertEqual((contract.width, contract.height), (1296, 720))
        self.assertEqual((contract.frames, contract.fps), (8, 24.0))
        self.assertEqual(contract.frame_semantics, "N")
        self.assertEqual(contract.version, MEDIA_CONTRACT_VERSION)

    def test_xefg_routes_use_between_frame_semantics(self) -> None:
        for backend in ("gpu-block", "gpu-dis", "cpu-dis"):
            with self.subTest(backend=backend):
                contract = plan_output_contract(
                    "fg", backend, MediaSpec(1280, 720, 30.0, 8))
                self.assertEqual((contract.frames, contract.fps), (15, 60.0))
                self.assertEqual(
                    contract.frame_semantics, "2N-1-between-source-frames")

    def test_intel_fi_declares_drained_tail(self) -> None:
        contract = plan_output_contract(
            "fg", "intel-vpl-ai", MediaSpec(1280, 720, 30.0, 8))
        self.assertEqual((contract.frames, contract.fps), (16, 60.0))
        self.assertEqual(contract.frame_semantics, "2N-drained")

    def test_output_rate_preserves_fraction_type(self) -> None:
        from fractions import Fraction
        rate = Fraction(24000, 1001)
        self.assertEqual(expected_output_rate("sr", rate), rate)
        self.assertEqual(expected_output_rate("fg", rate), rate * 2)

    def test_unknown_fg_backend_is_not_guessed(self) -> None:
        with self.assertRaises(MediaContractError):
            expected_output_frames("fg", "mystery", 8)

    def test_declared_report_contract_must_match(self) -> None:
        contract = plan_output_contract(
            "fg", "gpu-block", MediaSpec(1280, 720, 30.0, 8))
        require_compatible_report({"media_contract": contract.as_dict()}, contract)
        broken = contract.as_dict()
        broken["frames"] = 16
        with self.assertRaisesRegex(MediaContractError, "frames"):
            require_compatible_report({"media_contract": broken}, contract)

    def test_legacy_report_without_contract_remains_readable(self) -> None:
        contract = plan_output_contract(
            "sr", "cpu-dis", MediaSpec(864, 480, 24.0, 8))
        require_compatible_report({"status": "complete"}, contract)


if __name__ == "__main__":
    unittest.main()
