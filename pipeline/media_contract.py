"""Session-level media and timeline contracts for R4 product routes.

This module deliberately contains no decoder, encoder, GPU or pixel code.  It
is evaluated once while a job is planned and gives every product adapter the
same answers for dimensions, frame count and frame-rate semantics.  Keeping
these rules out of native workers avoids both duplicated guesses and a
per-frame abstraction cost.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


MEDIA_CONTRACT_VERSION = "r4.media-contract.v1"


class MediaContractError(ValueError):
    """Input metadata or a requested output contract is invalid."""


@dataclass(frozen=True)
class MediaSpec:
    """The small, immutable part of one video stream used for job planning."""

    width: int
    height: int
    fps: float
    frames: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise MediaContractError(
                f"invalid media dimensions: {self.width}x{self.height}")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise MediaContractError(f"invalid media frame rate: {self.fps!r}")
        if self.frames < 1:
            raise MediaContractError(f"invalid media frame count: {self.frames}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *,
                     frame_limit: int = 0, minimum_frames: int = 1) -> "MediaSpec":
        """Parse probe metadata without silently accepting partial values."""
        try:
            frames = int(value["frames"])
            if frame_limit > 0:
                frames = min(frames, int(frame_limit))
            result = cls(width=int(value["width"]), height=int(value["height"]),
                         fps=float(value["fps"]), frames=frames)
        except (KeyError, TypeError, ValueError) as exc:
            raise MediaContractError(f"invalid media metadata: {value!r}") from exc
        if result.frames < minimum_frames:
            raise MediaContractError(
                f"media requires at least {minimum_frames} frames; got {result.frames}")
        return result

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class OutputContract:
    """Expected media result for one selected backend and operation."""

    operation: str
    backend: str
    width: int
    height: int
    fps: float
    frames: int
    frame_semantics: str
    pts_mode: str = "constant-fps"
    audio_policy: str = "preserve-if-present"
    version: str = MEDIA_CONTRACT_VERSION

    def __post_init__(self) -> None:
        MediaSpec(self.width, self.height, self.fps, self.frames)
        if self.operation not in {"sr", "fg"}:
            raise MediaContractError(f"unsupported operation: {self.operation!r}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def scaled_dimensions(width: int, height: int, scale: float,
                      *, alignment: int = 16) -> tuple[int, int]:
    """Return the product SR geometry using the existing nearest-16 rule."""
    if width <= 0 or height <= 0:
        raise MediaContractError(f"invalid media dimensions: {width}x{height}")
    if not math.isfinite(scale) or scale <= 0:
        raise MediaContractError("scale must be finite and greater than zero")
    if alignment <= 0:
        raise MediaContractError("alignment must be greater than zero")
    return (max(alignment, int(round(width * scale / alignment) * alignment)),
            max(alignment, int(round(height * scale / alignment) * alignment)))


def expected_output_frames(operation: str, backend: str,
                           input_frames: int) -> tuple[int, str]:
    """Return count and explicit drain semantics for a selected route.

    Intel oneVPL AI FI emits a drained tail frame (2N).  XeFG-based routes
    interpolate only between source frames (2N-1).  Treating those as the same
    contract previously forced callers to guess from the backend name.
    """
    if input_frames < 1:
        raise MediaContractError("input frame count must be greater than zero")
    if operation == "sr":
        return input_frames, "N"
    if operation != "fg":
        raise MediaContractError(f"unsupported operation: {operation!r}")
    if backend == "intel-vpl-ai":
        return input_frames * 2, "2N-drained"
    if backend in {"gpu-block", "gpu-dis", "cpu-dis"}:
        return input_frames * 2 - 1, "2N-1-between-source-frames"
    raise MediaContractError(f"unknown FG backend: {backend!r}")


def expected_output_rate(operation: str, input_rate: Any) -> Any:
    """Preserve exact Fraction rates while sharing the SR/FG multiplier."""
    try:
        valid = input_rate > 0
    except TypeError as exc:
        raise MediaContractError(f"invalid input frame rate: {input_rate!r}") from exc
    if not valid:
        raise MediaContractError(f"invalid input frame rate: {input_rate!r}")
    if operation == "sr":
        return input_rate
    if operation == "fg":
        return input_rate * 2
    raise MediaContractError(f"unsupported operation: {operation!r}")


def plan_output_contract(operation: str, backend: str, source: MediaSpec,
                         *, width: int | None = None,
                         height: int | None = None) -> OutputContract:
    """Plan one backend result; called once before workers are launched."""
    frames, semantics = expected_output_frames(operation, backend, source.frames)
    return OutputContract(
        operation=operation, backend=backend,
        width=source.width if width is None else int(width),
        height=source.height if height is None else int(height),
        fps=float(expected_output_rate(operation, source.fps)),
        frames=frames, frame_semantics=semantics)


def require_compatible_report(report: Mapping[str, Any],
                              expected: OutputContract) -> None:
    """Reject a contradictory completed report while accepting legacy reports.

    R3 and early R4 reports have no ``media_contract`` member, so absence is a
    compatibility case.  Once a worker declares the versioned contract it may
    not contradict the session plan.  This check is deliberately performed
    after the worker exits, never at a frame boundary.
    """
    raw = report.get("media_contract")
    if raw in (None, {}):
        return
    if not isinstance(raw, Mapping):
        raise MediaContractError("worker media_contract is not an object")
    wanted = expected.as_dict()
    exact_fields = (
        "version", "operation", "backend", "width", "height", "frames",
        "frame_semantics", "pts_mode", "audio_policy",
    )
    for field in exact_fields:
        if raw.get(field) != wanted[field]:
            raise MediaContractError(
                f"worker media contract mismatch for {field}: "
                f"expected {wanted[field]!r}, got {raw.get(field)!r}")
    try:
        reported_fps = float(raw.get("fps"))
    except (TypeError, ValueError) as exc:
        raise MediaContractError("worker media contract has invalid fps") from exc
    if not math.isclose(reported_fps, expected.fps, rel_tol=1e-9, abs_tol=1e-9):
        raise MediaContractError(
            f"worker media contract fps mismatch: expected {expected.fps:g}, "
            f"got {reported_fps:g}")


__all__ = [
    "MEDIA_CONTRACT_VERSION", "MediaContractError", "MediaSpec",
    "OutputContract", "expected_output_frames", "expected_output_rate",
    "plan_output_contract",
    "require_compatible_report", "scaled_dimensions",
]
