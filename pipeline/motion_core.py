#!/usr/bin/env python3
"""Shared motion, depth, confidence, and scene-cut algorithms.

Standalone SR and FG own separate processor state. The combined CPU path runs
one source analyzer and passes its full arrays to both consumers.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction

import cv2
import numpy as np


def _span(timer, name):
    """Stage-timing span that degrades to a no-op when timing is off."""
    return timer.span(name) if timer is not None else contextlib.nullcontext()


def fail(message: str) -> None:
    raise RuntimeError(message)


def robust_normalize(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(depth, (2.0, 98.0))
    if not np.isfinite(low + high) or high - low < 1e-6:
        return np.full(depth.shape, 0.5, dtype=np.float32)
    return np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32)


def remap(array: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, border=0.0,
          interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.remap(array, map_x, map_y, interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def sampling_map(flow: np.ndarray,
                 grids: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (map_x, map_y, inside) for one-way warping of `flow`.

    `grids` is a reusable (grid_x, grid_y) pair for the same frame size; the
    grids depend only on the dimensions, so hot loops pass a cached copy.
    Values (and therefore every downstream result) are byte-identical.
    """
    height, width = flow.shape[:2]
    if grids is None:
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                     np.arange(height, dtype=np.float32))
    else:
        grid_x, grid_y = grids
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    inside = ((map_x >= 0.0) & (map_x <= width - 1.0) &
              (map_y >= 0.0) & (map_y <= height - 1.0))
    return map_x, map_y, inside


def flow_consistency(backward: np.ndarray, forward: np.ndarray, threshold: float):
    map_x, map_y, inside = sampling_map(backward)
    forward_at_previous = remap(forward, map_x, map_y)
    error = np.linalg.norm(backward + forward_at_previous, axis=2)
    magnitude = np.linalg.norm(backward, axis=2) + np.linalg.norm(forward_at_previous, axis=2)
    limit = threshold + 0.05 * magnitude
    reliable = inside & (error <= limit)
    confidence = np.clip(1.0 - error / np.maximum(limit * 2.0, 1e-4), 0.0, 1.0)
    confidence *= inside.astype(np.float32)
    return reliable, confidence.astype(np.float32), map_x, map_y, error


def single_direction_confidence(backward: np.ndarray, uncertainty: np.ndarray | None,
                                previous_gray: np.ndarray | None = None,
                                current_gray: np.ndarray | None = None,
                                grids: tuple[np.ndarray, np.ndarray] | None = None):
    map_x, map_y, inside = sampling_map(backward, grids)
    if uncertainty is None:
        confidence = inside.astype(np.float32)
    else:
        confidence = np.clip(1.0 - uncertainty, 0.0, 1.0) * inside.astype(np.float32)
    if previous_gray is not None and current_gray is not None:
        previous_smooth = cv2.GaussianBlur(previous_gray, (3, 3), 0.65)
        current_smooth = cv2.GaussianBlur(current_gray, (3, 3), 0.65)
        warped = remap(previous_smooth, map_x, map_y, border=0.0)
        residual = np.abs(current_smooth.astype(np.float32) - warped.astype(np.float32))
        brightness = np.maximum(current_smooth.astype(np.float32), warped.astype(np.float32))
        photometric = np.exp(-residual / (12.0 + 0.12 * brightness))
        # The photometric term catches disocclusions that one-way flow cannot see;
        # an explicit uncertainty map (when a flow engine provides one) stays primary.
        exponent = 0.45 if uncertainty is not None else 0.75
        confidence *= np.power(np.clip(photometric, 0.0, 1.0), exponent)
    reliable = confidence >= 0.42
    error = (1.0 - confidence) * 4.0
    return reliable, confidence, map_x, map_y, error


def align_relative_depth(current: np.ndarray, previous_warped: np.ndarray,
                         reliable: np.ndarray) -> np.ndarray:
    if np.count_nonzero(reliable) < min(1024, reliable.size // 8):
        return current
    current_values = current[reliable]
    previous_values = previous_warped[reliable]
    current_q = np.percentile(current_values, (10.0, 50.0, 90.0))
    previous_q = np.percentile(previous_values, (10.0, 50.0, 90.0))
    denominator = current_q[2] - current_q[0]
    if denominator < 1e-5:
        return current
    scale = float(np.clip((previous_q[2] - previous_q[0]) / denominator, 0.5, 2.0))
    shift = float(np.clip(previous_q[1] - scale * current_q[1], -0.5, 0.5))
    return np.clip(current * scale + shift, 0.0, 1.0).astype(np.float32)


def stabilize_depth(current: np.ndarray, previous: np.ndarray, backward: np.ndarray,
                    reliable: np.ndarray, confidence: np.ndarray,
                    temporal: float) -> np.ndarray:
    map_x, map_y, _ = sampling_map(backward)
    previous_warped = remap(previous, map_x, map_y, border=0.5)
    aligned = align_relative_depth(current, previous_warped, reliable)
    kernel = np.ones((3, 3), np.uint8)
    depth_range = cv2.dilate(aligned, kernel) - cv2.erode(aligned, kernel)
    history_weight = temporal * reliable.astype(np.float32) * confidence
    history_weight *= np.clip(1.0 - depth_range * 10.0, 0.0, 1.0)
    stable = aligned * (1.0 - history_weight) + previous_warped * history_weight
    return np.clip(stable, 0.0, 1.0).astype(np.float32)


def depth_aware_dilate_reference(flow: np.ndarray, inverse_depth: np.ndarray,
                                 reliable: np.ndarray, iterations: int,
                                 edge_threshold: float) -> np.ndarray:
    """Reference implementation (advanced-index temporaries); kept only for
    the bit-exact equivalence tests of the fast rewrite below."""
    if iterations <= 0:
        return flow
    height, width = inverse_depth.shape
    result = flow.copy()
    valid = reliable.copy()
    kernel = np.ones((3, 3), np.uint8)
    edge = (cv2.dilate(inverse_depth, kernel) - cv2.erode(inverse_depth, kernel)) > edge_threshold
    for _ in range(iterations):
        padded_depth = np.pad(inverse_depth, 1, mode="edge")
        padded_flow = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_valid = np.pad(valid, 1, mode="constant", constant_values=False)
        best_depth = np.full((height, width), -np.inf, dtype=np.float32)
        best_flow = result.copy()
        found = np.zeros((height, width), dtype=bool)
        for dy in range(3):
            for dx in range(3):
                candidate_depth = padded_depth[dy:dy + height, dx:dx + width]
                candidate_valid = padded_valid[dy:dy + height, dx:dx + width]
                better = candidate_valid & (candidate_depth > best_depth)
                best_depth[better] = candidate_depth[better]
                best_flow[better] = padded_flow[dy:dy + height, dx:dx + width][better]
                found |= candidate_valid
        replace = found & ((~valid) | edge)
        result[replace] = best_flow[replace]
        valid |= found
    return result


def depth_aware_dilate(flow: np.ndarray, inverse_depth: np.ndarray,
                       reliable: np.ndarray, iterations: int,
                       edge_threshold: float) -> np.ndarray:
    """Depth-guided 3x3 dilation, bit-identical to depth_aware_dilate_reference
    but without per-iteration advanced-index temporaries: buffers are
    allocated once and np.copyto(..., where=...) does elementwise masked
    stores, so peak memory stays flat and the loop is vectorised."""
    if iterations <= 0:
        return flow
    height, width = inverse_depth.shape
    result = flow.copy()
    valid = reliable.copy()
    kernel = np.ones((3, 3), np.uint8)
    edge = (cv2.dilate(inverse_depth, kernel) - cv2.erode(inverse_depth, kernel)) > edge_threshold
    # reused iteration buffers: no fresh advanced-index arrays per pass
    best_depth = np.empty((height, width), np.float32)
    best_flow = np.empty((height, width, 2), np.float32)
    better = np.empty((height, width), np.bool_)
    replace = np.empty((height, width), np.bool_)
    found = np.empty((height, width), np.bool_)
    for _ in range(iterations):
        padded_depth = np.pad(inverse_depth, 1, mode="edge")
        padded_flow = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_valid = np.pad(valid, 1, mode="constant", constant_values=False)
        best_depth.fill(-np.inf)
        np.copyto(best_flow, result)
        found.fill(False)
        for dy in range(3):
            for dx in range(3):
                cand_depth = padded_depth[dy:dy + height, dx:dx + width]
                cand_valid = padded_valid[dy:dy + height, dx:dx + width]
                cand_flow = padded_flow[dy:dy + height, dx:dx + width]
                np.greater(cand_depth, best_depth, out=better)
                np.logical_and(cand_valid, better, out=better)
                np.copyto(best_depth, cand_depth, where=better)
                np.copyto(best_flow, cand_flow,
                          where=better[:, :, np.newaxis])
                np.logical_or(found, cand_valid, out=found)
        np.logical_not(valid, out=replace)
        np.logical_or(replace, edge, out=replace)
        np.logical_and(found, replace, out=replace)
        np.copyto(result, best_flow, where=replace[:, :, np.newaxis])
        np.logical_or(valid, found, out=valid)
    return result


def detect_scene_cut(previous_gray: np.ndarray, current_gray: np.ndarray,
                     reliable_fraction: float) -> tuple[bool, dict[str, float]]:
    mean_change = float(np.mean(np.abs(current_gray.astype(np.float32) - previous_gray.astype(np.float32))))
    hist_prev = cv2.calcHist([previous_gray], [0], None, [32], [0, 256])
    hist_curr = cv2.calcHist([current_gray], [0], None, [32], [0, 256])
    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)
    histogram_distance = float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA))
    cut = ((mean_change > 52.0 and reliable_fraction < 0.30) or
           (histogram_distance > 0.62 and reliable_fraction < 0.45) or
           (mean_change > 75.0 and histogram_distance > 0.48))
    return cut, {"mean_luma_change": mean_change, "histogram_distance": histogram_distance,
                 "reliable_fraction": reliable_fraction}


def make_responsive_mask(previous_gray: np.ndarray, current_gray: np.ndarray,
                         backward: np.ndarray, confidence: np.ndarray,
                         inverse_depth: np.ndarray | None,
                         uncertainty: np.ndarray | None,
                         maximum: float = 0.8,
                         grids: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    map_x, map_y, inside = sampling_map(backward, grids)
    warped = remap(previous_gray, map_x, map_y, border=0.0)
    luma_change = np.abs(current_gray.astype(np.float32) - warped.astype(np.float32)) / 80.0
    response = np.maximum(1.0 - confidence, np.clip(luma_change, 0.0, 1.0) * 0.65)
    response = np.maximum(response, (~inside).astype(np.float32))
    if uncertainty is not None:
        response = np.maximum(response, np.clip(uncertainty, 0.0, 1.0) * 0.85)
    if inverse_depth is not None:
        kernel = np.ones((3, 3), np.uint8)
        depth_range = cv2.dilate(inverse_depth, kernel) - cv2.erode(inverse_depth, kernel)
        response = np.maximum(response, np.clip(depth_range / 0.10, 0.0, 1.0) * 0.70)
    response = cv2.GaussianBlur(response.astype(np.float32), (3, 3), 0.65)
    return np.clip(response, 0.0, maximum).astype(np.float32)


class DepthEstimator:
    def __init__(self, model_path: str, device: str):
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("OpenVINO runtime is missing") from exc
        if not os.path.isfile(model_path):
            fail(f"depth model is missing: {model_path}")
        core = ov.Core()
        requested = device.upper()
        root_device = requested.split(":", 1)[0]
        if root_device not in ("AUTO", "MULTI") and root_device not in core.available_devices:
            fail(f"OpenVINO device {device} is unavailable; available: {', '.join(core.available_devices)}")
        model = core.read_model(model_path)
        if model.input(0).partial_shape.is_dynamic:
            input_size = 518
            metadata_path = os.path.join(os.path.dirname(model_path), "model.json")
            if os.path.isfile(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as file:
                    input_size = int(json.load(file).get("input_size", input_size))
            model.reshape({model.input(0): [1, 3, input_size, input_size]})
        config = {}
        cache_dir = os.environ.get("OPENVINO_CACHE_DIR")
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            config["CACHE_DIR"] = cache_dir
        print(f"[motion] compiling depth model on {device}", file=sys.stderr, flush=True)
        self.compiled = core.compile_model(model, device, config)
        self.input = self.compiled.input(0)
        self.output = self.compiled.output(0)
        shape = list(self.input.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            fail(f"unexpected depth-model input shape: {shape}")
        self.input_h, self.input_w = int(shape[2]), int(shape[3])
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_CUBIC)
        tensor = resized.astype(np.float32) / 255.0
        tensor = ((tensor - self.mean) / self.std).transpose(2, 0, 1)[None]
        prediction = self.compiled([np.ascontiguousarray(tensor)])[self.output]
        depth = np.asarray(prediction).squeeze()
        if depth.ndim != 2:
            fail(f"unexpected depth-model output shape: {np.asarray(prediction).shape}")
        return robust_normalize(cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC))


class DisFlow:
    def __init__(self, bidirectional: bool):
        self.bidirectional_enabled = bidirectional
        preset = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM if bidirectional else cv2.DISOPTICAL_FLOW_PRESET_FAST
        self.backward = cv2.DISOpticalFlow_create(preset)
        self.forward = cv2.DISOpticalFlow_create(preset) if bidirectional else None
        self.directional_dispatch_count = 0

    def infer(self, previous_rgb: np.ndarray, current_rgb: np.ndarray,
              timer=None):
        """Returns (backward, forward, uncertainty).  timer splits the gray
        conversion and the two DIS passes so the profile is auditable."""
        with _span(timer, "dis_gray"):
            previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
            current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
        with _span(timer, "dis_backward"):
            backward = self.backward.calc(current_gray, previous_gray,
                                          None).astype(np.float32)
            self.directional_dispatch_count += 1
        forward = None
        if self.forward is not None:
            with _span(timer, "dis_forward"):
                forward = self.forward.calc(previous_gray, current_gray,
                                            None).astype(np.float32)
                self.directional_dispatch_count += 1
        return backward, forward, None


@dataclass
class MotionResult:
    flow: np.ndarray
    depth: np.ndarray | None
    mask: np.ndarray | None
    confidence: np.ndarray
    scene_cut: bool
    metrics: dict[str, float]


class GpuBlockMetadataAnalyzer:
    """Prepare only XeFG metadata while motion stays GPU-resident.

    Reading the GPU velocity back merely to stabilize CPU depth would defeat
    the route.  Scene cuts are therefore detected from luma/histograms, AI
    depth (when requested) is emitted per frame, and the ABI motion plane is
    a cached zero field that xess-fg explicitly ignores.
    """

    def __init__(self, depth_estimator: DepthEstimator | None, *,
                 responsive_max: float, timer=None):
        self.depth_estimator = depth_estimator
        self.responsive_max = responsive_max
        self.timer = timer
        self.previous_gray: np.ndarray | None = None
        self.zero_flow: np.ndarray | None = None

    def _zeros(self, height: int, width: int) -> np.ndarray:
        if self.zero_flow is None or self.zero_flow.shape[:2] != (height, width):
            self.zero_flow = np.zeros((height, width, 2), np.float32)
        return self.zero_flow

    def first(self, rgb: np.ndarray, with_mask: bool) -> MotionResult:
        height, width = rgb.shape[:2]
        with _span(self.timer, "depth_infer"):
            depth = self.depth_estimator.infer(rgb) if self.depth_estimator else None
        with _span(self.timer, "gray_convert"):
            self.previous_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return MotionResult(
            flow=self._zeros(height, width), depth=depth,
            mask=(np.full((height, width), self.responsive_max, np.float32)
                  if with_mask else None),
            confidence=np.ones((height, width), np.float32), scene_cut=True,
            metrics={"reliable_fraction": 1.0, "mean_flow_error": 0.0})

    def next(self, rgb: np.ndarray, with_mask: bool,
             dilate_highres: bool) -> MotionResult:
        del dilate_highres
        assert self.previous_gray is not None
        height, width = rgb.shape[:2]
        with _span(self.timer, "gray_convert"):
            current_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        with _span(self.timer, "scene_detect"):
            scene_cut, metrics = detect_scene_cut(
                self.previous_gray, current_gray, 1.0)
        with _span(self.timer, "depth_infer"):
            depth = self.depth_estimator.infer(rgb) if self.depth_estimator else None
        self.previous_gray = current_gray
        metrics["reliable_fraction"] = 1.0
        metrics["mean_flow_error"] = 0.0
        return MotionResult(
            flow=self._zeros(height, width), depth=depth,
            mask=(np.full((height, width), self.responsive_max, np.float32)
                  if with_mask else None),
            confidence=np.ones((height, width), np.float32),
            scene_cut=scene_cut, metrics=metrics)


class FrameAnalyzer:
    def __init__(self, flow_engine, depth_estimator: DepthEstimator | None, *,
                 temporal: float, consistency: float, dilation: int,
                 depth_edge: float, responsive_max: float,
                 photometric_confidence: bool = False, timer=None):
        self.flow_engine = flow_engine
        self.depth_estimator = depth_estimator
        self.temporal = temporal
        self.consistency = consistency
        self.dilation = dilation
        self.depth_edge = depth_edge
        self.responsive_max = responsive_max
        self.photometric_confidence = photometric_confidence
        self.timer = timer
        self.previous_rgb: np.ndarray | None = None
        self.previous_gray: np.ndarray | None = None
        self.previous_depth: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._grids: tuple[np.ndarray, np.ndarray] | None = None

    def _sampling_grids(self, shape):
        height, width = shape[:2]
        if self._grid_shape != (height, width):
            grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                         np.arange(height, dtype=np.float32))
            self._grid_shape = (height, width)
            self._grids = (grid_x, grid_y)
        return self._grids

    def first(self, rgb: np.ndarray, with_mask: bool) -> MotionResult:
        height, width = rgb.shape[:2]
        with _span(self.timer, "depth_infer"):
            depth = self.depth_estimator.infer(rgb) if self.depth_estimator else None
        with _span(self.timer, "state_copy"):
            self.previous_rgb = rgb.copy()
            self.previous_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            self.previous_depth = depth
        return MotionResult(flow=np.zeros((height, width, 2), np.float32), depth=depth,
                            mask=np.full((height, width), self.responsive_max, np.float32) if with_mask else None,
                            confidence=np.ones((height, width), np.float32), scene_cut=True,
                            metrics={"reliable_fraction": 1.0})

    def next(self, rgb: np.ndarray, with_mask: bool, dilate_highres: bool) -> MotionResult:
        assert self.previous_rgb is not None and self.previous_gray is not None
        with _span(self.timer, "gray_convert"):
            current_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        backward, forward, uncertainty = self.flow_engine.infer(
            self.previous_rgb, rgb, timer=self.timer)
        with _span(self.timer, "consistency"):
            if forward is not None:
                reliable, confidence, _, _, error = flow_consistency(
                    backward, forward, self.consistency)
            else:
                reliable, confidence, _, _, error = single_direction_confidence(
                    backward, uncertainty,
                    self.previous_gray if self.photometric_confidence else None,
                    current_gray if self.photometric_confidence else None,
                    grids=self._sampling_grids(backward.shape))
            reliable_fraction = float(np.mean(reliable))
        with _span(self.timer, "scene_detect"):
            scene_cut, metrics = detect_scene_cut(self.previous_gray,
                                                  current_gray,
                                                  reliable_fraction)
        with _span(self.timer, "depth_infer"):
            current_depth = (self.depth_estimator.infer(rgb)
                             if self.depth_estimator else None)
        stable_depth = current_depth
        with _span(self.timer, "depth_stabilize"):
            if current_depth is not None and self.previous_depth is not None \
                    and not scene_cut:
                stable_depth = stabilize_depth(
                    current_depth, self.previous_depth, backward,
                    reliable, confidence, self.temporal)
        with _span(self.timer, "scene_cut_reset"):
            if scene_cut:
                backward.fill(0.0)
                reliable.fill(True)
                confidence.fill(1.0)
        motion = backward
        with _span(self.timer, "dilate"):
            if dilate_highres and stable_depth is not None:
                motion = depth_aware_dilate(backward, stable_depth, reliable,
                                            self.dilation, self.depth_edge)
            # DisFlow 输出已是 float32;只做无拷贝视图,保持逐位一致且省去整帧复制
            motion = np.asarray(motion, dtype=np.float32)
            if not np.isfinite(motion).all():
                motion = np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)
        mask = None
        if with_mask:
            with _span(self.timer, "responsive_mask"):
                mask = make_responsive_mask(self.previous_gray, current_gray,
                                            backward, confidence, stable_depth,
                                            uncertainty, self.responsive_max,
                                            grids=self._sampling_grids(
                                                backward.shape))
                if scene_cut:
                    mask.fill(self.responsive_max)
        with _span(self.timer, "state_copy"):
            self.previous_rgb = rgb.copy()
            self.previous_gray = current_gray
            self.previous_depth = stable_depth
            metrics["mean_flow_error"] = float(np.mean(error))
        return MotionResult(flow=motion, depth=stable_depth, mask=mask,
                            confidence=confidence, scene_cut=scene_cut, metrics=metrics)


def write_debug(debug_dir: str, index: int, result: MotionResult) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    if result.depth is not None:
        depth_u8 = np.clip(result.depth * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f"depth_{index:06d}.png"),
                    cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO))
    confidence_u8 = np.clip(result.confidence * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(debug_dir, f"confidence_{index:06d}.png"), confidence_u8)
    if result.mask is not None:
        cv2.imwrite(os.path.join(debug_dir, f"responsive_{index:06d}.png"),
                    np.clip(result.mask * 255.0, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Shared MotionPacket core
# ---------------------------------------------------------------------------


def _pts(value) -> Fraction:
    """Normalize timestamps without losing the rational PTS contract.

    Float callers are accepted for the legacy runners, but are converted from
    their decimal spelling rather than from a binary float approximation.
    New callers should pass ``Fraction`` (or an integer timebase tick).
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("PTS must be finite")
        # Media-rate floats such as ``1 / 24`` should identify the same pair
        # as an explicit Fraction(1, 24), while still retaining a rational
        # value for strict monotonicity checks.
        return Fraction(value).limit_denominator(1_000_000)
    return Fraction(value)


def adapt_motion(motion: np.ndarray, source_size: tuple[int, int],
                 target_size: tuple[int, int]) -> np.ndarray:
    """Sample a current→previous MV field onto a target grid.

    ``cv2.resize`` changes the grid but not the vector units.  The explicit
    component scale below is therefore required for XeSS HIGH_RES_MV and XeFG
    when consumers use different dimensions.  The sign/direction is kept
    unchanged; this is not a second flow computation.
    """
    source_w, source_h = (int(source_size[0]), int(source_size[1]))
    target_w, target_h = (int(target_size[0]), int(target_size[1]))
    array = np.asarray(motion, dtype=np.float32)
    if array.shape != (source_h, source_w, 2):
        raise ValueError(
            f"motion shape {array.shape} does not match source {source_w}x{source_h}")
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target dimensions must be positive")
    if (source_w, source_h) == (target_w, target_h):
        return array
    sampled = cv2.resize(array, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    sampled = np.ascontiguousarray(sampled, dtype=np.float32)
    sampled[..., 0] *= np.float32(target_w / source_w)
    sampled[..., 1] *= np.float32(target_h / source_h)
    return sampled


@dataclass(frozen=True, slots=True)
class MotionPacket:
    """One immutable-by-contract result for one adjacent input-frame pair.

    ``frame_id`` identifies the current frame; ``frame_id - 1`` is the
    previous frame.  The vector field is always current→previous, in source
    pixels.  SR and FG receive this same packet and only create lightweight
    adapted views for their own grids.
    """

    frame_id: int
    prev_pts: Fraction
    current_pts: Fraction
    source_width: int
    source_height: int
    motion: np.ndarray = field(repr=False)
    confidence: np.ndarray | None = field(default=None, repr=False)
    occlusion: np.ndarray | None = field(default=None, repr=False)
    scene_cut: bool = False
    backend: str = "cpu-dis"
    producer_fence: int | None = None
    cpu_ready: bool = True
    depth: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.frame_id < 1:
            raise ValueError("MotionPacket frame_id must identify a pair (>= 1)")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("MotionPacket source dimensions must be positive")
        if _pts(self.current_pts) <= _pts(self.prev_pts):
            raise ValueError("MotionPacket PTS must be strictly increasing")
        vector = np.asarray(self.motion, dtype=np.float32)
        expected = (self.source_height, self.source_width, 2)
        if vector.shape != expected:
            raise ValueError(f"MotionPacket motion shape {vector.shape} != {expected}")
        if not np.isfinite(vector).all():
            raise ValueError("MotionPacket motion contains non-finite values")
        object.__setattr__(self, "prev_pts", _pts(self.prev_pts))
        object.__setattr__(self, "current_pts", _pts(self.current_pts))
        object.__setattr__(self, "motion", np.ascontiguousarray(vector))
        for name in ("confidence", "occlusion", "depth"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, np.ascontiguousarray(value))

    @property
    def source_size(self) -> tuple[int, int]:
        return self.source_width, self.source_height

    @property
    def mv_current_to_previous(self) -> np.ndarray:
        return self.motion

    def view(self, target_width: int, target_height: int) -> "MotionPacketView":
        return MotionPacketView(self, int(target_width), int(target_height))


@dataclass(frozen=True, slots=True)
class MotionPacketView:
    """Consumer-specific grid view; it retains packet identity/timestamps."""

    packet: MotionPacket
    target_width: int
    target_height: int
    motion: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError("MotionPacketView dimensions must be positive")
        object.__setattr__(
            self, "motion",
            adapt_motion(self.packet.motion, self.packet.source_size,
                         (self.target_width, self.target_height)))

    @property
    def frame_id(self) -> int:
        return self.packet.frame_id

    @property
    def prev_pts(self) -> Fraction:
        return self.packet.prev_pts

    @property
    def current_pts(self) -> Fraction:
        return self.packet.current_pts

    @property
    def source_size(self) -> tuple[int, int]:
        return self.packet.source_size

    @property
    def scene_cut(self) -> bool:
        return self.packet.scene_cut

    @property
    def backend(self) -> str:
        return self.packet.backend

    @property
    def scale(self) -> tuple[float, float]:
        return (self.target_width / self.packet.source_width,
                self.target_height / self.packet.source_height)


@dataclass(slots=True)
class MotionCounters:
    """Hard accounting shared by the provider and both consumers."""

    flow_pairs_requested: int = 0
    flow_pairs_computed: int = 0
    consumed_by_sr: int = 0
    consumed_by_fg: int = 0
    cpu_upload_bytes: int = 0
    cpu_readback_bytes: int = 0
    gpu_copies: int = 0

    # Short aliases keep reports readable while preserving the required keys.
    @property
    def requested(self) -> int:
        return self.flow_pairs_requested

    @property
    def computed(self) -> int:
        return self.flow_pairs_computed

    def record_consume(self, consumer: str) -> None:
        if consumer == "sr":
            self.consumed_by_sr += 1
        elif consumer == "fg":
            self.consumed_by_fg += 1
        else:
            raise ValueError(f"unknown MotionPacket consumer: {consumer}")

    def assert_complete(self, expected_pairs: int) -> None:
        expected_pairs = int(expected_pairs)
        if self.flow_pairs_computed != expected_pairs:
            raise AssertionError(
                f"computed {self.flow_pairs_computed} != expected {expected_pairs}")
        if self.flow_pairs_requested < expected_pairs:
            raise AssertionError("requested count is below computed pair count")

    def as_dict(self) -> dict[str, int]:
        return {
            "flow_pairs_requested": self.flow_pairs_requested,
            "flow_pairs_computed": self.flow_pairs_computed,
            "consumed_by_sr": self.consumed_by_sr,
            "consumed_by_fg": self.consumed_by_fg,
            "cpu_upload_bytes": self.cpu_upload_bytes,
            "cpu_readback_bytes": self.cpu_readback_bytes,
            "gpu_copies": self.gpu_copies,
        }


@dataclass(slots=True)
class _MotionEntry:
    packet: MotionPacket
    consumers: set[str] = field(default_factory=set)


class MotionPacketRing:
    """Bounded fan-out ring with explicit acquire/release backpressure."""

    def __init__(self, slots: int = 4, *, timeout: float = 30.0):
        if slots < 2:
            raise ValueError("MotionPacketRing needs at least two slots")
        if timeout <= 0:
            raise ValueError("MotionPacketRing timeout must be positive")
        self.slots = int(slots)
        self.timeout = float(timeout)
        self._entries: OrderedDict[int, _MotionEntry] = OrderedDict()
        self._condition = threading.Condition()
        self.wait_seconds = 0.0

    def publish(self, packet: MotionPacket) -> None:
        started = time.perf_counter()
        with self._condition:
            deadline = time.monotonic() + self.timeout
            while packet.frame_id not in self._entries and len(self._entries) >= self.slots:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a free MotionPacket slot")
                self._condition.wait(remaining)
            self.wait_seconds += time.perf_counter() - started
            if packet.frame_id in self._entries:
                existing = self._entries[packet.frame_id].packet
                if existing is not packet:
                    raise RuntimeError("duplicate frame_id has a different MotionPacket")
                return
            self._entries[packet.frame_id] = _MotionEntry(packet)
            self._condition.notify_all()

    def acquire(self, frame_id: int, consumer: str, *, target_size=None):
        if consumer not in ("sr", "fg"):
            raise ValueError("consumer must be 'sr' or 'fg'")
        started = time.perf_counter()
        with self._condition:
            deadline = time.monotonic() + self.timeout
            while frame_id not in self._entries:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for MotionPacket {frame_id}")
                self._condition.wait(remaining)
            self.wait_seconds += time.perf_counter() - started
            entry = self._entries[frame_id]
            if consumer in entry.consumers:
                raise RuntimeError(f"MotionPacket {frame_id} already acquired by {consumer}")
            entry.consumers.add(consumer)
            packet = entry.packet
            result = (packet.view(*target_size) if target_size is not None else packet)
            self._condition.notify_all()
            return result

    def release(self, frame_id: int, consumer: str) -> None:
        with self._condition:
            entry = self._entries.get(frame_id)
            if entry is None:
                raise KeyError(f"MotionPacket {frame_id} is not in the ring")
            if consumer not in entry.consumers:
                raise RuntimeError(f"MotionPacket {frame_id} not held by {consumer}")
            entry.consumers.remove(consumer)
            if not entry.consumers:
                del self._entries[frame_id]
            self._condition.notify_all()

    def contains(self, frame_id: int) -> bool:
        with self._condition:
            return frame_id in self._entries

    def __len__(self) -> int:
        with self._condition:
            return len(self._entries)


class SharedMotionCore:
    """Compute each adjacent pair once and fan out one packet to SR and FG.

    ``provider`` receives ``(previous_rgb, current_rgb)`` and may return a
    ``MotionResult``, a motion array, or a mapping containing ``flow``.  The
    provider is deliberately synchronous: callers can place this object in a
    resident preparer process and use the bounded ring for downstream
    backpressure.  GPU Block providers may return a fence-backed packet with
    ``backend='gpu-block'`` without forcing a CPU MV readback.
    """

    def __init__(self, provider, *, source_size: tuple[int, int],
                 backend: str = "cpu-dis", slots: int = 4,
                 timeout: float = 30.0, counters: MotionCounters | None = None):
        self.provider = provider
        self.source_size = (int(source_size[0]), int(source_size[1]))
        if min(self.source_size) <= 0:
            raise ValueError("source_size must be positive")
        self.backend = backend
        self.counters = counters or MotionCounters()
        self.ring = MotionPacketRing(slots, timeout=timeout)
        self._cache: OrderedDict[tuple, MotionPacket] = OrderedDict()
        self._cache_limit = max(2, int(slots))

    @staticmethod
    def _result_parts(result):
        if isinstance(result, MotionResult):
            return result.flow, result.confidence, result.mask, result.scene_cut, result.depth
        if isinstance(result, dict):
            return (result["flow"], result.get("confidence"),
                    result.get("occlusion", result.get("mask")),
                    bool(result.get("scene_cut", False)), result.get("depth"))
        if isinstance(result, tuple):
            if len(result) == 2:
                return result[0], result[1], None, False, None
            if len(result) >= 5:
                return result[0], result[1], result[2], bool(result[3]), result[4]
        return result, None, None, False, None

    def request_pair(self, frame_id: int, prev_pts, current_pts,
                     previous_rgb: np.ndarray, current_rgb: np.ndarray,
                     *, producer_fence: int | None = None,
                     cpu_ready: bool | None = None) -> MotionPacket:
        """Return the canonical packet for one pair; duplicate asks never recompute."""
        self.counters.flow_pairs_requested += 1
        key = (int(frame_id), _pts(prev_pts), _pts(current_pts))
        cached = self._cache.get(key)
        if cached is not None:
            # A packet may have been fully released from the bounded ring
            # while it is still in the small deduplication cache.  Re-publish
            # the canonical object so a late consumer still sees the same
            # packet rather than triggering a second DIS run.
            if not self.ring.contains(cached.frame_id):
                self.ring.publish(cached)
            return cached
        result = self.provider(previous_rgb, current_rgb)
        flow, confidence, occlusion, scene_cut, depth = self._result_parts(result)
        packet = MotionPacket(
            frame_id=int(frame_id), prev_pts=key[1], current_pts=key[2],
            source_width=self.source_size[0], source_height=self.source_size[1],
            motion=flow, confidence=confidence, occlusion=occlusion,
            scene_cut=scene_cut, backend=self.backend,
            producer_fence=producer_fence,
            cpu_ready=(self.backend == "cpu-dis" if cpu_ready is None else bool(cpu_ready)),
            depth=depth)
        self._cache[key] = packet
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        self.counters.flow_pairs_computed += 1
        self.ring.publish(packet)
        return packet

    def consume(self, frame_id: int, consumer: str, *, target_size=None):
        view = self.ring.acquire(frame_id, consumer, target_size=target_size)
        self.counters.record_consume(consumer)
        return view

    def release(self, frame_id: int, consumer: str) -> None:
        self.ring.release(frame_id, consumer)

    def record_transfer(self, *, upload_bytes: int = 0,
                        readback_bytes: int = 0, gpu_copies: int = 0) -> None:
        self.counters.cpu_upload_bytes += int(upload_bytes)
        self.counters.cpu_readback_bytes += int(readback_bytes)
        self.counters.gpu_copies += int(gpu_copies)

    def assert_complete(self, input_frames: int) -> None:
        self.counters.assert_complete(max(0, int(input_frames) - 1))

    def report(self) -> dict[str, object]:
        return {**self.counters.as_dict(), "backend": self.backend,
                "ring_slots": self.ring.slots,
                "ring_wait_seconds": self.ring.wait_seconds}
