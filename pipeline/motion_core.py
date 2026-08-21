#!/usr/bin/env python3
"""Shared motion, depth, confidence, and scene-cut algorithms.

SR and FG instantiate their own processor state.  This module shares code, not
per-video geometry caches, so FG always analyses the actual frames it receives.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np


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


def sampling_map(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
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
                                current_gray: np.ndarray | None = None):
    map_x, map_y, inside = sampling_map(backward)
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
        # SEA-RAFT uncertainty remains the primary signal when present; the
        # photometric term catches disocclusions that one-way flow cannot see.
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


def depth_aware_dilate(flow: np.ndarray, inverse_depth: np.ndarray,
                       reliable: np.ndarray, iterations: int,
                       edge_threshold: float) -> np.ndarray:
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
                         maximum: float = 0.8) -> np.ndarray:
    map_x, map_y, inside = sampling_map(backward)
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

    def infer(self, previous_rgb: np.ndarray, current_rgb: np.ndarray):
        previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
        backward = self.backward.calc(current_gray, previous_gray, None).astype(np.float32)
        forward = None
        if self.forward is not None:
            forward = self.forward.calc(previous_gray, current_gray, None).astype(np.float32)
        return backward, forward, None


class SeaRaftFlow:
    def __init__(self, root: str, model_dir: str, device_name: str, bidirectional: bool):
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("PyTorch and safetensors are required for SEA-RAFT") from exc
        core_dir = os.path.join(root, "sea_raft_core")
        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "sea_raft_s_full.safetensors")
        for path in (core_dir, config_path, model_path):
            if not os.path.exists(path):
                fail(f"missing SEA-RAFT component: {path}")
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        from raft import RAFT
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        self.torch = torch
        self.device = torch.device(device_name)
        if self.device.type == "xpu" and not torch.xpu.is_available():
            fail("PyTorch XPU is unavailable")
        self.model = RAFT(SimpleNamespace(**config))
        self.model.load_state_dict(load_file(model_path), strict=True)
        self.model.to(self.device).eval()
        self.bidirectional_enabled = bidirectional
        print(f"[motion] PyTorch {torch.__version__}, {self.device}, SEA-RAFT S, "
              f"{'two-way' if bidirectional else 'one-way'}", file=sys.stderr, flush=True)

    def _uncertainty(self, info) -> np.ndarray:
        torch = self.torch
        weight = torch.softmax(info[:, :2], dim=1)
        raw_b = info[:, 2:]
        large = torch.exp(torch.clamp(raw_b[:, 0], min=0.0, max=6.0))
        small = torch.exp(torch.clamp(raw_b[:, 1], min=-6.0, max=0.0))
        expected = weight[:, 0] * large + weight[:, 1] * small
        normalized = torch.clamp((expected - 0.15) / 3.0, 0.0, 1.0)
        return normalized.float().cpu().numpy()

    def infer(self, previous_rgb: np.ndarray, current_rgb: np.ndarray):
        torch = self.torch
        with torch.inference_mode():
            previous = torch.from_numpy(np.ascontiguousarray(previous_rgb)).permute(2, 0, 1)
            current = torch.from_numpy(np.ascontiguousarray(current_rgb)).permute(2, 0, 1)
            if self.bidirectional_enabled:
                image1 = torch.stack((current, previous)).to(self.device, dtype=torch.float32)
                image2 = torch.stack((previous, current)).to(self.device, dtype=torch.float32)
            else:
                image1 = current.unsqueeze(0).to(self.device, dtype=torch.float32)
                image2 = previous.unsqueeze(0).to(self.device, dtype=torch.float32)
            output = self.model(image1, image2, test_mode=True)
            flow = output["final"].float().cpu().numpy().transpose(0, 2, 3, 1)
            uncertainty = self._uncertainty(output["info"][-1])
        return flow[0], (flow[1] if self.bidirectional_enabled else None), uncertainty[0]


@dataclass
class MotionResult:
    flow: np.ndarray
    depth: np.ndarray | None
    mask: np.ndarray | None
    confidence: np.ndarray
    scene_cut: bool
    metrics: dict[str, float]


class FrameAnalyzer:
    def __init__(self, flow_engine, depth_estimator: DepthEstimator | None, *,
                 temporal: float, consistency: float, dilation: int,
                 depth_edge: float, responsive_max: float,
                 photometric_confidence: bool = False):
        self.flow_engine = flow_engine
        self.depth_estimator = depth_estimator
        self.temporal = temporal
        self.consistency = consistency
        self.dilation = dilation
        self.depth_edge = depth_edge
        self.responsive_max = responsive_max
        self.photometric_confidence = photometric_confidence
        self.previous_rgb: np.ndarray | None = None
        self.previous_gray: np.ndarray | None = None
        self.previous_depth: np.ndarray | None = None

    def first(self, rgb: np.ndarray, with_mask: bool) -> MotionResult:
        height, width = rgb.shape[:2]
        depth = self.depth_estimator.infer(rgb) if self.depth_estimator else None
        self.previous_rgb = rgb.copy()
        self.previous_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self.previous_depth = depth
        return MotionResult(flow=np.zeros((height, width, 2), np.float32), depth=depth,
                            mask=np.full((height, width), self.responsive_max, np.float32) if with_mask else None,
                            confidence=np.ones((height, width), np.float32), scene_cut=True,
                            metrics={"reliable_fraction": 1.0})

    def next(self, rgb: np.ndarray, with_mask: bool, dilate_highres: bool) -> MotionResult:
        assert self.previous_rgb is not None and self.previous_gray is not None
        current_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        backward, forward, uncertainty = self.flow_engine.infer(self.previous_rgb, rgb)
        if forward is not None:
            reliable, confidence, _, _, error = flow_consistency(backward, forward, self.consistency)
        else:
            reliable, confidence, _, _, error = single_direction_confidence(
                backward, uncertainty,
                self.previous_gray if self.photometric_confidence else None,
                current_gray if self.photometric_confidence else None)
        reliable_fraction = float(np.mean(reliable))
        scene_cut, metrics = detect_scene_cut(self.previous_gray, current_gray, reliable_fraction)
        current_depth = self.depth_estimator.infer(rgb) if self.depth_estimator else None
        stable_depth = current_depth
        if current_depth is not None and self.previous_depth is not None and not scene_cut:
            stable_depth = stabilize_depth(current_depth, self.previous_depth, backward,
                                           reliable, confidence, self.temporal)
        if scene_cut:
            backward.fill(0.0)
            reliable.fill(True)
            confidence.fill(1.0)
        motion = backward
        if dilate_highres and stable_depth is not None:
            motion = depth_aware_dilate(backward, stable_depth, reliable,
                                        self.dilation, self.depth_edge)
        motion = np.nan_to_num(motion.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mask = None
        if with_mask:
            mask = make_responsive_mask(self.previous_gray, current_gray, backward,
                                        confidence, stable_depth, uncertainty,
                                        self.responsive_max)
            if scene_cut:
                mask.fill(self.responsive_max)
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
