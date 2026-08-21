#!/usr/bin/env python3
"""Conservative five-frame motion/depth refinement for XeSS frame generation.

The filter does not synthesize or blend colour frames.  It transports the
already-computed per-pair motion fields into the centre frame, rejects
inconsistent candidates and only corrects the current field where temporal
support is strong.  This keeps XeFG as the image generator while giving it
cleaner geometry around difficult motion and occlusion boundaries.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np


def _grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32))


def _inside(x: np.ndarray, y: np.ndarray, width: int, height: int) -> np.ndarray:
    return ((x >= 0.0) & (x <= width - 1.0) &
            (y >= 0.0) & (y <= height - 1.0))


def _sample(array: np.ndarray, x: np.ndarray, y: np.ndarray,
            border: float = 0.0) -> np.ndarray:
    return cv2.remap(array, x, y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _crosses_cut(by_index: dict[int, object], start: int, end: int) -> bool:
    """Return whether correspondence from start to end crosses a reset."""
    low, high = sorted((start, end))
    return any(by_index[index].result.scene_cut
               for index in range(low + 1, high + 1)
               if index in by_index)


def _coordinates_in_frame(by_index: dict[int, object], centre: int, target: int,
                          grid_x: np.ndarray, grid_y: np.ndarray):
    """Map centre-frame pixels to their corresponding coordinates in target."""
    height, width = grid_x.shape
    x, y = grid_x.copy(), grid_y.copy()
    valid = np.ones((height, width), dtype=bool)
    if target < centre:
        for index in range(centre, target, -1):
            flow = by_index[index].result.flow
            valid &= _inside(x, y, width, height)
            sampled = _sample(flow, x, y)
            x += sampled[..., 0]
            y += sampled[..., 1]
            valid &= _inside(x, y, width, height)
    elif target > centre:
        for index in range(centre + 1, target + 1):
            flow = by_index[index].result.flow
            previous_x, previous_y = x, y
            guess_x, guess_y = previous_x.copy(), previous_y.copy()
            # Invert x_previous = x_current + backward(x_current).  Two
            # fixed-point iterations are accurate enough for adjacent video
            # frames and avoid a costly dense forward splat.
            for _ in range(2):
                sampled = _sample(flow, guess_x, guess_y)
                guess_x = previous_x - sampled[..., 0]
                guess_y = previous_y - sampled[..., 1]
            x, y = guess_x, guess_y
            valid &= _inside(x, y, width, height)
    return x, y, valid


def refine_five_frame(entries: list[object], centre: int, *,
                      motion_strength: float = 0.65,
                      depth_strength: float = 0.18,
                      analysis_width: int = 720):
    """Return a refined copy of the result belonging to ``centre``.

    Entries are small objects with ``index`` and ``result`` attributes.  Up to
    two past and two future results may be supplied; edges naturally use a
    shorter window.
    """
    by_index = {entry.index: entry for entry in entries}
    current = by_index[centre].result
    if centre == 0 or current.scene_cut or motion_strength <= 0.0:
        return current

    height, width = current.flow.shape[:2]
    if analysis_width > 0 and width > analysis_width:
        scale = analysis_width / width
        small_width = analysis_width
        small_height = max(2, int(round(height * scale)))
        small_entries = []
        for entry in entries:
            result = entry.result
            small_flow = cv2.resize(result.flow, (small_width, small_height),
                                    interpolation=cv2.INTER_AREA) * scale
            small_confidence = cv2.resize(result.confidence, (small_width, small_height),
                                          interpolation=cv2.INTER_AREA)
            small_depth = (cv2.resize(result.depth, (small_width, small_height),
                                      interpolation=cv2.INTER_AREA)
                           if result.depth is not None else None)
            small_result = replace(result, flow=small_flow.astype(np.float32),
                                   confidence=small_confidence.astype(np.float32),
                                   depth=(small_depth.astype(np.float32)
                                          if small_depth is not None else None))
            small_entries.append(SimpleNamespace(index=entry.index, result=small_result))
        small_current = next(entry.result for entry in small_entries if entry.index == centre)
        small_refined = refine_five_frame(
            small_entries, centre, motion_strength=motion_strength,
            depth_strength=depth_strength, analysis_width=0)
        small_correction = small_refined.flow - small_current.flow
        correction = cv2.resize(small_correction, (width, height),
                                interpolation=cv2.INTER_LINEAR) / scale
        refined_flow = np.nan_to_num(current.flow + correction, nan=0.0,
                                     posinf=0.0, neginf=0.0).astype(np.float32)
        refined_depth = current.depth
        if current.depth is not None and small_refined.depth is not None:
            depth_correction = small_refined.depth - small_current.depth
            refined_depth = np.clip(
                current.depth + cv2.resize(depth_correction, (width, height),
                                           interpolation=cv2.INTER_LINEAR),
                0.0, 1.0).astype(np.float32)
        metrics = dict(current.metrics)
        metrics.update(small_refined.metrics)
        metrics["five_frame_analysis_scale"] = scale
        return replace(current, flow=refined_flow, depth=refined_depth, metrics=metrics)

    grid_x, grid_y = _grid(height, width)
    centre_depth = current.depth
    flows: list[np.ndarray] = []
    depths: list[np.ndarray | None] = []
    weights: list[np.ndarray] = []
    current_slot = -1

    for index in sorted(by_index):
        entry = by_index[index]
        # result[index].flow describes index -> index-1.  A reset has no
        # meaningful velocity and must never vote in another frame.
        if index == 0 or entry.result.scene_cut or _crosses_cut(by_index, centre, index):
            continue
        x, y, valid = _coordinates_in_frame(by_index, centre, index, grid_x, grid_y)
        flow = entry.result.flow if index == centre else _sample(entry.result.flow, x, y)
        confidence = (entry.result.confidence if index == centre else
                      _sample(entry.result.confidence, x, y))
        weight = np.clip(confidence, 0.0, 1.0) * valid.astype(np.float32)
        weight *= 1.0 / (1.0 + 0.45 * abs(index - centre))

        aligned_depth = None
        if entry.result.depth is not None:
            aligned_depth = (entry.result.depth if index == centre else
                             _sample(entry.result.depth, x, y, border=0.5))
            if centre_depth is not None:
                # Relative monocular depth is most useful as an object-boundary
                # gate.  Do not allow another surface to vote through an edge.
                weight *= np.exp(-np.abs(aligned_depth - centre_depth) / 0.12)
        if index == centre:
            current_slot = len(flows)
            weight = np.maximum(weight, 0.30) * 1.35
        flows.append(flow.astype(np.float32, copy=False))
        depths.append(aligned_depth)
        weights.append(weight.astype(np.float32, copy=False))

    if current_slot < 0 or len(flows) < 2:
        return current

    flow_stack = np.stack(flows)
    weight_stack = np.stack(weights)
    total_weight = np.maximum(np.sum(weight_stack, axis=0), 1e-5)
    estimate = np.sum(flow_stack * weight_stack[..., None], axis=0) / total_weight[..., None]

    # Two robust reweighting passes suppress a bad neighbouring flow without
    # blindly median-filtering real acceleration.
    for _ in range(2):
        residual = np.linalg.norm(flow_stack - estimate[None], axis=3)
        magnitude = np.linalg.norm(estimate, axis=2)
        scale = 0.75 + 0.10 * magnitude
        robust = weight_stack / (1.0 + (residual / scale[None]) ** 2)
        robust_total = np.maximum(np.sum(robust, axis=0), 1e-5)
        estimate = np.sum(flow_stack * robust[..., None], axis=0) / robust_total[..., None]

    residual = np.linalg.norm(flow_stack - estimate[None], axis=3)
    robust_total = np.maximum(np.sum(robust, axis=0), 1e-5)
    dispersion = np.sum(residual * robust, axis=0) / robust_total
    current_delta = np.linalg.norm(current.flow - estimate, axis=2)
    magnitude = np.linalg.norm(current.flow, axis=2)
    neighbour_weight = np.sum(weight_stack, axis=0) - weight_stack[current_slot]
    support = np.clip(neighbour_weight / 1.15, 0.0, 1.0)
    stability = np.exp(-dispersion / (1.25 + 0.08 * magnitude))
    low_confidence = 1.0 - np.clip(current.confidence, 0.0, 1.0)
    outlier = np.clip((current_delta - (0.35 + 0.04 * magnitude)) /
                      (1.25 + 0.08 * magnitude), 0.0, 1.0)
    blend = motion_strength * support * stability * np.maximum(low_confidence, 0.40 * outlier)

    correction = estimate - current.flow
    correction_length = np.linalg.norm(correction, axis=2)
    correction_limit = 2.5 + 0.20 * magnitude
    correction_scale = np.minimum(1.0, correction_limit /
                                  np.maximum(correction_length, 1e-5))
    refined_flow = current.flow + correction * (blend * correction_scale)[..., None]
    refined_flow = np.nan_to_num(refined_flow.astype(np.float32), nan=0.0,
                                 posinf=0.0, neginf=0.0)

    refined_depth = centre_depth
    if centre_depth is not None and depth_strength > 0.0:
        depth_sum = np.zeros_like(centre_depth, dtype=np.float32)
        depth_weight = np.zeros_like(centre_depth, dtype=np.float32)
        for aligned_depth, weight in zip(depths, weight_stack):
            if aligned_depth is not None:
                depth_sum += aligned_depth * weight
                depth_weight += weight
        depth_target = depth_sum / np.maximum(depth_weight, 1e-5)
        kernel = np.ones((3, 3), np.uint8)
        depth_range = cv2.dilate(centre_depth, kernel) - cv2.erode(centre_depth, kernel)
        depth_blend = depth_strength * support * stability
        depth_blend *= np.clip(current.confidence, 0.0, 1.0)
        depth_blend *= np.exp(-depth_range / 0.035)
        refined_depth = np.clip(centre_depth * (1.0 - depth_blend) +
                                depth_target * depth_blend, 0.0, 1.0).astype(np.float32)

    metrics = dict(current.metrics)
    metrics["five_frame_corrected_fraction"] = float(np.mean(blend > 0.05))
    metrics["five_frame_mean_correction"] = float(np.mean(
        np.linalg.norm(refined_flow - current.flow, axis=2)))
    return replace(current, flow=refined_flow, depth=refined_depth, metrics=metrics)
