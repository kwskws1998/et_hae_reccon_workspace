"""Heatmap conversion and validation utilities."""

from __future__ import annotations

import numpy as np

from et_hae_reccon.constants import EPS


def clean_trt(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(array, 0.0)


def trt_to_heatmap(
    trt: np.ndarray | list[float],
    mask: np.ndarray | list[bool] | None = None,
    eps: float = EPS,
) -> np.ndarray:
    values = clean_trt(trt)
    if values.ndim != 1:
        raise ValueError("TRT values must be a 1D array.")
    if values.size == 0:
        raise ValueError("TRT values cannot be empty.")
    if mask is None:
        valid = np.ones(values.shape, dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != values.shape:
            raise ValueError("mask must have the same shape as trt.")
    weights = np.zeros_like(values, dtype=np.float64)
    weights[valid] = np.log1p(values[valid]) + eps
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        count = int(valid.sum())
        if count <= 0:
            raise ValueError("mask contains no valid positions.")
        weights[valid] = 1.0 / count
        return weights
    return weights / total


def validate_heatmap(
    heatmap: np.ndarray | list[float],
    mask: np.ndarray | list[bool] | None = None,
    atol: float = 1e-6,
) -> None:
    values = np.asarray(heatmap, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("heatmap must be a 1D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("heatmap contains non-finite values.")
    if np.any(values < -atol):
        raise ValueError("heatmap contains negative values.")
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != values.shape:
            raise ValueError("mask must have the same shape as heatmap.")
        if np.any(np.abs(values[~valid]) > atol):
            raise ValueError("masked heatmap positions must have zero mass.")
    total = float(values.sum())
    if abs(total - 1.0) > atol:
        raise ValueError(f"heatmap must sum to 1.0, got {total}.")


def corrupt_trt_from_target(
    target_trt: np.ndarray | list[float],
    seed: int,
    noise_std: float = 0.25,
    dropout_prob: float = 0.05,
) -> np.ndarray:
    values = clean_trt(target_trt)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=values.shape)
    noisy = np.maximum(values * np.exp(noise), 0.0)
    if dropout_prob > 0.0:
        keep = rng.random(values.shape) >= dropout_prob
        if np.any(keep):
            noisy = noisy * keep
    return noisy


def corrupt_heatmap_from_target(
    target_trt: np.ndarray | list[float],
    seed: int,
    noise_std: float = 0.25,
    dropout_prob: float = 0.05,
) -> np.ndarray:
    return corrupt_trt_from_target(target_trt, seed=seed, noise_std=noise_std, dropout_prob=dropout_prob)
