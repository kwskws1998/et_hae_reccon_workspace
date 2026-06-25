from __future__ import annotations

import numpy as np
import pytest

from et_hae_reccon.heatmap import corrupt_trt_from_target, trt_to_heatmap, validate_heatmap


def test_trt_to_heatmap_sums_to_one() -> None:
    heatmap = trt_to_heatmap([0.0, 2.0, 4.0])
    assert np.isclose(float(heatmap.sum()), 1.0)
    assert np.all(heatmap >= 0.0)


def test_trt_to_heatmap_zero_case_is_finite() -> None:
    heatmap = trt_to_heatmap([0.0, 0.0, 0.0])
    assert np.isfinite(heatmap).all()
    assert np.isclose(float(heatmap.sum()), 1.0)


def test_validate_heatmap_rejects_masked_mass() -> None:
    with pytest.raises(ValueError, match="masked"):
        validate_heatmap(np.asarray([0.5, 0.5]), mask=np.asarray([True, False]))


def test_corrupt_trt_from_target_preserves_length() -> None:
    noisy = corrupt_trt_from_target([1.0, 2.0, 3.0], seed=1)
    assert noisy.shape == (3,)
    assert np.isfinite(noisy).all()
