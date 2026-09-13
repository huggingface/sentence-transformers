from __future__ import annotations

import numpy as np
import pytest

from sentence_transformers.util import (
    PairedBootstrapTestResult,
    bootstrap_confidence_interval,
    bootstrap_indices,
    paired_bootstrap_test,
)


@pytest.fixture
def random_values() -> np.ndarray:
    return np.random.default_rng(0).random(50)


def test_bootstrap_indices_shape_and_range() -> None:
    indices = bootstrap_indices(n_samples=7, n_resamples=25, seed=3)
    assert indices.shape == (25, 7)
    assert np.issubdtype(indices.dtype, np.integer)
    assert indices.min() >= 0
    assert indices.max() < 7
    # Sampling is with replacement, so some resample repeats an index.
    assert any(len(set(row)) < 7 for row in indices.tolist())


def test_bootstrap_indices_seed_determinism() -> None:
    assert np.array_equal(bootstrap_indices(10, 20, seed=1), bootstrap_indices(10, 20, seed=1))
    assert not np.array_equal(bootstrap_indices(10, 20, seed=1), bootstrap_indices(10, 20, seed=2))


def test_constant_values_give_degenerate_interval() -> None:
    low, high = bootstrap_confidence_interval([0.7] * 20, n_resamples=100)
    # Every resample is the same multiset, so both percentiles are the (rounded) mean of 20 x 0.7.
    assert low == high == pytest.approx(0.7)
    assert isinstance(low, float) and isinstance(high, float)
    assert bootstrap_confidence_interval([0.5] * 20, n_resamples=100) == (0.5, 0.5)


def test_interval_contains_point_mean(random_values: np.ndarray) -> None:
    low, high = bootstrap_confidence_interval(random_values, n_resamples=500)
    assert low <= random_values.mean() <= high
    assert low < high


def test_bounds_within_value_range(random_values: np.ndarray) -> None:
    low, high = bootstrap_confidence_interval(random_values, n_resamples=500, confidence_level=0.999)
    assert random_values.min() <= low <= high <= random_values.max()


def test_seed_determinism(random_values: np.ndarray) -> None:
    first = bootstrap_confidence_interval(random_values, n_resamples=200, seed=7)
    second = bootstrap_confidence_interval(random_values, n_resamples=200, seed=7)
    other_seed = bootstrap_confidence_interval(random_values, n_resamples=200, seed=8)
    assert first == second
    assert first != other_seed


def test_wider_confidence_level_gives_wider_interval(random_values: np.ndarray) -> None:
    narrow = bootstrap_confidence_interval(random_values, n_resamples=500, confidence_level=0.5)
    wide = bootstrap_confidence_interval(random_values, n_resamples=500, confidence_level=0.99)
    assert wide[0] <= narrow[0] <= narrow[1] <= wide[1]
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def test_percentile_formula_exact_with_explicit_indices() -> None:
    # Resample means are [0.0, 0.5, 1.0, 0.5]: sorted [0, 0.5, 0.5, 1], so with linear interpolation
    # the 2.5th percentile sits at position 0.075 (0.0375) and the 97.5th at 2.925 (0.9625).
    indices = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    interval = bootstrap_confidence_interval([0.0, 1.0], indices=indices)
    assert interval == pytest.approx((0.0375, 0.9625), abs=1e-12)
    # Positions 0.75 and 2.25 for the 25th and 75th percentile.
    interval = bootstrap_confidence_interval([0.0, 1.0], indices=indices, confidence_level=0.5)
    assert interval == pytest.approx((0.375, 0.625), abs=1e-12)


def test_explicit_indices_ignore_seed_and_n_resamples(random_values: np.ndarray) -> None:
    indices = bootstrap_indices(len(random_values), n_resamples=100, seed=5)
    from_indices = bootstrap_confidence_interval(random_values, indices=indices, seed=1, n_resamples=3)
    assert from_indices == bootstrap_confidence_interval(random_values, indices=indices, seed=2, n_resamples=999)
    assert from_indices == bootstrap_confidence_interval(random_values, n_resamples=100, seed=5)


def test_single_stratum_equals_plain_call(random_values: np.ndarray) -> None:
    plain = bootstrap_confidence_interval(random_values, n_resamples=300, seed=11)
    stratified = bootstrap_confidence_interval([random_values], n_resamples=300, seed=11)
    assert stratified == plain


def test_stratified_uses_independent_resamples_per_stratum(random_values: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    strata = [rng.random(30), rng.random(20) + 1.0]
    low, high = bootstrap_confidence_interval(strata, n_resamples=500, seed=3)
    point = np.mean([stratum.mean() for stratum in strata])
    assert low <= point <= high
    # A resample never mixes strata: the aggregate of the two means stays between the two stratum means.
    assert strata[0].mean() < low and high < strata[1].mean()

    # The first stratum is drawn exactly like the plain call and a constant stratum adds no variance,
    # so the aggregate interval is the plain interval shifted towards the constant.
    plain_low, plain_high = bootstrap_confidence_interval(random_values, n_resamples=300, seed=3)
    shifted = bootstrap_confidence_interval([random_values, np.full(10, 2.0)], n_resamples=300, seed=3)
    assert shifted == pytest.approx(((plain_low + 2.0) / 2, (plain_high + 2.0) / 2))


def test_stratified_statistic_max_is_monotone() -> None:
    rng = np.random.default_rng(2)
    strata = [rng.random(30), rng.random(30) + 0.5, rng.random(30) + 1.0]
    mean_ci = bootstrap_confidence_interval(strata, n_resamples=500, seed=4)
    max_ci = bootstrap_confidence_interval(strata, n_resamples=500, seed=4, statistic=np.max)
    min_ci = bootstrap_confidence_interval(strata, n_resamples=500, seed=4, statistic=np.min)
    assert min_ci[0] <= mean_ci[0] <= max_ci[0]
    assert min_ci[1] <= mean_ci[1] <= max_ci[1]
    # The strata are shifted by 0.5 each, so the extreme stratum means dominate the max / min statistic.
    assert max_ci[0] <= strata[2].mean() <= max_ci[1]
    assert min_ci[0] <= strata[0].mean() <= min_ci[1]
    assert max_ci[0] > strata[1].mean() > min_ci[1]


def test_stratified_skips_empty_strata(random_values: np.ndarray) -> None:
    expected = bootstrap_confidence_interval([random_values], n_resamples=100, seed=9)
    assert bootstrap_confidence_interval([random_values, np.array([])], n_resamples=100, seed=9) == expected
    assert bootstrap_confidence_interval([[], random_values], n_resamples=100, seed=9) == expected
    with pytest.raises(ValueError):
        bootstrap_confidence_interval([np.array([]), []], n_resamples=100)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, 1.5, -0.1])
def test_invalid_confidence_level(confidence_level: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_confidence_interval([0.1, 0.2, 0.3], confidence_level=confidence_level)
    with pytest.raises(ValueError, match="confidence_level"):
        paired_bootstrap_test([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], confidence_level=confidence_level)


def test_empty_values_raise() -> None:
    with pytest.raises(ValueError):
        bootstrap_confidence_interval([])
    with pytest.raises(ValueError):
        bootstrap_confidence_interval(np.array([]))
    with pytest.raises(ValueError):
        paired_bootstrap_test([], [])


@pytest.mark.parametrize("n_resamples", [0, -5])
def test_invalid_n_resamples(n_resamples: int) -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_confidence_interval([0.1, 0.2, 0.3], n_resamples=n_resamples)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_indices(3, n_resamples=n_resamples)
    with pytest.raises(ValueError, match="n_resamples"):
        paired_bootstrap_test([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], n_resamples=n_resamples)


def test_paired_test_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="length"):
        paired_bootstrap_test([0.1, 0.2, 0.3], [0.1, 0.2])


def test_paired_test_identical_arrays(random_values: np.ndarray) -> None:
    result = paired_bootstrap_test(random_values, random_values.copy(), n_resamples=200)
    assert isinstance(result, PairedBootstrapTestResult)
    assert result.difference == 0.0
    assert (result.ci_low, result.ci_high) == (0.0, 0.0)
    assert result.p_value == 1.0
    assert result.n_samples == len(random_values)
    assert result.n_resamples == 200
    assert result.confidence_level == 0.95


def test_paired_test_detects_clear_improvement(random_values: np.ndarray) -> None:
    better = random_values + 0.5 + np.random.default_rng(3).normal(0.0, 0.1, size=len(random_values))
    result = paired_bootstrap_test(random_values, better, n_resamples=1000)
    assert result.difference == pytest.approx(np.mean(better - random_values))
    assert result.difference == pytest.approx(0.5, abs=0.1)
    assert 0.0 < result.ci_low <= result.difference <= result.ci_high
    assert result.p_value < 0.05
    assert 0.0 <= result.p_value <= 1.0


def test_paired_test_symmetry(random_values: np.ndarray) -> None:
    other = random_values + np.random.default_rng(4).normal(0.0, 0.2, size=len(random_values))
    forward = paired_bootstrap_test(random_values, other, n_resamples=500, seed=6)
    backward = paired_bootstrap_test(other, random_values, n_resamples=500, seed=6)
    assert backward.difference == -forward.difference
    assert backward.ci_low == pytest.approx(-forward.ci_high)
    assert backward.ci_high == pytest.approx(-forward.ci_low)
    assert backward.p_value == pytest.approx(forward.p_value)


def test_paired_test_is_paired_not_independent() -> None:
    # Per-sample scores vary a lot, but B is consistently a little better on every sample: two
    # separate intervals overlap widely while the paired difference is clearly positive.
    rng = np.random.default_rng(5)
    values_a = rng.random(40)
    values_b = values_a + 0.02
    a_low, a_high = bootstrap_confidence_interval(values_a, n_resamples=500)
    b_low, b_high = bootstrap_confidence_interval(values_b, n_resamples=500)
    assert b_low < a_high
    result = paired_bootstrap_test(values_a, values_b, n_resamples=500)
    assert result.ci_low == pytest.approx(0.02)
    assert result.ci_high == pytest.approx(0.02)
    assert result.p_value == 0.0


def test_paired_test_seed_determinism(random_values: np.ndarray) -> None:
    other = random_values + np.random.default_rng(7).normal(0.0, 0.2, size=len(random_values))
    assert paired_bootstrap_test(random_values, other, seed=1) == paired_bootstrap_test(random_values, other, seed=1)
    assert paired_bootstrap_test(random_values, other, seed=1) != paired_bootstrap_test(random_values, other, seed=2)
