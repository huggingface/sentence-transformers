from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


def _check_n_resamples(n_resamples: int) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError(f"n_resamples must be a positive integer, got {n_resamples!r}.")


def _check_confidence_level(confidence_level: float) -> None:
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be strictly between 0 and 1, got {confidence_level!r}.")


def _check_indices(indices: np.ndarray, n_samples: int) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.size == 0:
        raise ValueError(
            f"indices must be a non-empty 2-D array of shape (n_resamples, n_samples), got shape {indices.shape}."
        )
    if not np.issubdtype(indices.dtype, np.integer) or indices.min() < 0 or indices.max() >= n_samples:
        raise ValueError(f"indices must contain integers in [0, {n_samples}) to index the {n_samples} values.")
    return indices


def _draw_indices(rng: np.random.Generator, n_samples: int, n_resamples: int) -> np.ndarray:
    return rng.integers(0, n_samples, size=(n_resamples, n_samples))


def _percentile_interval(statistics: np.ndarray, confidence_level: float) -> tuple[float, float]:
    low, high = np.percentile(statistics, [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100])
    return float(low), float(high)


def bootstrap_indices(n_samples: int, n_resamples: int, seed: int | None = 42) -> np.ndarray:
    """
    Draws the sample indices of ``n_resamples`` bootstrap resamples, each of ``n_samples`` samples drawn with replacement.

    Args:
        n_samples (int): Number of samples to resample from.
        n_resamples (int): Number of bootstrap resamples to draw.
        seed (int, optional): Seed for ``np.random.default_rng``. Defaults to 42.

    Returns:
        np.ndarray: Integer array of shape ``(n_resamples, n_samples)`` with values in ``[0, n_samples)``.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")
    _check_n_resamples(n_resamples)
    return _draw_indices(np.random.default_rng(seed), n_samples, n_resamples)


def bootstrap_confidence_interval(
    values: Sequence[float] | np.ndarray | Sequence[np.ndarray],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = 42,
    statistic: Callable[[np.ndarray], float] | None = None,
    indices: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval of the mean of ``values``, obtained by resampling the samples with
    replacement, e.g. the per-query values of a retrieval metric over the queries.

    Passing a sequence of arrays treats each array as a stratum (e.g. the per-query values of one dataset each):
    every stratum is resampled independently, its mean is taken per resample, and ``statistic`` combines the
    per-stratum means of each resample into the value whose percentiles form the interval. With a single stratum
    and the default ``statistic`` the result equals the plain call.

    Args:
        values: A 1-D array of per-sample values, or a sequence of such arrays (strata). Empty strata are skipped.
        n_resamples (int): Number of bootstrap resamples. Defaults to 1000.
        confidence_level (float): Confidence level of the interval, strictly between 0 and 1. Defaults to 0.95.
        seed (int, optional): Seed of the resampling. Defaults to 42.
        statistic (Callable[[np.ndarray], float], optional): Only used with strata: maps the 1-D array of
            per-stratum means of one resample to a scalar. Defaults to ``np.mean``.
        indices (np.ndarray, optional): Precomputed ``(n_resamples, n_samples)`` resample indices as returned by
            :func:`bootstrap_indices`, e.g. to share one draw across several metrics of the same samples so their
            intervals are comparable. Only supported for a single (non-stratified) array of values; ``n_resamples``
            and ``seed`` are then ignored. Defaults to None.

    Returns:
        tuple[float, float]: The lower and upper bound of the interval.
    """
    _check_confidence_level(confidence_level)
    if len(values) == 0:
        raise ValueError("values must not be empty.")
    stratified = np.ndim(values[0]) > 0
    strata = [np.asarray(stratum, dtype=float) for stratum in (values if stratified else [values])]
    if any(stratum.ndim != 1 for stratum in strata):
        raise ValueError("values must be a 1-D array of per-sample values, or a sequence of such arrays (strata).")
    strata = [stratum for stratum in strata if stratum.size > 0]
    if not strata:
        raise ValueError("values must contain at least one non-empty array.")

    if indices is not None:
        if len(strata) != 1:
            raise ValueError("indices can only be given for a single (non-stratified) array of values.")
        resample_indices = [_check_indices(indices, len(strata[0]))]
    else:
        _check_n_resamples(n_resamples)
        rng = np.random.default_rng(seed)
        resample_indices = [_draw_indices(rng, len(stratum), n_resamples) for stratum in strata]

    # (n_resamples, n_strata): the mean of every stratum in every resample
    stratum_means = np.stack(
        [stratum[stratum_indices].mean(axis=1) for stratum, stratum_indices in zip(strata, resample_indices)], axis=1
    )
    if statistic is None:
        statistics = stratum_means.mean(axis=1)
    else:
        statistics = np.asarray([statistic(resample_means) for resample_means in stratum_means], dtype=float)
    return _percentile_interval(statistics, confidence_level)


@dataclass
class PairedBootstrapTestResult:
    """
    Result of :func:`paired_bootstrap_test`.

    Attributes:
        difference (float): Mean of ``values_b - values_a``, i.e. how much higher B scores than A.
        ci_low (float): Lower bound of the bootstrap confidence interval of ``difference``.
        ci_high (float): Upper bound of the bootstrap confidence interval of ``difference``.
        p_value (float): Two-sided bootstrap p-value of the difference being zero. This is an approximation whose
            resolution is limited by ``n_resamples``; the confidence interval is the primary output.
        n_samples (int): Number of paired samples.
        n_resamples (int): Number of bootstrap resamples.
        confidence_level (float): Confidence level of the interval.
    """

    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n_samples: int
    n_resamples: int
    confidence_level: float


def paired_bootstrap_test(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = 42,
) -> PairedBootstrapTestResult:
    """
    Paired bootstrap test of the mean difference between two sets of per-sample scores, e.g. the per-query
    nDCG@10 of two models on the same queries as found in ``InformationRetrievalEvaluator.per_query_metrics``.

    The per-sample differences ``values_b - values_a`` are resampled jointly, which keeps the pairing of the two
    score sets on the same samples and makes the test far more sensitive than comparing two separate intervals.
    The confidence interval is the primary output: the difference is significant at ``confidence_level`` when the
    interval excludes zero. The p-value is the two-sided fraction of resampled mean differences on the other side
    of zero, ``2 * min(P(diff* <= 0), P(diff* >= 0))`` clipped to ``[0, 1]``, which is an approximation whose
    resolution is limited by ``n_resamples``: ``0.0`` means that no resample crossed zero.

    Args:
        values_a (Sequence[float] | np.ndarray): Per-sample scores of the first system (the baseline).
        values_b (Sequence[float] | np.ndarray): Per-sample scores of the second system, in the same sample order.
        n_resamples (int): Number of bootstrap resamples. Defaults to 1000.
        confidence_level (float): Confidence level of the interval, strictly between 0 and 1. Defaults to 0.95.
        seed (int, optional): Seed of the resampling. Defaults to 42.

    Returns:
        PairedBootstrapTestResult: The mean difference, its confidence interval and the approximate p-value.
    """
    _check_confidence_level(confidence_level)
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.ndim != 1 or values_b.ndim != 1:
        raise ValueError("values_a and values_b must be 1-D arrays of per-sample scores.")
    if len(values_a) != len(values_b):
        raise ValueError(
            "values_a and values_b must have the same length as they are paired per sample, "
            f"got {len(values_a)} and {len(values_b)}."
        )
    if len(values_a) == 0:
        raise ValueError("values_a and values_b must not be empty.")

    differences = values_b - values_a
    resampled = differences[bootstrap_indices(len(differences), n_resamples, seed)].mean(axis=1)
    ci_low, ci_high = _percentile_interval(resampled, confidence_level)
    p_value = min(1.0, 2 * min(float(np.mean(resampled <= 0)), float(np.mean(resampled >= 0))))
    return PairedBootstrapTestResult(
        difference=float(differences.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_samples=len(differences),
        n_resamples=n_resamples,
        confidence_level=confidence_level,
    )
