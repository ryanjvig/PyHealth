"""Value-set-level metrics for retrieval-augmented value set completion.

Author: Ryan Vig (NetID: rvig2)
Paper: Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value
    Set Authoring (Mukherjee et al., 2026)
Paper link: https://arxiv.org/abs/2604.14616
Description: Scores the codes predicted for each value set against its full
    true code set, so true codes that retrieval never surfaced count against
    recall, and aggregates over value sets with percentile bootstrap
    intervals that resample whole value sets. Works with the samples of
    :class:`~pyhealth.tasks.ValueSetCompletion`.

The paper's value-set F1 is the mean of per-set F1 (``mean_f1``). The F1 of
the macro precision and recall (``f1_of_macro``) is usually higher; report
both, since they are easy to confuse.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

SUMMARY_METRICS = (
    "macro_precision",
    "macro_recall",
    "mean_f1",
    "f1_of_macro",
    "retrieval_recall",
)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Returns the probability cutoff with the highest pair-level F1.

    Tune it on validation pairs and apply it to test pairs, as the paper does.

    Args:
        y_true: Binary labels, shape ``(n,)``.
        y_prob: Predicted probabilities, shape ``(n,)``.

    Returns:
        The threshold; predict positive when ``y_prob >= threshold``.

    Examples:
        >>> import numpy as np
        >>> best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
        0.35
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # The last precision/recall point has no threshold.
    return float(thresholds[int(np.argmax(f1[:-1]))])


def value_set_scores(
    set_ids: Sequence[str],
    codes: Sequence[str],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_true_codes: Sequence[int],
    threshold: float,
) -> pd.DataFrame:
    """Scores each value set's predicted codes against its true code set.

    Codes are counted once per value set, so a code retrieved under two code
    systems is one prediction. Recall divides by ``n_true_codes``, which
    includes true codes that are not among the candidates. A value set with
    no predicted code has precision 0.

    Args:
        set_ids: Value set id of each candidate pair (``patient_id``).
        codes: Candidate code of each pair.
        y_true: 1 when the code is in the value set.
        y_prob: Predicted probability of each pair.
        n_true_codes: Size of the pair's true value set; constant per set.
        threshold: Predict a code when ``y_prob >= threshold``.

    Returns:
        One row per value set, with columns ``set_id``, ``n_true_codes``,
        ``n_candidates``, ``n_predicted``, ``n_hits``, ``precision``,
        ``recall``, ``f1`` and ``retrieval_recall`` (share of true codes
        among the candidates).

    Raises:
        ValueError: If the inputs differ in length, or a value set has
            inconsistent ``n_true_codes`` or more true candidates than true
            codes.

    Examples:
        >>> scores = value_set_scores(
        ...     set_ids=["a", "a", "a", "b", "b"],
        ...     codes=["1", "2", "3", "1", "4"],
        ...     y_true=[1, 0, 1, 0, 1],
        ...     y_prob=[0.9, 0.8, 0.2, 0.1, 0.7],
        ...     n_true_codes=[4, 4, 4, 1, 1],
        ...     threshold=0.5,
        ... )
        >>> scores[["set_id", "precision", "recall", "retrieval_recall"]]
          set_id  precision  recall  retrieval_recall
        0      a        0.5    0.25               0.5
        1      b        1.0    1.00               1.0
    """
    lengths = {len(set_ids), len(codes), len(y_true), len(y_prob), len(n_true_codes)}
    if len(lengths) != 1:
        raise ValueError(
            "set_ids, codes, y_true, y_prob, n_true_codes differ in length."
        )
    pairs = pd.DataFrame(
        {
            "set_id": np.asarray(set_ids, dtype=object),
            "code": np.asarray(codes, dtype=object),
            "true": np.asarray(y_true).astype(int).ravel() == 1,
            "pred": np.asarray(y_prob, dtype=float).ravel() >= threshold,
            "n_true_codes": np.asarray(n_true_codes).astype(int).ravel(),
        }
    )
    sizes = pairs.groupby("set_id", sort=False)["n_true_codes"]
    if (sizes.nunique() > 1).any():
        raise ValueError("n_true_codes must be constant within each value set.")
    # One row per (set, code): a code is true or predicted under any system.
    per_code = pairs.groupby(["set_id", "code"], sort=False).agg(
        true=("true", "any"), pred=("pred", "any")
    )
    per_code["hit"] = per_code["true"] & per_code["pred"]
    counts = per_code.groupby(level="set_id", sort=False).agg(
        n_candidates=("true", "size"),
        n_retrieved_true=("true", "sum"),
        n_predicted=("pred", "sum"),
        n_hits=("hit", "sum"),
    )
    scores = sizes.first().to_frame().join(counts).reset_index()
    if (scores["n_retrieved_true"] > scores["n_true_codes"]).any():
        raise ValueError("A value set has more true candidates than n_true_codes.")
    hits = scores["n_hits"].to_numpy(dtype=float)
    predicted = scores["n_predicted"].to_numpy(dtype=float)
    total = scores["n_true_codes"].to_numpy(dtype=float)
    precision = np.divide(hits, predicted, out=np.zeros_like(hits), where=predicted > 0)
    recall = np.divide(hits, total, out=np.zeros_like(hits), where=total > 0)
    denom = precision + recall
    scores["precision"] = precision
    scores["recall"] = recall
    scores["f1"] = np.divide(
        2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0
    )
    scores["retrieval_recall"] = np.divide(
        scores["n_retrieved_true"].to_numpy(dtype=float),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    return scores.drop(columns="n_retrieved_true")[
        [
            "set_id",
            "n_true_codes",
            "n_candidates",
            "n_predicted",
            "n_hits",
            "precision",
            "recall",
            "f1",
            "retrieval_recall",
        ]
    ]


def value_set_summary(scores: pd.DataFrame) -> dict[str, float]:
    """Aggregates per-set scores over value sets.

    Args:
        scores: Output of :func:`value_set_scores`, or a subset of its rows.

    Returns:
        ``n_value_sets`` and each of ``SUMMARY_METRICS``: ``macro_precision``
        and ``macro_recall`` (means over sets), ``mean_f1`` (mean of per-set
        F1, the paper's value-set F1), ``f1_of_macro`` (harmonic mean of the
        macro precision and recall) and ``retrieval_recall`` (mean over sets).
        Metrics are NaN when ``scores`` is empty.

    Examples:
        >>> import pandas as pd
        >>> scores = pd.DataFrame({
        ...     "precision": [1.0, 0.2], "recall": [0.2, 1.0],
        ...     "f1": [1 / 3, 1 / 3], "retrieval_recall": [1.0, 1.0],
        ... })
        >>> summary = value_set_summary(scores)
        >>> round(summary["mean_f1"], 3), round(summary["f1_of_macro"], 3)
        (0.333, 0.6)
    """
    if len(scores) == 0:
        return {"n_value_sets": 0, **{m: float("nan") for m in SUMMARY_METRICS}}
    p = float(scores["precision"].mean())
    r = float(scores["recall"].mean())
    return {
        "n_value_sets": len(scores),
        "macro_precision": p,
        "macro_recall": r,
        "mean_f1": float(scores["f1"].mean()),
        "f1_of_macro": 2 * p * r / (p + r) if p + r > 0 else 0.0,
        "retrieval_recall": float(scores["retrieval_recall"].mean()),
    }


def _percentile_ci(
    estimate: float, stats: np.ndarray, n_boot: int, alpha: float
) -> dict[str, float]:
    lower, upper = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(estimate),
        "lower": float(lower),
        "upper": float(upper),
        "n_boot": n_boot,
    }


def _bootstrap(
    n_sets: int,
    statistic: Callable[[np.ndarray], dict[str, float]],
    metrics: Sequence[str],
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict[str, dict[str, float]]:
    if n_sets == 0:
        raise ValueError("Need at least one value set to bootstrap.")
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}.")
    unknown = set(metrics) - set(SUMMARY_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics {sorted(unknown)}; use {SUMMARY_METRICS}.")
    rng = np.random.default_rng(seed)
    draws = [statistic(rng.integers(0, n_sets, size=n_sets)) for _ in range(n_boot)]
    estimate = statistic(np.arange(n_sets))
    return {
        m: _percentile_ci(estimate[m], np.array([d[m] for d in draws]), n_boot, alpha)
        for m in metrics
    }


def value_set_bootstrap_ci(
    scores: pd.DataFrame,
    metrics: Sequence[str] = SUMMARY_METRICS,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap intervals of value-set metrics over value sets.

    Args:
        scores: Output of :func:`value_set_scores`.
        metrics: Names from ``SUMMARY_METRICS``.
        n_boot: Number of resamples of the value sets.
        seed: Random seed; results are deterministic given it.
        alpha: 1 - confidence level (0.05 gives a 95% interval).

    Returns:
        For each metric, ``estimate`` (on all sets), ``lower``, ``upper`` and
        ``n_boot``.

    Raises:
        ValueError: If ``scores`` is empty, ``n_boot`` is not positive or a
            metric is unknown.

    Examples:
        >>> import pandas as pd
        >>> scores = pd.DataFrame({
        ...     "precision": [0.5, 1.0, 0.0], "recall": [0.25, 1.0, 0.0],
        ...     "f1": [1 / 3, 1.0, 0.0], "retrieval_recall": [0.5, 1.0, 0.5],
        ... })
        >>> ci = value_set_bootstrap_ci(scores, metrics=["mean_f1"], n_boot=200)
        >>> f1 = ci["mean_f1"]
        >>> f1["lower"] <= f1["estimate"] <= f1["upper"]
        True
    """
    scores = scores.reset_index(drop=True)
    return _bootstrap(
        len(scores),
        lambda idx: value_set_summary(scores.iloc[idx]),
        metrics,
        n_boot,
        seed,
        alpha,
    )


def value_set_paired_bootstrap_diff(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    metrics: Sequence[str] = SUMMARY_METRICS,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Bootstrap intervals of metric(A) - metric(B) on the same value sets.

    Both models are scored on identical resamples of the value sets, so the
    interval reflects their difference. An interval that excludes 0
    indicates a difference at level ``alpha``.

    Args:
        scores_a: Model A's :func:`value_set_scores`.
        scores_b: Model B's :func:`value_set_scores`, over the same set ids.
        metrics: Names from ``SUMMARY_METRICS``.
        n_boot: Number of resamples of the value sets.
        seed: Random seed; results are deterministic given it.
        alpha: 1 - confidence level.

    Returns:
        For each metric, ``estimate`` (A - B on all sets), ``lower``,
        ``upper`` and ``n_boot``.

    Raises:
        ValueError: If the two frames cover different value sets, or as in
            :func:`value_set_bootstrap_ci`.

    Examples:
        >>> import pandas as pd
        >>> a = pd.DataFrame({
        ...     "set_id": ["x", "y"], "precision": [1.0, 0.5], "recall": [0.5, 0.5],
        ...     "f1": [2 / 3, 0.5], "retrieval_recall": [1.0, 1.0],
        ... })
        >>> b = a.assign(f1=[0.5, 0.25])
        >>> diff = value_set_paired_bootstrap_diff(a, b, metrics=["mean_f1"], n_boot=50)
        >>> round(diff["mean_f1"]["estimate"], 4)
        0.2083
    """
    if set(scores_a["set_id"]) != set(scores_b["set_id"]) or len(scores_a) != len(
        scores_b
    ):
        raise ValueError("scores_a and scores_b must cover the same value sets.")
    a = scores_a.sort_values("set_id").reset_index(drop=True)
    b = scores_b.sort_values("set_id").reset_index(drop=True)

    def diff(idx: np.ndarray) -> dict[str, float]:
        sa, sb = value_set_summary(a.iloc[idx]), value_set_summary(b.iloc[idx])
        return {m: sa[m] - sb[m] for m in SUMMARY_METRICS}

    return _bootstrap(len(a), diff, metrics, n_boot, seed, alpha)
