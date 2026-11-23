pyhealth.metrics.value_set
===================================

Value-set-level metrics for :class:`~pyhealth.tasks.ValueSetCompletion`, from
RASC (Mukherjee et al., 2026). Each value set's predicted codes are scored
against its full true code set, so true codes that retrieval never surfaced
count against recall. The paper's value-set F1 is the mean of per-set F1
(``mean_f1``), which is not the F1 of the macro precision and recall
(``f1_of_macro``); report both. Bootstrap intervals resample whole value sets.

.. code-block:: python

    from pyhealth.metrics import (
        best_f1_threshold, value_set_bootstrap_ci, value_set_scores,
        value_set_summary,
    )

    threshold = best_f1_threshold(val_true, val_prob)
    scores = value_set_scores(
        [s["patient_id"] for s in test], [s["code"] for s in test],
        test_true, test_prob, [s["n_true_codes"] for s in test], threshold,
    )
    value_set_summary(scores)          # all test sets
    value_set_summary(scores[scores["retrieval_recall"] == 1])  # RR@K = 1
    value_set_bootstrap_ci(scores, metrics=["mean_f1"])

.. currentmodule:: pyhealth.metrics.value_set

.. autofunction:: best_f1_threshold

.. autofunction:: value_set_scores

.. autofunction:: value_set_summary

.. autofunction:: value_set_bootstrap_ci

.. autofunction:: value_set_paired_bootstrap_diff
