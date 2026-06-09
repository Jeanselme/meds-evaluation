"""Methods for evaluating different types of tasks and subpopulations on a standard set of metrics.

Most metrics will be directly based on sklearn implementation, and a standard set will be defined for
binary metrics initially. (TODO: add multiclass, multilabel, regression, ... evaluation).

See
    https://scikit-learn.org/stable/api/sklearn.metrics.html
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

Additionally, functionality for evaluating metrics on a per-sample vs per-subject basis is provided to
ensure balanced representation of all subjects in the dataset.

TODO: fairness functionality and filtering populations based on complex user-defined criteria.
"""

import logging
import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from aif360.sklearn.metrics import (
    average_odds_difference,
    conditional_demographic_disparity,
    equal_opportunity_difference,
    statistical_parity_difference,
)

from meds_evaluation.schema import (
    BOOLEAN_VALUE_FIELD,
    GROUPS_SCHEMA_DICT,
    PREDICTED_BOOLEAN_PROBABILITY_FIELD,
    PREDICTED_BOOLEAN_VALUE_FIELD,
    SUBJECT_ID_FIELD,
    validate_binary_classification_schema,
    validate_group_schema,
)
from meds_evaluation.utils import _resample

logger = logging.getLogger(__name__)

# Minimum requirements for a subgroup to be evaluated
MIN_SUBGROUP_SAMPLES = 10
MIN_POSITIVE_RATE = 0.0001
MAX_POSITIVE_RATE = 0.9999


def _is_valid_subgroup(true_values: ArrayLike) -> bool:
    """Returns False if the subgroup is too small or has only one class.

    Args:
        true_values: true binary labels for the subgroup.

    Returns:
        True if the subgroup meets minimum sample and class balance requirements.
    """
    if len(true_values) < MIN_SUBGROUP_SAMPLES:
        return False
    positive_rate = np.mean(np.asarray(true_values))
    return MIN_POSITIVE_RATE <= positive_rate <= MAX_POSITIVE_RATE


def _compute_ece(
    true_values: ArrayLike,
    predicted_probabilities: ArrayLike,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE).

    Bins predictions into equal-width intervals over [0, 1], then computes a
    sample-weighted average of the absolute difference between mean confidence
    and observed accuracy per bin. Fixed [0, 1] binning is intentional — a
    model that never predicts outside [0.3, 0.7] should reflect that
    underconfidence in its ECE.

    Args:
        true_values: true binary labels.
        predicted_probabilities: predicted probabilities in [0, 1].
        n_bins: number of equal-width bins.

    Returns:
        ECE as a float between 0 and 1.
    """
    predicted_probabilities = np.asarray(predicted_probabilities)
    true_values = np.asarray(true_values)
    n = len(true_values)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Exclude outer edges so values at exactly 0.0 and 1.0 fall into the
    # first and last bins respectively rather than out-of-range.
    bin_indices = np.digitize(predicted_probabilities, bin_edges[1:-1])

    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_indices == b
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        bin_confidence = predicted_probabilities[in_bin].mean()
        bin_accuracy = true_values[in_bin].mean()
        ece += (bin_count / n) * abs(bin_confidence - bin_accuracy)

    return ece


def _get_binary_classification_metrics(
    true_values: ArrayLike,
    predicted_values: ArrayLike | None,
    predicted_probabilities: ArrayLike | None,
) -> dict[str, float]:
    """Calculates binary classification metrics from true and predicted values.

    Args:
        true_values: true binary labels.
        predicted_values: predicted binary labels. If None, threshold-based
            metrics (accuracy, F1) are skipped.
        predicted_probabilities: predicted probabilities. If None,
            probability-based metrics (AUC, ECE, Brier) are skipped.

    Returns:
        A dictionary mapping metric names to their scalar values.
    """
    if not _is_valid_subgroup(true_values):
        logger.warning(
            f"Skipping metrics: n={len(true_values)}, "
            f"positive_rate={np.mean(true_values):.2f} — subgroup too small or single-class."
        )
        return {}

    results = {}

    if predicted_values is not None:
        results["binary_accuracy"] = accuracy_score(true_values, predicted_values)
        results["f1_score"] = f1_score(true_values, predicted_values)

    if predicted_probabilities is not None:
        results["roc_auc_score"] = roc_auc_score(true_values, predicted_probabilities)
        results["average_precision_score"] = average_precision_score(true_values, predicted_probabilities)
        results["brier_score"] = brier_score_loss(true_values, predicted_probabilities)
        results["ece"] = _compute_ece(true_values, predicted_probabilities)

    return results


def _get_fairness_binary_classification_metrics(
    true_values: ArrayLike,
    predicted_probabilities: ArrayLike | None,
    group_membership: ArrayLike | None,
) -> dict[str, float]:
    """Calculates fairness metrics from true values, predicted probabilities, and group membership.

    Args:
        true_values: true binary labels.
        predicted_probabilities: predicted probabilities (used as soft predictions
            for AIF360 fairness metrics).
        group_membership: boolean array indicating protected group membership.

    Returns:
        A dictionary mapping fairness metric names to their scalar values.
    """
    results = {}

    if predicted_probabilities is None:
        return results

    fairness_metrics = {
        "average_odds_difference": lambda: average_odds_difference(
            true_values, predicted_probabilities, prot_attr=group_membership
        ),
        "conditional_demographic_disparity": lambda: conditional_demographic_disparity(
            true_values, predicted_probabilities, prot_attr=group_membership
        ),
        "equal_opportunity_difference": lambda: equal_opportunity_difference(
            true_values, predicted_probabilities, prot_attr=group_membership, priv_group=True
        ),
        "statistical_parity_difference": lambda: statistical_parity_difference(
            true_values, predicted_probabilities, prot_attr=group_membership, priv_group=True
        ),
    }

    for name, fn in fairness_metrics.items():
        try:
            results[name] = fn()
        except Exception as e:
            logger.warning(f"Fairness metric '{name}' failed: {e}")

    return results


def evaluate_bootstrapped_binary_classification(
    predictions: pl.DataFrame,
    groups: pl.DataFrame,
    bootstrapping: int = 100,
) -> dict[str, dict[str, float]]:
    """Evaluates binary classification predictions with bootstrap confidence intervals.

    Computes metrics over the full population and per subgroup defined in `groups`,
    including difference metrics between each subgroup and its complement.
    Only subgroups meeting minimum sample size and class balance thresholds are evaluated.

    Args:
        predictions: DataFrame following the MEDS label schema with additional
            columns for predicted_boolean_value and predicted_boolean_probability.
        groups: DataFrame containing group membership columns joined on subject_id.
        bootstrapping: number of bootstrap resamples for confidence interval estimation.

    Returns:
        Nested dictionary of {group_key: {mean_<metric>: float, std_<metric>: float}}.

    Raises:
        ValueError: if bootstrapping is not a positive integer, or if predictions
            or groups do not conform to the expected schema.
    """
    if bootstrapping <= 0:
        raise ValueError("bootstrapping must be a positive integer.")

    validate_binary_classification_schema(predictions)
    validate_group_schema(groups)

    boot_res = {}
    all_keys: set[str] = set()

    for bi in range(bootstrapping):
        resampled_predictions = _resample(predictions, random_seed=bi)
        resampled_groups = (
            groups
            .join(resampled_predictions, how="right", on=SUBJECT_ID_FIELD)
            .select(groups.columns)
        )

        true_values = resampled_predictions[BOOLEAN_VALUE_FIELD]
        predicted_values = resampled_predictions[PREDICTED_BOOLEAN_VALUE_FIELD]
        predicted_probabilities = resampled_predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD]

        if predicted_values.is_null().all():
            predicted_values = None
        if predicted_probabilities.is_null().all():
            predicted_probabilities = None

        boot_res[bi] = {
            "all": _get_binary_classification_metrics(true_values, predicted_values, predicted_probabilities)
        }
        all_keys.update(boot_res[bi]["all"].keys())

        for group in groups.columns:
            if group in GROUPS_SCHEMA_DICT:
                continue

            for group_value in resampled_groups[group].drop_nulls().unique():
                mask = resampled_groups[group] == group_value
                complement_mask = resampled_groups[group] != group_value

                group_true = true_values.filter(mask)
                complement_true = true_values.filter(complement_mask)

                # Only compute difference if both sides are valid
                group_valid = _is_valid_subgroup(group_true)
                complement_valid = _is_valid_subgroup(complement_true)

                group_key = f"{group}_{group_value}"
                diff_key = f"{group_key}_difference"

                if group_valid:
                    group_metrics = _get_binary_classification_metrics(
                        group_true,
                        predicted_values.filter(mask) if predicted_values is not None else None,
                        predicted_probabilities.filter(mask),
                    )
                    group_metrics.update(_get_fairness_binary_classification_metrics(
                        group_true.to_pandas(),
                        predicted_probabilities.filter(mask).to_pandas(),
                        mask.to_pandas().fillna(False),
                    ))
                    boot_res[bi][group_key] = group_metrics
                    all_keys.update(group_metrics.keys())

                if group_valid and complement_valid:
                    complement_metrics = _get_binary_classification_metrics(
                        complement_true,
                        predicted_values.filter(complement_mask) if predicted_values is not None else None,
                        predicted_probabilities.filter(complement_mask),
                    )
                    boot_res[bi][diff_key] = {
                        metric: group_metrics[metric] - complement_metrics[metric]
                        for metric in complement_metrics
                        if metric in group_metrics
                    }
                    all_keys.update(boot_res[bi][diff_key].keys())

    # Aggregate bootstrap results
    results = {}
    for group_key in boot_res[0]:
        results[group_key] = {}
        for metric in all_keys:
            metric_values = [
                boot_res[bi][group_key][metric]
                for bi in range(bootstrapping)
                if group_key in boot_res[bi] and metric in boot_res[bi][group_key]
            ]
            if metric_values:
                results[group_key][f"mean_{metric}"] = np.mean(metric_values)
                results[group_key][f"std_{metric}"] = np.std(metric_values)

    return results


def evaluate_binary_classification(
    predictions: pl.DataFrame,
    samples_per_subject: int = 4,
    resampling_seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Evaluates binary classification predictions with sample- and subject-weighted metrics.

    Args:
        predictions: DataFrame following the MEDS label schema with additional
            columns for predicted_boolean_value and predicted_boolean_probability.
        samples_per_subject: number of samples per subject for subject-weighted metrics.
        resampling_seed: random seed for subject resampling.

    Returns:
        Dictionary with keys 'samples_equally_weighted' and 'subjects_equally_weighted',
        each mapping to a dict of metric name to scalar value.

    Raises:
        ValueError: if predictions does not conform to the expected schema.
    """
    validate_binary_classification_schema(predictions)

    true_values = predictions[BOOLEAN_VALUE_FIELD]
    predicted_values = predictions[PREDICTED_BOOLEAN_VALUE_FIELD]
    predicted_probabilities = predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD]

    resampled_predictions = _resample(
        predictions,
        sampling_column=SUBJECT_ID_FIELD,
        n_samples=samples_per_subject,
        random_seed=resampling_seed,
    )

    true_values_resampled = resampled_predictions[BOOLEAN_VALUE_FIELD]
    predicted_values_resampled = resampled_predictions[PREDICTED_BOOLEAN_VALUE_FIELD]
    predicted_probabilities_resampled = resampled_predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD]

    if predicted_values.is_null().all():
        predicted_values = None
        predicted_values_resampled = None

    if predicted_probabilities.is_null().all():
        predicted_probabilities = None
        predicted_probabilities_resampled = None

    return {
        "samples_equally_weighted": _get_binary_classification_metrics(
            true_values, predicted_values, predicted_probabilities
        ),
        "subjects_equally_weighted": _get_binary_classification_metrics(
            true_values_resampled, predicted_values_resampled, predicted_probabilities_resampled
        ),
    }


def evaluate_fairness_binary_classification(
    predictions: pl.DataFrame,
    groups: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Evaluates fairness of binary classification predictions across protected groups.

    For each group column in `groups`, computes both standard classification metrics
    and AIF360 fairness metrics per unique group value.

    Args:
        predictions: DataFrame following the MEDS label schema with additional
            columns for predicted_boolean_value and predicted_boolean_probability.
        groups: DataFrame containing protected group membership columns joined on subject_id.

    Returns:
        Nested dictionary of {group_column: {group_value: {metric: value}}}.

    Raises:
        ValueError: if predictions or groups do not conform to the expected schema.
    """
    validate_binary_classification_schema(predictions)
    validate_group_schema(groups)

    groups = (
        groups
        .join(predictions, how="right", on=SUBJECT_ID_FIELD)
        .select(groups.columns)
    )

    true_values = predictions[BOOLEAN_VALUE_FIELD]
    predicted_values = predictions[PREDICTED_BOOLEAN_VALUE_FIELD]
    predicted_probabilities = (
        None if predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD].is_null().all()
        else predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD]
    )

    results = {}
    for group in groups.columns:
        if group in GROUPS_SCHEMA_DICT:
            continue

        results[group] = {}
        for group_value in groups[group].drop_nulls().unique():
            mask = groups[group] == group_value
            group_true = true_values.filter(mask)

            if not _is_valid_subgroup(group_true):
                logger.warning(
                    f"Skipping group '{group}={group_value}': n={len(group_true)}, "
                    f"positive_rate={np.mean(group_true):.2f}"
                )
                continue

            fairness_metrics = _get_fairness_binary_classification_metrics(
                group_true.to_pandas(),
                predicted_probabilities.filter(mask).to_pandas() if predicted_probabilities is not None else None,
                mask.to_pandas().fillna(False),
            )
            classification_metrics = evaluate_binary_classification(predictions.filter(mask))

            results[group][group_value] = {**fairness_metrics, **classification_metrics}

    return results