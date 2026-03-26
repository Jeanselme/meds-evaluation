"""Methods for evaluating different types of tasks and subpopulations on a standard set of metrics.

Most metrics will be directly based on sklearn implementation, and a standard set will be defined for
binary metrics initially. (TODO: add multiclass, multilabel, regression, ... evaluation).

See
    https://scikit-learn.org/stable/api/sklearn.metrics.html
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

Additionally, functionality for evaluating metrics on a per-sample vs per-subject basis is provided to
ensure balanced representation of all subjects in the dataset.

TODO fairness functionality and filtering populations based on complex user-defined criteria.
"""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score
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

# TODO: input processing for different types of tasks
#   detect which set of metrics to obtain based on the task and the contents of the model prediction dataframe


def evaluate_bootstrapped_binary_classification(
    predictions: pl.DataFrame,
    groups: pl.DataFrame,
    bootstrapping=100,
) -> dict[str, dict[str, float | list[ArrayLike]]]:
    """Evaluates a set of model predictions for binary classification tasks with bootstrap confidence.

    Args:
        predictions: a DataFrame following the MEDS label schema and additional columns for
        "predicted_value" and "predicted_probability".
        bootstrapping: number of bootstrap samples to take for confidence intervals.
        # TODO consider adding a parameter for the metric set to evaluate

    Returns:
        A dictionary mapping the metric names to their values.
        The visual (curve-based) metrics will return the raw values needed to create the plot.

    Raises:
        ValueError: if the predictions dataframe does not contain the necessary columns.
    """
    assert bootstrapping > 0, "Bootstrapping must be a positive integer."
    validate_binary_classification_schema(predictions)
    validate_group_schema(groups)

    # Match as their might not be column time
    boot_res, all_keys = {}, set()
    for bi in range(bootstrapping):
        resampled_predictions = _resample(
            predictions,
            random_seed=bi,
        )
        resampled_groups = groups.join(resampled_predictions, how = 'right', left_on=SUBJECT_ID_FIELD, right_on=SUBJECT_ID_FIELD).select(groups.columns)

        true_values_resampled = resampled_predictions[BOOLEAN_VALUE_FIELD]
        predicted_values_resampled = resampled_predictions[PREDICTED_BOOLEAN_VALUE_FIELD]
        predicted_probabilities_resampled = resampled_predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD]

        if predicted_values_resampled.is_null().all():
            predicted_values_resampled = None

        if predicted_probabilities_resampled.is_null().all():
            predicted_probabilities_resampled = None

        boot_res[bi] = {'all': _get_binary_classification_metrics(
            true_values_resampled, predicted_values_resampled, predicted_probabilities_resampled
        )}

        for group in groups.columns:
            if group not in GROUPS_SCHEMA_DICT:
                for group_unique in resampled_groups[group].drop_nulls().unique():
                    boot_res[bi][group + '_' + group_unique] = _get_fairness_binary_classification_metrics(
                        true_values_resampled.to_pandas(), predicted_probabilities_resampled.to_pandas(), (resampled_groups[group] == group_unique).to_pandas().fillna(False)
                    )
                    boot_res[bi][group + '_' + group_unique] = _get_binary_classification_metrics(true_values_resampled.filter(resampled_groups[group] == group_unique) if true_values_resampled is not None else None,
                                                                                                 predicted_values_resampled.filter(resampled_groups[group] == group_unique) if predicted_values_resampled is not None else None,
                                                                                                 predicted_probabilities_resampled.filter(resampled_groups[group] == group_unique))

                    rest = _get_binary_classification_metrics(true_values_resampled.filter(resampled_groups[group] != group_unique) if true_values_resampled is not None else None, 
                                                                predicted_values_resampled.filter(resampled_groups[group] != group_unique) if predicted_values_resampled is not None else None, 
                                                                predicted_probabilities_resampled.filter(resampled_groups[group] != group_unique))
                    boot_res[bi][group + '_' + group_unique + '_difference'] = {metric: boot_res[bi][group + '_' + group_unique][metric] - rest[metric] for metric in rest}
                all_keys.update(boot_res[bi][group + '_' + group_unique].keys())

    # Summarize the results
    results = {}
    for group in boot_res[0]:
        results[group] = {}
        for metric in all_keys:
            metric_values = [boot_res[bi][group][metric] for bi in range(bootstrapping) if metric in boot_res[bi][group]]
            results[group]["mean_" + metric] = np.mean(metric_values)
            results[group]["std_" + metric] = np.std(metric_values)

    return results

def evaluate_binary_classification(
    predictions: pl.DataFrame, samples_per_subject=4, resampling_seed=0
) -> dict[str, dict[str, float | list[ArrayLike]]]:
    """Evaluates a set of model predictions for binary classification tasks.

    Args:
        predictions: a DataFrame following the MEDS label schema and additional columns for
        "predicted_value" and "predicted_probability".
        samples_per_subject: the number of samples to take for each unique subject_id in the dataframe for
        per-subject metrics.
        resampling_seed: random seed for resampling the dataframe.
        # TODO consider adding a parameter for the metric set to evaluate

    Returns:
        A dictionary mapping the metric names to their values.
        The visual (curve-based) metrics will return the raw values needed to create the plot.

    Raises:
        ValueError: if the predictions dataframe does not contain the necessary columns.
    """
    # Verify the dataframe schema to contain required fields for the binary classification metrics
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

    results = {
        "samples_equally_weighted": _get_binary_classification_metrics(
            true_values, predicted_values, predicted_probabilities
        ),
        "subjects_equally_weighted": _get_binary_classification_metrics(
            true_values_resampled, predicted_values_resampled, predicted_probabilities_resampled
        ),
    }

    # TODO write to output file
    return results

def evaluate_fairness_binary_classification(
    predictions: pl.DataFrame, groups: pl.DataFrame
) -> dict[str, dict[str, float | list[ArrayLike]]]:
    """Evaluates the fairness of a set of model predictions for binary classification tasks.

    Args:
        predictions: a DataFrame following the MEDS label schema and additional columns for
        "predicted_value" and "predicted_probability".
        groups: a DataFrame containing the group membership of each sample.
        samples_per_subject: the number of samples to take for each unique subject_id in the dataframe for
        per-subject metrics.
        resampling_seed: random seed for resampling the dataframe.
        # TODO consider adding a parameter for the metric set to evaluate

    Returns:
        A dictionary mapping the metric names to their values.
        The visual (curve-based) metrics will return the raw values needed to create the plot.

    Raises:
        ValueError: if the predictions dataframe does not contain the necessary columns.
    """
    # Verify the dataframe schema to contain required fields for the binary classification metrics
    validate_binary_classification_schema(predictions)
    validate_group_schema(groups)

    # Match as their might not be column time
    groups = groups.join(predictions, how = 'right', left_on=SUBJECT_ID_FIELD, right_on=SUBJECT_ID_FIELD).select(groups.columns)

    true_values = predictions[BOOLEAN_VALUE_FIELD]
    predicted_values = predictions[PREDICTED_BOOLEAN_VALUE_FIELD]
    predicted_probabilities = predictions[PREDICTED_BOOLEAN_PROBABILITY_FIELD]

    if predicted_probabilities.is_null().all():
        predicted_probabilities = None

    results = {}
    for group in groups.columns:
        if group not in GROUPS_SCHEMA_DICT:
            results[group] = {}
            for group_unique in groups[group].drop_nulls().unique():
                results[group][group_unique] = _get_fairness_binary_classification_metrics(
                    true_values.to_pandas(), predicted_values.to_pandas(), (groups[group] == group_unique).to_pandas().fillna(False)
                )
                results[group][group_unique].update(evaluate_binary_classification(predictions.filter(groups[group] == group_unique)))
            
    # TODO write to output file
    return results


def _get_binary_classification_metrics(
    true_values: ArrayLike,
    predicted_values: ArrayLike | None,
    predicted_probabilities: ArrayLike | None,
) -> dict[str, float | list[ArrayLike]]:
    """Calculates a set of binary classification metrics based on the true and predicted values.

    Args:
        true_values: the true binary values
        predicted_values: the predicted binary values
        predicted_probabilities: the predicted probabilities
        TODO consider the list of metrics

    Returns:
        A dictionary mapping the metric names to their values.
        The visual (curve-based) metrics will return the raw values needed to create the plot.
    """
    results = {}

    if predicted_values is not None:
        results["binary_accuracy"] = accuracy_score(true_values, predicted_values)
        results["f1_score"] = f1_score(true_values, predicted_values)

    if predicted_probabilities is not None:
        results["roc_auc_score"] = roc_auc_score(true_values, predicted_probabilities)
        results["average_precision_score"] = average_precision_score(true_values, predicted_probabilities)

        c = calibration_curve(true_values, predicted_probabilities, n_bins=10)
        results["calibration_error"] = np.abs(c[0] - c[1]).mean()
        results["brier_score"] = brier_score_loss(true_values, predicted_probabilities)

    return results


def _get_fairness_binary_classification_metrics(
    true_values: ArrayLike,
    predicted_values: ArrayLike | None,
    group_membership: ArrayLike | None,
) -> dict[str, float | list[ArrayLike]]:
    """Calculates a set of fairness metrics based on the true and predicted values and group membership.

    Args:
        true_values: the true binary values
        predicted_values: the predicted binary values
        group_membership: the group membership of each sample
        TODO consider the list of metrics

    Returns:
        A dictionary mapping the metric names to their values.
    """
    results = {}

    if predicted_values is not None:
        try: results["average_odds_difference"] = average_odds_difference(true_values, predicted_values, prot_attr = group_membership)
        except: pass        
        try: results["conditional_demographic_disparity"] = conditional_demographic_disparity(true_values, predicted_values, prot_attr = group_membership)
        except: pass
        try: results["equal_opportunity_difference"] = equal_opportunity_difference(true_values, predicted_values, prot_attr = group_membership, priv_group = True)
        except: pass
        try: results["statistical_parity_difference"] = statistical_parity_difference(true_values, predicted_values, prot_attr = group_membership, priv_group = True)
        except: pass

    return results