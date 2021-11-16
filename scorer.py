"""
Functions associated with scoring and calculating metrics.
"""

from sklearn.metrics import cohen_kappa_score
import dask.dataframe as dd
from estimator import estimate_metrics
from utils import get_final_grade


def calc_metrics(result_df, human_df, sample, classes):
    """Calculate the acc and kappa on the test set after replacing sampled rows."""

    # Replace predictions with ground truth
    result_df.loc[result_df.index.isin(sample.index), "predictions"] = sample["rater1"]

    # Recalculate final label for predictions
    result_human_df = dd.from_pandas(human_df.copy(), npartitions=10)
    result_human_df["prediction"] = result_human_df["test_response_id"].apply(
        get_final_grade,
        args=(
            result_df,
            "predictions",
        ),
        meta=("prediction", str),
    )
    result_human_df = result_human_df.compute(scheduler="distributed")

    metrics = {}
    metrics["acc"] = (
        (result_human_df.label == result_human_df.prediction).astype(int).mean()
    )
    metrics["kappa"] = cohen_kappa_score(
        result_human_df.label,
        result_human_df.prediction,
        weights="quadratic",
        labels=classes,
    )

    metrics = estimate_metrics(result_human_df, result_df, metrics, classes)
    return metrics


def aggregate_metrics(metric_list, sample_method, metrics):
    """Aggregate metrics into list for plotting."""
    for metric_name, val in metrics.items():
        key = "%s_%s" % (sample_method, metric_name)
        if key not in metric_list:
            metric_list[key] = []
        metric_list[key].append(val)
    return metric_list
