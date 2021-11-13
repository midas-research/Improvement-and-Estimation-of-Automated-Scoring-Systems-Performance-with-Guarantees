"""
Functions associated with estimating metrics
"""

from sklearn.metrics import confusion_matrix, cohen_kappa_score
import math
import numpy as np
import pandas as pd

from utils import get_final_grade


def acc_lower_bound(df):
    """Calculate 95% CI lower bound for accuracy."""
    results = (df["label"] == df["prediction"]).astype(int)
    mean = results.mean()
    std = results.std()
    n = len(df)
    std = std / math.sqrt(n)
    return mean - 1.96 * std


def kappa_lower_bound(df, classes):
    """Calculate 95% CI lower bound for quadratic weighted kappa."""
    # Source from sklearn.metrics.cohen_kappa_score rewritten to generate confidence interval

    confusion = confusion_matrix(
        df["label"], df["prediction"], labels=classes, normalize="all"
    )

    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.zeros([n_classes, n_classes], dtype=int)
    w_mat += np.arange(n_classes)
    w_mat = (w_mat - w_mat.T) ** 2

    p_o = 1 - np.sum(w_mat * confusion)
    p_e = 1 - np.sum(w_mat * expected)
    kappa = 1 - ((1 - p_o) / (1 - p_e))
    std_err = math.sqrt((p_o * (1 - p_o)) / (len(df) * (1 - p_e) ** 2))
    return kappa - 1.96 * std_err


def aggregate_confidence(response_id, df):
    records = df[
        df.test_response_id == response_id
    ]  # Fetch all responses associated with id
    return (1 - records["uncertainty"].sum()) ** 2


def estimate_metrics(human_df, record_df, metrics, classes):
    """Estimate accuracy and kappa with a secondary sample."""
    sample_size = 200

    human_df["confidence"] = human_df["test_response_id"].apply(
        aggregate_confidence, args=(record_df,)
    )
    human_df["confidence"] /= human_df["confidence"].sum()
    sample = np.random.choice(
        human_df.index, size=sample_size, replace=False, p=human_df["confidence"]
    )
    sample = human_df[human_df.index.isin(sample)]

    metrics["estim_acc"] = acc_lower_bound(sample)
    metrics["estim_kappa"] = kappa_lower_bound(sample, classes)

    return metrics


def estimate_reward_metrics(sample, result_df, metrics, classes):
    """Estimate accuracy and kappa using existing sample."""

    # Replacing ground truth with predictions since ground truth is not known while estimating
    result_df["rater1"] = result_df["predictions"]
    # Adding back ground truth for sampled records
    result_df.loc[result_df.index.isin(sample.index), "rater1"] = sample["rater1"]

    # Aggregate df for sample
    sample_human_df = {}
    sample_human_df["test_response_id"] = pd.Series(sample.test_response_id.unique())
    sample_human_df["label"] = sample_human_df["test_response_id"].apply(
        get_final_grade, args=(result_df, "rater1")
    )
    sample_human_df["prediction"] = sample_human_df["test_response_id"].apply(
        get_final_grade, args=(result_df, "predictions")
    )
    sample_human_df = pd.concat(list(sample_human_df.values()), axis=1)
    sample_human_df.columns = ["test_response_id", "label", "prediction"]

    metrics["estim_acc"] = acc_lower_bound(sample_human_df)
    metrics["estim_kappa"] = kappa_lower_bound(sample_human_df, classes)

    return metrics
