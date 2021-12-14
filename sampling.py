"""
Functions associated with sampling records according to various methods.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import dask.dataframe as dd

from utils import format_and_grade_response


def gen_human_machine_matrix(df, classes):
    """Generate matrix containing human-machine correlation."""
    # Assume rater1 as ground truth
    human_machine_matrix = {}
    for cls in classes:
        machine_cls_preds = df[df.predictions == cls]
        # Find human-machine correlation
        try:
            human_machine_matrix["m/" + cls] = [
                len(machine_cls_preds[machine_cls_preds.rater1 == c])
                / len(machine_cls_preds)
                for c in classes
            ]
        except ZeroDivisionError:
            human_machine_matrix["m/" + cls] = [0] * len(classes)
    human_machine_matrix = pd.DataFrame.from_dict(
        human_machine_matrix, orient="index", columns=["h/" + c for c in classes]
    )
    plot_human_machine_matrix(df, classes)
    return human_machine_matrix


def plot_human_machine_matrix(df, classes):
    """Plot the human-machine matrix for visualiaztion."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import matplotlib

    # Use TrueType fonts
    matplotlib.rcParams['pdf.fonttype'] = 42

    conf_mat = confusion_matrix(
        df.predictions, df.rater1, labels=classes, normalize="true"
    )
    plt.rcParams.update({"font.size": 26})
    plt.rcParams["figure.figsize"] = (16, 16)
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, colorbar=False)
    disp.ax_.xaxis.set_label_position("top")
    disp.ax_.xaxis.tick_top()
    plt.xlabel("Human Label")
    plt.ylabel("Machine Predictions")
    plt.savefig("images/human_machine_matrix.pdf")
    return


def calc_cross_ent(human_machine_matrix):
    """Calculate the cross entropy loss associated with each class the machine predicts."""
    hm_scores = torch.tensor(human_machine_matrix.to_numpy(), dtype=torch.float32)
    target = torch.tensor(range(len(human_machine_matrix)), dtype=torch.int64)

    # Scaling for stronger values
    hm_scores = hm_scores / torch.mean(hm_scores)
    cross_ents = F.cross_entropy(hm_scores, target, reduction="none")
    return cross_ents


def calc_expected_reward(idx, df, human_machine_matrix, classes):
    """
    Calculate the expected reward from sampling one particular record.
    The weighted sum of the probability of a change in class by the reward gotten by said change.
    """
    response_id = df[df.index == idx]["test_response_id"].iloc[0]
    records = df[df.test_response_id == response_id]  # Fetch all responses by tester
    pred_label = format_and_grade_response(records, "predictions")
    expected_reward = sum(
        [
            human_machine_matrix["h/%s" % c]["m/%s" % pred_label]
            * calc_reward(pred_label, records.copy(), idx, c, classes)
            for c in classes
        ]
    )
    return expected_reward


def calc_reward(pred_label, records, idx, new_cls, classes):
    """Calculate the reward associated with changing the predicted class of one record."""
    cls_int_map = {i: j for i, j in zip(classes, range(len(classes)))}
    records[records.index == idx] = new_cls  # Replace by new class
    new_label = format_and_grade_response(
        records, "predictions"
    )  # New final grade after replacement

    pred_val = cls_int_map[pred_label]
    new_val = cls_int_map[new_label]
    reward = abs(pred_val - new_val)
    return reward


def get_sample(df, sample_method, sample_size):
    """Sample from dataset based on provided sample_method."""
    if sample_method == "random":
        sample = np.random.choice(df.index, size=sample_size, replace=False)
    elif sample_method == "uncertainty":
        sample = np.random.choice(
            df.index, size=sample_size, replace=False, p=df["uncertainty"]
        )
    elif sample_method == "reward":
        sample = np.random.choice(
            df.index, size=sample_size, replace=False, p=df["scaled_reward"]
        )
    sample = df[df.index.isin(sample)]
    return sample


def calc_uncertainty_reward(df, human_machine_matrix, classes):
    """
    Calculate the uncertainty and reward associated with each record, to be used
    while sampling.
    """
    # Smoothing to prevent zero probability in reward
    delta = 0.001

    # Uncertainty
    cross_entropy_list = calc_cross_ent(human_machine_matrix)
    # Assign uncertainty based on the prediction made
    df["uncertainty"] = 0
    for cls, cross_ent in zip(classes, cross_entropy_list):
        df.loc[df["predictions"] == cls, "uncertainty"] = float(cross_ent)
    df["uncertainty"] = df["uncertainty"] / df["uncertainty"].sum()  # Scaling

    # Reward
    ddf = dd.from_pandas(df, npartitions=10)
    df["reward"] = (
        ddf.index.to_series()
        .apply(
            calc_expected_reward,
            args=(df, human_machine_matrix, classes),
            meta=("reward", float),
        )
        .compute(scheduler="distributed")
    )
    df["reward"] += delta
    df["scaled_reward"] = df["reward"] / df["reward"].sum()
    return df
