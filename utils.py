"""
Various utility functions
"""
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt


def load_data(classes):
    # csv contains scores of 57860 humans on 6 questions, each graded by 2 humans
    # df = pd.read_csv('data/human_scores.csv', index_col=0)
    dfList = []
    for fname in os.listdir("data"):
        if fname.startswith("prompt"):
            dfList.append(pd.read_csv("data/" + fname, index_col=0))
    df = pd.concat(dfList).reset_index(drop=True)
    # Transform from integer
    int_cls_map = {i: j for i, j in zip(range(len(classes)), classes)}

    for col in list(df.columns)[1:]:  # All columns except for test_response_id
        df.replace({col: int_cls_map}, inplace=True)

    return df


def get_final_grade(response_id, df, key):
    """For a given test_response_id, return associated grade."""
    records = df[
        df.test_response_id == response_id
    ]  # Fetch all responses associated with id
    grade = format_and_grade_response(records, key)
    return grade


def format_and_grade_response(records, key):
    """Format as expected by aggregate_score and calculate final grade."""
    records = records[key].reset_index(drop=True)
    records.index += 1  # Shift by 1 as expected by aggregate_score
    records = records.to_dict()
    grade = aggregate_score(records)
    return grade


def get_predictions(df, model_name, classes):
    """Get predictions for a given model."""

    if model_name.startswith("Pseudo"):
        reqd_acc = float(model_name.split("-")[1])
        # Assume rater1 as ground truth, copy to predictions
        df["predictions"] = df["rater1"]
        # Create a mask of rows to be replaced
        mask = np.random.choice([True, False], size=len(df), p=[1 - reqd_acc, reqd_acc])
        # Replace masked rows with a random prediction
        # Actual accuracy of the model at a question level will be slightly higher
        # than expected because we are including the correct class when sampling a replacment.
        df.loc[mask, "predictions"] = np.random.choice(
            classes, size=len(df.loc[mask, "predictions"])
        )
    else:
        df["predictions"] = df[model_name]
    return df


def calc_final_df(df):
    """
    Calculate the final grade of the given column,
    combining individual grades, for each tester.
    """
    final_df = {}
    final_df["test_response_id"] = dd.from_pandas(
        pd.Series(df.test_response_id.unique()), npartitions=10
    )
    final_df["label"] = final_df["test_response_id"].apply(
        get_final_grade,
        args=(
            df,
            "rater1",
        ),
        meta=("label", str),
    )
    final_df["prediction"] = final_df["test_response_id"].apply(
        get_final_grade,
        args=(
            df,
            "predictions",
        ),
        meta=("prediction", str),
    )

    final_df = dd.concat(list(final_df.values()), axis=1)
    final_df.columns = ["test_response_id", "label", "prediction"]
    return final_df.compute(scheduler="distributed")


def plot_graphs(sample_size_list, metric_list, model_name):
    """Configure, save and display plots."""
    plt.rcParams.update({"font.size": 44})
    fig, axis = plt.subplots(1, 2)
    fig.set_size_inches(30, 15)

    for i, metric in enumerate(["acc", "kappa"]):
        axis[i].locator_params(nbins=10)
        axis[i].set_xlim([0, 82])
        axis[i].set_xlabel("% responses scored by humans", labelpad=15)
        axis[i].set_ylim([0.5, 1])
        axis[i].set_ylabel(metric, labelpad=15)

        actual = metric_list.pop("actual_%s" % metric)
        for metric_name in metric_list.keys():
            if metric_name.endswith(metric):
                axis[i].plot(
                    sample_size_list,
                    metric_list[metric_name],
                    linewidth=5,
                    label=metric_name.split("_")[0] + " sampling",
                )

        axis[i].axline(
            (0, actual),
            slope=0,
            linestyle="--",
            linewidth=5,
            color="red",
            label="actual acc/kappa",
        )  # Plot horizontal line
        handles_labels = axis[i].get_legend_handles_labels()

    fig.tight_layout()
    fig.savefig("images/%s.pdf" % model_name)

    figlegend = plt.figure(figsize=(10, 10))
    figlegend.legend(*handles_labels, loc="center")
    figlegend.savefig("images/legend.pdf")
    # plt.show()


def plot_estim_graph(model_metrics, sample_size_list):
    plt.rcParams.update({"font.size": 37})
    fig, axis = plt.subplots(1, 2)
    fig.set_size_inches(30, 15)
    nmodels = 2

    for i, metric in enumerate(["acc", "kappa"]):
        axis[i].locator_params(nbins=12)
        axis[i].set_xlim([-5, 85])
        axis[i].set_xlabel("% responses scored by humans", labelpad=15)
        axis[i].set_xticks(list(range(0, 90, 10)))
        axis[i].set_ylim([0.4, 1])
        axis[i].set_ylabel(metric, labelpad=15)

        # Plot of only the first nmodels for visual clarity
        reward = model_metrics.pop("reward_%s" % metric)[:nmodels]
        estim = model_metrics.pop("reward_estim_%s" % metric)[:nmodels]

        for j in range(nmodels):
            axis[i].plot(
                sample_size_list,
                reward[j],
                label="Reward Sampling (RS) - k=%s" % j,
                linewidth=5,
            )
            axis[i].plot(
                sample_size_list,
                estim[j],
                label="Estimate after RS - k=%s" % j,
                linewidth=5,
            )
        axis[i].legend(loc="lower right")
    fig.savefig("images/estim_plot.pdf")


def split_dataframe_by_test_id(df):
    """
    Split dataframe into train/test.
    However we split using test_response_ids, making sure
    all 6 responses given by one one human belong in one set,
    and dont' get split.
    """
    df_id = df.test_response_id.unique()
    # Use train data for human-machine matrix only
    train_df_id = np.random.choice(df_id, size=round(0.3 * len(df_id)), replace=False)
    train_df = df.loc[df.test_response_id.isin(train_df_id)]
    test_df = df.loc[~df.test_response_id.isin(train_df_id)].copy()
    return train_df, test_df


def aggregate_score(scores):
    """Aggregates grades of 6 responses and calculates the overall grade.
    Consult the SOPI grading algorithm sheet for more information.
    Parameters
    ----------
    scores : dict
        Dictionary with keys as question number and values as CEFR grade.
        See the example below:
        scores = {
            1: 'High B1',
            2: 'Low B1',
            3: 'Low B1',
            4: 'Low B1',
            5: 'Low B1',
            6: 'Low B1'
        }
    Returns
    -------
    str
        Calculated overall grade.
    """

    PROMPT_LEVEL = {1: "B1", 2: "B1", 3: "B2", 4: "C1", 5: "C1", 6: "B1"}

    POINTS = {
        "Unrated": 0,
        "A2": 2,
        "Low B1": 4,
        "High B1": 6,
        "Low B2": 8,
        "High B2": 10,
        "C1": 12,
    }

    b1_points, b2_points, c1_points = 0, 0, 0
    unrated_counter = 0

    for ques_num, grade in scores.items():
        if grade == "Unrated":
            unrated_counter += 1

        if PROMPT_LEVEL[ques_num] == "B1":
            b1_points += POINTS[grade]
        elif PROMPT_LEVEL[ques_num] == "B2":
            b2_points += POINTS[grade]
        elif PROMPT_LEVEL[ques_num] == "C1":
            c1_points += POINTS[grade]

    if unrated_counter >= 2:
        return "Unratable"

    x_total = b2_points + c1_points
    y_total = b1_points

    if (x_total == 4 and y_total == 4) or (x_total <= 2) or (y_total <= 2):
        return "Unratable"

    if (x_total == 4 and y_total >= 6) or (x_total >= 6 and y_total == 4):
        return "A2"

    if x_total <= 14:
        if x_total <= 8 and y_total <= 8:
            return "A2"

        if x_total >= 12 and y_total >= 16:
            return "High B1"

        return "Low B1"

    if x_total >= 16 and x_total <= 20:
        if y_total <= 10:
            return "Low B1"

        return "High B1"

    if x_total >= 22 and x_total <= 32:
        if x_total <= 26 and y_total <= 14:
            return "High B1"

        if x_total >= 28 and y_total >= 16:
            return "High B2"

        return "Low B2"

    if x_total >= 34 and y_total <= 14:
        return "High B2"
    else:
        return "C1"
