"""
Entrypoint for running experiments.
"""
import pandas as pd
import dask.dataframe as dd
import dask.distributed
from sklearn.metrics import cohen_kappa_score

from scorer import calc_metrics, aggregate_metrics
from utils import (
    get_predictions,
    calc_final_df,
    plot_graphs,
    split_dataframe_by_test_id,
    load_data,
    plot_estim_graph,
)
from sampling import gen_human_machine_matrix, get_sample, calc_uncertainty_reward


def main():
    classes = ["A2", "Low B1", "High B1", "Low B2", "High B2", "C1"]
    df = load_data(classes)
    models = [
        "BERT-TwoStage",
        "LSTM-Baseline",
        "BERT-Baseline",
        "LSTMAttn-Baseline",
        "LSTMAttn-TwoStage",
        "Pseudo-0.75",
    ]
    client = dask.distributed.Client()
    model_metrics = {
        "actual_acc": [],
        "actual_kappa": [],
        "estim_acc": [],
        "estim_kappa": [],
        "reward_acc": [],
        "reward_kappa": [],
    }

    for model in models:
        metric_list = {}
        sample_size_list = []

        # Get predictions for given model, and calculate correlation matrix
        df = get_predictions(df, model, classes)
        train_df, test_df = split_dataframe_by_test_id(df)
        human_machine_matrix = gen_human_machine_matrix(train_df, classes)
        print(human_machine_matrix)

        # Calculate uncertainty and reward values for each record
        print("Calculating Uncertainty and Reward (this will take a while ...)")
        test_df = calc_uncertainty_reward(test_df, human_machine_matrix, classes)

        # df containing labels and predictions per *human*, aggregating responses to all 6 questions
        test_human_df = calc_final_df(test_df)

        N = len(test_df)
        sample_sizes = [int(i * 0.01 * N) for i in range(2, 82, 2)]  # i.e. human_budget
        for sample_size in sample_sizes:
            for sample_method in [
                "reward",
                "uncertainty",
                "random",
            ]:  # Types of samples
                sample = get_sample(test_df, sample_method, sample_size)
                metrics = calc_metrics(test_df.copy(), test_human_df, sample, classes)
                metric_list = aggregate_metrics(metric_list, sample_method, metrics)
            sample_size_list.append(round(sample_size * 100 / N, 2))
            print(
                "Experiment - [Model Name-%s , Sample Size-%s] - complete"
                % (model, sample_size)
            )

        metric_list["actual_acc"] = (
            (test_human_df.label == test_human_df.prediction).astype(int).mean()
        )
        metric_list["actual_kappa"] = cohen_kappa_score(
            test_human_df.label,
            test_human_df.prediction,
            weights="quadratic",
            labels=classes,
        )

        print(metric_list)
        for key in ["actual_acc", "actual_kappa", "estim_acc", "estim_kappa"]:
            model_metrics[key].append(metric_list[key])
        model_metrics["reward_acc"].append(metric_list["reward_acc"][-1])
        model_metrics["reward_kappa"].append(metric_list["reward_kappa"][-1])

        plot_graphs(sample_size_list, metric_list, model)
        print("Experiments for model '%s' complete" % model)
        print("-----------------")
    plot_estim_graph(model_metrics)
    print("All experiments complete!")


if __name__ == "__main__":
    main()
