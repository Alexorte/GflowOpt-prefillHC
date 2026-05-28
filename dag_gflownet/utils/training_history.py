import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from dag_gflownet.utils.gflownet_v2 import posterior_estimate
from dag_gflownet.utils.metrics import (
    expected_shd,
    expected_edges,
    threshold_metrics,
    posterior_score_stats,
)
from dag_gflownet.utils.posterior_predictive import (
    posterior_neg_avg_posterior_predictive_loglik,
)


def should_compute_training_history(iteration, start_iteration, history_training, history_training_until):
    if history_training <= 0:
        return False
    if iteration < start_iteration:
        return False
    if iteration > history_training_until:
        return False
    early_checkpoints = {0, 1, 10, 50, 100, 500}
    if iteration in early_checkpoints:
        return True
    return iteration % history_training == 0


def should_compute_slow_history_metrics(iteration, history_training, history_training_slow_multiplier):
    if history_training <= 0:
        return False
    slow_every = history_training * max(1, history_training_slow_multiplier)
    return iteration % slow_every == 0


def compute_training_history_row(*, iteration: int, train_start_time: float, gflownet, params_online, env, key, normalization, graph, data_train, data_test, current_loss, avg_loss, smooth_loss, epsilon, learning_rate, num_samples_history_posterior, include_slow_metrics, equivalent_sample_size=1.0):
    eval_start = time.time()

    posterior, _ = posterior_estimate(
        gflownet,
        params_online,
        env,
        key,
        normalization,
        num_samples=num_samples_history_posterior,
        desc=f"History eval @ {iteration}",
    )

    ground_truth = nx.to_numpy_array(graph, weight=None)
    threshold_results = threshold_metrics(posterior, ground_truth)
    score_results = posterior_score_stats(
        posterior,
        data_test,
        equivalent_sample_size,
    )

    row = {
        "iteration": int(iteration),
        "training_time_seconds": float(time.time() - train_start_time),
        "history_eval_time_seconds": 0.0,

        "expected_shd": float(expected_shd(posterior, ground_truth)),
        "expected_edges": float(expected_edges(posterior)),

        "roc_auc": float(threshold_results["roc_auc"]),
        "prc_auc": float(threshold_results["prc_auc"]),
        "ave_prec": float(threshold_results["ave_prec"]),

        "bic_mean": float(score_results["bic_mean"]),
        "bic_max": float(score_results["bic_max"]),
        "bdeu_mean": float(score_results["bdeu_mean"]),
        "bdeu_max": float(score_results["bdeu_max"]),

        "loss": float(current_loss),
        "avg_loss": float(avg_loss),
        "smooth_loss": float(smooth_loss),
        "epsilon": float(epsilon),
        "learning_rate": float(learning_rate),
    }

    if include_slow_metrics:
        mll_results = posterior_neg_avg_posterior_predictive_loglik(
            posterior,
            data_train,
            data_test,
            equivalent_sample_size=equivalent_sample_size,
        )
        row.update({
            "neg_avg_predictive_log_likelihood": float(
                mll_results["neg_avg_predictive_log_likelihood"]
            ),
            "predictive_log_likelihood_best": float(
                mll_results["predictive_log_likelihood_best"]
            ),
        })

    row["history_eval_time_seconds"] = float(time.time() - eval_start)
    return row


def save_training_history(output_folder, training_history):
    if len(training_history) == 0:
        return

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(training_history)
    df.to_csv(output_folder / "training_history_gflownet.csv", index=False)