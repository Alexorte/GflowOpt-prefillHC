import json
import numpy as np
import pandas as pd
import networkx as nx

from dag_gflownet.utils.metrics import (
    bic_score_of_dag,
    bdeu_score_of_dag,
)
from dag_gflownet.utils.posterior_predictive import posterior_predictive_loglik_of_dag

def compute_single_hc_structure_metrics(dag, graph, data_train, data_test, equivalent_sample_size=1.0):
    ground_truth = nx.to_numpy_array(graph, weight=None)

    log_mll = posterior_predictive_loglik_of_dag(
        dag,
        data_train,
        data_test,
        equivalent_sample_size=equivalent_sample_size
    )

    return {
        "shd": float(np.sum(np.abs(dag - ground_truth))),
        "num_edges": float(np.sum(dag)),
        "bic": float(bic_score_of_dag(dag, data_test)),
        "bdeu": float(bdeu_score_of_dag(dag, data_test, equivalent_sample_size=equivalent_sample_size)),
        "posterior_predictive_loglik": float(log_mll),
        "neg_posterior_predictive_loglik": float(-log_mll),
    }


def summarize_hc_final_results(diverse_top_results, graph, data_train, data_test, equivalent_sample_size=1.0):
    hc_final_metrics = []

    for i, result in enumerate(diverse_top_results):
        dag = np.asarray(result["optimized_matrix"], dtype=float)
        metrics_i = compute_single_hc_structure_metrics(
            dag,
            graph,
            data_train,
            data_test,
            equivalent_sample_size=equivalent_sample_size
        )
        metrics_i["rank"] = i + 1
        hc_final_metrics.append(metrics_i)

    hc_results_summary = {}

    if len(hc_final_metrics) > 0:
        hc_results_summary = {
            "HC_final_shd": hc_final_metrics[0]["shd"],
            "HC_final_num_edges": hc_final_metrics[0]["num_edges"],
            "HC_final_bic": hc_final_metrics[0]["bic"],
            "HC_final_bdeu": hc_final_metrics[0]["bdeu"],
            "HC_final_neg_posterior_predictive_loglik": hc_final_metrics[0]["neg_posterior_predictive_loglik"],

            "HC_top_mean_shd": float(np.mean([m["shd"] for m in hc_final_metrics])),
            "HC_top_best_shd": float(np.min([m["shd"] for m in hc_final_metrics])),

            "HC_top_mean_bic": float(np.mean([m["bic"] for m in hc_final_metrics])),
            "HC_top_best_bic": float(np.max([m["bic"] for m in hc_final_metrics])),

            "HC_top_mean_bdeu": float(np.mean([m["bdeu"] for m in hc_final_metrics])),
            "HC_top_best_bdeu": float(np.max([m["bdeu"] for m in hc_final_metrics])),

            "HC_top_mean_neg_posterior_predictive_loglik": float(
                np.mean([m["neg_posterior_predictive_loglik"] for m in hc_final_metrics])
            ),
            "HC_top_best_neg_posterior_predictive_loglik": float(
                np.min([m["neg_posterior_predictive_loglik"] for m in hc_final_metrics])
            ),
        }

    return hc_final_metrics, hc_results_summary


def save_hc_final_metrics(output_dir, hc_final_metrics, hc_results_summary):
    with open(output_dir / "hc_final_metrics.json", "w") as f:
        json.dump(hc_results_summary, f, indent=2)

    pd.DataFrame(hc_final_metrics).to_csv(
        output_dir / "hc_final_top_structures_metrics.csv",
        index=False
    )