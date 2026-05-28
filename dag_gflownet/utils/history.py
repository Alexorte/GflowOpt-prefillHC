import numpy as np
import networkx as nx
from dag_gflownet.utils.metrics import bic_score_of_dag, bdeu_score_of_dag

def summarize_prefill_history(stats, episode_num, graph, data_test):
    dags = np.stack([s["final_adj"] for s in stats], axis=0)
    final_edges = np.array([s["final_num_edges"] for s in stats], dtype=float)

    ground_truth_prefill = nx.to_numpy_array(graph, weight=None)
    shds = np.sum(np.abs(dags - ground_truth_prefill), axis=(1, 2))

    bic_vals = np.array(
        [bic_score_of_dag(dag, data_test) for dag in dags],
        dtype=float
    )
    bdeu_vals = np.array(
        [bdeu_score_of_dag(dag, data_test, equivalent_sample_size=1.0) for dag in dags],
        dtype=float
    )

    return {
        "episode": int(episode_num),
        "mean_final_edges": float(np.mean(final_edges)),
        "mean_shd": float(np.mean(shds)),
        "best_shd": float(np.min(shds)),
        "mean_bic": float(np.mean(bic_vals)),
        "best_bic": float(np.max(bic_vals)),
        "mean_bdeu": float(np.mean(bdeu_vals)),
        "best_bdeu": float(np.max(bdeu_vals)),
    }