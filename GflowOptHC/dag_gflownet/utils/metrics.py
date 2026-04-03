"""
The code is adapted from:
https://github.com/larslorch/dibs/blob/master/dibs/metrics.py

MIT License

Copyright (c) 2021 Lars Lorch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np

from sklearn import metrics
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore, BDeuScore

def adjacency_to_edges(adj_matrix, node_names):
    edges = []
    num_vars = adj_matrix.shape[0]
    for i in range(num_vars):
        for j in range(num_vars):
            if adj_matrix[i, j] != 0:
                edges.append((node_names[i], node_names[j]))
    return edges

# Con pgmpy --> Son más eficientes

def bic_score_of_dag(adj_matrix, data):
    node_names = list(data.columns)
    edges = adjacency_to_edges(adj_matrix, node_names)
    model = BayesianNetwork(edges)
    model.add_nodes_from(node_names)
    scorer = BicScore(data)
    return float(scorer.score(model))


def bdeu_score_of_dag(adj_matrix, data, equivalent_sample_size=1.0):
    node_names = list(data.columns)
    edges = adjacency_to_edges(adj_matrix, node_names)
    model = BayesianNetwork(edges)
    model.add_nodes_from(node_names)
    scorer = BDeuScore(data, equivalent_sample_size=equivalent_sample_size)
    return float(scorer.score(model))

def score_stats_of_dags(dags, data, score='bic', equivalent_sample_size=1.0):
    scores = []

    for dag in dags:
        if score.lower() == 'bic':
            value = bic_score_of_dag(dag, data)
        elif score.lower() == 'bdeu':
            value = bdeu_score_of_dag(
                dag,
                data,
                equivalent_sample_size=equivalent_sample_size
            )
        else:
            raise ValueError(f"Score desconocido: {score}")

        scores.append(value)

    scores = np.asarray(scores, dtype=float)

    return {
        f'{score.lower()}_mean': float(np.mean(scores)),
        f'{score.lower()}_max': float(np.max(scores)),
    }

def posterior_score_stats(posterior, data, equivalent_sample_size=1.0):
    results = {}
    results.update(score_stats_of_dags(
        posterior,
        data,
        score='bic',
        equivalent_sample_size=equivalent_sample_size
    ))
    results.update(score_stats_of_dags(
        posterior,
        data,
        score='bdeu',
        equivalent_sample_size=equivalent_sample_size
    ))
    return results


def expected_shd(posterior, ground_truth):
    """Compute the Expected Structural Hamming Distance.

    This function computes the Expected SHD between a posterior approximation
    given as a collection of samples from the posterior, and the ground-truth
    graph used in the original data generation process.

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.

    Returns
    -------
    e_shd : float
        The Expected SHD.
    """
    # Compute the pairwise differences
    diff = np.abs(posterior - np.expand_dims(ground_truth, axis=0))
    diff = diff + diff.transpose((0, 2, 1))

    # Ignore double edges
    diff = np.minimum(diff, 1)
    shds = np.sum(diff, axis=(1, 2)) / 2

    return np.mean(shds)


def expected_edges(posterior):
    """Compute the expected number of edges.

    This function computes the expected number of edges in graphs sampled from
    the posterior approximation.

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    Returns
    -------
    e_edges : float
        The expected number of edges.
    """
    num_edges = np.sum(posterior, axis=(1, 2))
    return np.mean(num_edges)


def threshold_metrics(posterior, ground_truth):
    """Compute threshold metrics (e.g. AUROC, Precision, Recall, etc...).

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.

    Returns
    -------
    metrics : dict
        The threshold metrics.
    """
    # Expected marginal edge features
    p_edge = np.mean(posterior, axis=0)
    p_edge_flat = p_edge.reshape(-1)
    
    gt_flat = ground_truth.reshape(-1)

    # Threshold metrics 
    fpr, tpr, _ = metrics.roc_curve(gt_flat, p_edge_flat)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(gt_flat, p_edge_flat)
    prc_auc = metrics.auc(recall, precision)
    ave_prec = metrics.average_precision_score(gt_flat, p_edge_flat)
    
    return {
        # 'fpr': fpr,
        # 'tpr': tpr,
        'roc_auc': roc_auc,
        # 'precision': precision,
        # 'recall': recall,
        'prc_auc': prc_auc,
        'ave_prec': ave_prec,
    }
