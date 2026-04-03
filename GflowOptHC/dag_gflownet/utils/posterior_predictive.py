import numpy as np
from scipy.special import gammaln
from pgmpy.estimators import BDeuScore

# Se calcula la verosimilitud predictiva sobre un conjunto de test para evaluar la capacidad de 
# generalización de los grafos muestreados del posterior.

def parent_indices_from_adj(adj_matrix, target):
    return [i for i in range(adj_matrix.shape[0]) if adj_matrix[i, target] != 0]

def _local_bdeu_posterior_predictive_loglik(train_counts, test_counts, equivalent_sample_size=1.0):
    train_counts = np.asarray(train_counts, dtype=float)
    test_counts = np.asarray(test_counts, dtype=float)

    r_i, q_i = train_counts.shape

    alpha_ij = equivalent_sample_size / q_i
    alpha_ijk = equivalent_sample_size / (r_i * q_i)

    train_sum = np.sum(train_counts, axis=0)
    test_sum = np.sum(test_counts, axis=0)

    score = np.sum(
        gammaln(alpha_ij + train_sum)
        - gammaln(alpha_ij + train_sum + test_sum)
        + np.sum(
            gammaln(alpha_ijk + train_counts + test_counts)
            - gammaln(alpha_ijk + train_counts),
            axis=0
        )
    )
    return float(score)

def posterior_predictive_loglik_of_dag(adj_matrix, data_train, data_test, equivalent_sample_size=1.0):
    train_cat = data_train.copy()
    test_cat = data_test.copy()

    for col in train_cat.columns:
        if col != 'INT':
            train_cat[col] = train_cat[col].astype('category')
    for col in test_cat.columns:
        if col != 'INT':
            test_cat[col] = test_cat[col].astype('category')

    scorer_train = BDeuScore(train_cat, equivalent_sample_size=equivalent_sample_size)
    scorer_test = BDeuScore(test_cat, equivalent_sample_size=equivalent_sample_size)

    node_names = [c for c in train_cat.columns if c != 'INT']
    total_score = 0.0
    num_vars = adj_matrix.shape[0]

    for target in range(num_vars):
        variable = node_names[target]
        parents_idx = parent_indices_from_adj(adj_matrix, target)
        parents = [node_names[i] for i in parents_idx]

        train_state_counts = scorer_train.state_counts(variable, parents)
        test_state_counts = scorer_test.state_counts(variable, parents)

        train_counts = np.asarray(train_state_counts, dtype=float)
        test_counts = np.asarray(test_state_counts, dtype=float)

        # Caso nodo raíz: aseguramos forma [r_i, 1]
        if train_counts.ndim == 1:
            train_counts = train_counts[:, None]
        if test_counts.ndim == 1:
            test_counts = test_counts[:, None]

        total_score += _local_bdeu_posterior_predictive_loglik(
            train_counts,
            test_counts,
            equivalent_sample_size=equivalent_sample_size
        )

    return float(total_score)

def posterior_neg_avg_posterior_predictive_loglik(posterior, data_train, data_test, equivalent_sample_size=1.0):
    log_vals = [
        posterior_predictive_loglik_of_dag(
            dag,
            data_train,
            data_test,
            equivalent_sample_size=equivalent_sample_size
        )
        for dag in posterior
    ]

    log_vals = np.asarray(log_vals, dtype=float)

    return {
        "neg_avg_predictive_log_likelihood": float(-np.mean(log_vals)),
        "predictive_log_likelihood_best": float(np.max(log_vals)),
    }