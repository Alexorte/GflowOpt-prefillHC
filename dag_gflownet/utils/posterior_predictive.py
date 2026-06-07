import numpy as np
from scipy.special import gammaln
from pgmpy.estimators import BDeuScore


# -----------------------------------------------------------------------------
# Métrica predictiva posterior auxiliar
# -----------------------------------------------------------------------------
# Este módulo calcula una verosimilitud predictiva posterior aproximada para un
# conjunto de DAGs muestreados por la GFlowNet.
#
# La idea general es:
#   1. Usar data_train para obtener los conteos locales de cada variable dado
#      su conjunto de padres en un DAG concreto.
#   2. Usar data_test para obtener los conteos locales equivalentes sobre datos
#      no vistos.
#   3. Calcular, para cada familia local Xi | Pa(Xi), la probabilidad predictiva
#      de los conteos de test bajo una posterior Dirichlet actualizada con los
#      conteos de train.
#   4. Sumar las contribuciones locales de todos los nodos del DAG.
#   5. Promediar la métrica sobre todos los DAGs del posterior aproximado.
#
# Se reporta finalmente el valor negativo medio para que valores menores indiquen
# mejor comportamiento predictivo, siguiendo la convención habitual de pérdidas.
# -----------------------------------------------------------------------------


def parent_indices_from_adj(adj_matrix, target):
    """Devuelve los índices de los padres de un nodo en una matriz de adyacencia.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Matriz de adyacencia del DAG, con forma ``[num_variables, num_variables]``.
        Se asume la convención ``adj_matrix[i, j] != 0`` si existe la arista
        dirigida ``Xi -> Xj``.
    target : int
        Índice del nodo destino ``X_target`` para el que se quieren recuperar
        sus padres.

    Returns
    -------
    list[int]
        Lista con los índices ``i`` tales que existe la arista ``Xi -> X_target``.
    """
    return [i for i in range(adj_matrix.shape[0]) if adj_matrix[i, target] != 0]


def _local_bdeu_posterior_predictive_loglik(
    train_counts,
    test_counts,
    equivalent_sample_size=1.0,
):
    """Calcula la contribución predictiva local de una familia ``Xi | Pa(Xi)``.

    Esta función aplica la forma Dirichlet-multinomial usada en scores de tipo
    BDeu. Los conteos de entrenamiento actualizan la posterior Dirichlet y los
    conteos de test se evalúan bajo la distribución predictiva resultante.

    Parameters
    ----------
    train_counts : array-like
        Matriz de conteos locales obtenida en entrenamiento, con forma
        ``[r_i, q_i]``, donde ``r_i`` es el número de estados de la variable
        ``Xi`` y ``q_i`` el número de configuraciones posibles de sus padres.
    test_counts : array-like
        Matriz de conteos locales equivalente sobre el conjunto de test, con la
        misma forma ``[r_i, q_i]``.
    equivalent_sample_size : float, default=1.0
        Parámetro de tamaño muestral equivalente del prior BDeu. Controla la
        fuerza del suavizado Dirichlet.

    Returns
    -------
    float
        Log-verosimilitud predictiva posterior local para la familia evaluada.
    """
    train_counts = np.asarray(train_counts, dtype=float)
    test_counts = np.asarray(test_counts, dtype=float)

    # r_i: número de estados de la variable objetivo.
    # q_i: número de configuraciones de los padres.
    r_i, q_i = train_counts.shape

    # Hiperparámetros BDeu repartidos uniformemente.
    # alpha_ij es la masa total asignada a cada configuración de padres.
    # alpha_ijk es la masa asignada a cada estado concreto de Xi para una
    # configuración concreta de padres.
    alpha_ij = equivalent_sample_size / q_i
    alpha_ijk = equivalent_sample_size / (r_i * q_i)

    # Conteos agregados por configuración de padres.
    train_sum = np.sum(train_counts, axis=0)
    test_sum = np.sum(test_counts, axis=0)

    # Cálculo en log-espacio usando gammaln para evitar problemas numéricos con
    # productos de funciones Gamma.
    score = np.sum(
        gammaln(alpha_ij + train_sum)
        - gammaln(alpha_ij + train_sum + test_sum)
        + np.sum(
            gammaln(alpha_ijk + train_counts + test_counts)
            - gammaln(alpha_ijk + train_counts),
            axis=0,
        )
    )

    return float(score)


def posterior_predictive_loglik_of_dag(
    adj_matrix,
    data_train,
    data_test,
    equivalent_sample_size=1.0,
):
    """Calcula la log-verosimilitud predictiva posterior de un DAG concreto.

    Para cada nodo ``Xi`` del DAG, se identifica su conjunto de padres
    ``Pa(Xi)`` a partir de la matriz de adyacencia. Después, se extraen los
    conteos locales ``Xi | Pa(Xi)`` tanto en train como en test usando
    ``pgmpy``. La puntuación total del DAG se obtiene sumando las contribuciones
    locales de todos los nodos.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Matriz de adyacencia del DAG evaluado, con forma
        ``[num_variables, num_variables]``.
    data_train : pandas.DataFrame
        Datos de entrenamiento usados para construir la posterior local.
    data_test : pandas.DataFrame
        Datos de test usados para evaluar la capacidad predictiva del DAG.
    equivalent_sample_size : float, default=1.0
        Parámetro de tamaño muestral equivalente usado por BDeu.

    Returns
    -------
    float
        Log-verosimilitud predictiva posterior total del DAG.
    """
    # Copiamos los DataFrames para no modificar los datos originales fuera de la
    # función.
    train_cat = data_train.copy()
    test_cat = data_test.copy()

    # Convertimos las variables discretas a tipo category. La columna 'INT', si
    # existe, se omite porque actúa como columna auxiliar de intervención y no
    # representa una variable ordinaria del DAG.
    for col in train_cat.columns:
        if col != "INT":
            train_cat[col] = train_cat[col].astype("category")
    for col in test_cat.columns:
        if col != "INT":
            test_cat[col] = test_cat[col].astype("category")

    # BDeuScore se utiliza aquí para reutilizar su método state_counts, no para
    # devolver directamente el score BDeu completo.
    scorer_train = BDeuScore(train_cat, equivalent_sample_size=equivalent_sample_size)
    scorer_test = BDeuScore(test_cat, equivalent_sample_size=equivalent_sample_size)

    node_names = [c for c in train_cat.columns if c != "INT"]
    total_score = 0.0
    num_vars = adj_matrix.shape[0]

    for target in range(num_vars):
        variable = node_names[target]

        # Recuperar los padres del nodo target en formato índice y traducirlos a
        # nombres de columna, que es el formato esperado por pgmpy.
        parents_idx = parent_indices_from_adj(adj_matrix, target)
        parents = [node_names[i] for i in parents_idx]

        # Conteos locales de Xi condicionado a sus padres en train y test.
        train_state_counts = scorer_train.state_counts(variable, parents)
        test_state_counts = scorer_test.state_counts(variable, parents)

        train_counts = np.asarray(train_state_counts, dtype=float)
        test_counts = np.asarray(test_state_counts, dtype=float)

        # Si el nodo no tiene padres, pgmpy puede devolver un vector 1D. Para que
        # el cálculo local sea uniforme, se fuerza la forma [r_i, 1].
        if train_counts.ndim == 1:
            train_counts = train_counts[:, None]
        if test_counts.ndim == 1:
            test_counts = test_counts[:, None]

        total_score += _local_bdeu_posterior_predictive_loglik(
            train_counts,
            test_counts,
            equivalent_sample_size=equivalent_sample_size,
        )

    return float(total_score)


def posterior_neg_avg_posterior_predictive_loglik(
    posterior,
    data_train,
    data_test,
    equivalent_sample_size=1.0,
):
    """Evalúa la métrica predictiva media sobre un conjunto de DAGs muestreados.

    Parameters
    ----------
    posterior : np.ndarray or list[np.ndarray]
        Conjunto de DAGs muestreados por la GFlowNet. Cada DAG se representa
        mediante una matriz de adyacencia.
    data_train : pandas.DataFrame
        Datos de entrenamiento usados para actualizar los conteos posteriores.
    data_test : pandas.DataFrame
        Datos de test usados para evaluar la predicción.
    equivalent_sample_size : float, default=1.0
        Parámetro de tamaño muestral equivalente usado por BDeu.

    Returns
    -------
    dict
        Diccionario con dos métricas:

        - ``neg_avg_predictive_log_likelihood``: negativo de la media de las
          log-verosimilitudes predictivas. Al estar en negativo, menor es mejor.
        - ``predictive_log_likelihood_best``: mejor log-verosimilitud predictiva
          entre los DAGs evaluados. En este caso, mayor es mejor.
    """
    log_vals = [
        posterior_predictive_loglik_of_dag(
            dag,
            data_train,
            data_test,
            equivalent_sample_size=equivalent_sample_size,
        )
        for dag in posterior
    ]

    log_vals = np.asarray(log_vals, dtype=float)

    return {
        "neg_avg_predictive_log_likelihood": float(-np.mean(log_vals)),
        "predictive_log_likelihood_best": float(np.max(log_vals)),
    }