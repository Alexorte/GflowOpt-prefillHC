"""Búsqueda local guiada por la política (Local Search GFlowNets, Kim et al., ICLR 2024) para DAG-GFlowNet.

Cada llamada parte de DAGs muestreados por la política y aplica `rounds` rondas de:
  1. destruir: quitar al azar una fracción `destroy_frac` de las aristas (pasos hacia atrás con P_B uniforme);
  2. reconstruir: completar el DAG parcial con la política actual (sin exploración) hasta la acción stop;
  3. aceptar si el score mejora (filtrado determinista).
Las trayectorias de los DAGs aceptados (orden aleatorio hasta el DAG parcial + reconstrucción) se devuelven
como transiciones (adj, acción, delta) para insertarlas en un ReplayBuffer con warmstart_v2.add_transitions.
La pérdida DB modificada es off-policy, así que estas transiciones son datos de entrenamiento válidos.
"""
import numpy as np

from dag_gflownet.utils.jraph_utils import to_graphs_tuple
from dag_gflownet.warmstart_v2 import env_like_mask, backward_trajectories


def destroy(adj, frac, rng):
    edges = np.argwhere(adj > 0)
    out = adj.copy()
    if len(edges) == 0:
        return out
    k = max(1, int(np.ceil(frac * len(edges))))
    drop = rng.choice(len(edges), size=min(k, len(edges)), replace=False)
    out[edges[drop, 0], edges[drop, 1]] = 0
    return out


def reconstruct(gflownet, params, key, start, features, max_parents, normalization):
    """Completa en paralelo los DAGs parciales `start` (B, d, d) con la política (epsilon = 1: sin exploración).

    Devuelve (DAGs finales, transiciones por DAG [[(adj, acción, delta), ...], ...], key)."""
    adj = np.asarray(start).astype(np.int_).copy()
    B, d, _ = adj.shape
    done = np.zeros(B, dtype=bool)
    trans = [[] for _ in range(B)]
    for _ in range(d * d):
        if done.all():
            break
        masks = np.stack([env_like_mask(a, max_parents) for a in adj])
        obs = {'adjacency': adj, 'mask': masks, 'graph': to_graphs_tuple(adj)}
        actions, key, _, _ = gflownet.act(params, key, obs, 1., normalization)
        actions = np.asarray(actions).reshape(-1)
        for b in np.nonzero(~done)[0]:
            a = int(actions[b])
            if a == d * d:
                done[b] = True
                continue
            i, j = divmod(a, d)
            trans[b].append((adj[b].copy(), a, features.delta_add(adj[b], i, j)))
            adj[b, i, j] = 1
    return adj, trans, key


def local_search(gflownet, params, key, dags, features, rng, rounds=4, destroy_frac=0.5,
                 max_parents=None, normalization=1.):
    cur = np.asarray(dags).astype(np.int_).copy()
    cur_scores = np.array([features.score(g) for g in cur])
    init_scores = cur_scores.copy()
    transitions = []
    accepted = 0
    for _ in range(rounds):
        partial = np.stack([destroy(g, destroy_frac, rng) for g in cur])
        new, rec_trans, key = reconstruct(gflownet, params, key, partial, features, max_parents, normalization)
        new_scores = np.array([features.score(g) for g in new])
        acc = new_scores > cur_scores + 1e-9
        for b in np.nonzero(acc)[0]:
            transitions += backward_trajectories(features, partial[b], 1, rng)
            transitions += rec_trans[b]
        cur[acc] = new[acc]
        cur_scores[acc] = new_scores[acc]
        accepted += int(acc.sum())
    stats = {
        'ls_accept_rate': accepted / max(1, rounds * len(cur)),
        'ls_mean_gain': float(np.mean(cur_scores - init_scores)),
        'ls_best_score': float(np.max(cur_scores)),
        'ls_num_transitions': len(transitions),
    }
    return key, transitions, cur, stats
