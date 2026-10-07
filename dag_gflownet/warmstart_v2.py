"""Utilidades de warm-start persistente y de política informada para DAG-GFlowNet (ampliación del TFG).

Contenido:
  * DeltaFeatures: matriz de delta-scores D[i, j] = s_j(Pa_j U {i}) - s_j(Pa_j) con caché, usando el
    mismo scorer que el entorno (BIC/BDe del código base). Es la entrada extra de la política informada.
  * env_like_mask: máscara de acciones idéntica a la de GFlowNetDAGEnv (aciclicidad + max_parents).
  * backward_trajectories (W2): K órdenes aleatorios de las aristas de un DAG experto.
  * boundary_transitions: G* -> G* + arista (delta normalmente < 0). Enseñan a PARAR en G*.
  * add_transitions: inserta transiciones (adj, acción, delta) en un ReplayBuffer del código base
    con exactamente el mismo formato que las que genera el entorno.

Validado en el prototipo NumPy (ver prototipo_numpy/). Este port a JAX NO se ha podido ejecutar
en el entorno donde se escribió (sin acceso a PyPI): compruébalo primero con pocas iteraciones.
"""
import numpy as np


class DeltaFeatures:
    def __init__(self, scorer, num_variables):
        self.scorer = scorer
        self.d = num_variables
        self._local = {}
        self._vec = {}

    def local(self, target, parents):
        key = (target, tuple(sorted(int(p) for p in parents)))
        v = self._local.get(key)
        if v is None:
            _, after = self.scorer.get_local_scores(target, list(key[1]), indices_after=None)
            v = float(after.score + after.prior)
            self._local[key] = v
        return v

    def vector(self, target, parents):
        key = (target, tuple(sorted(int(p) for p in parents)))
        v = self._vec.get(key)
        if v is None:
            base = self.local(target, key[1])
            v = np.zeros(self.d, dtype=np.float32)
            for i in range(self.d):
                if i != target and i not in key[1]:
                    v[i] = self.local(target, key[1] + (i,)) - base
            self._vec[key] = v
        return v

    def matrix(self, adjacency):
        """adjacency (B, d, d) -> D (B, d, d) float32."""
        adjacency = np.asarray(adjacency)
        B, d, _ = adjacency.shape
        D = np.zeros((B, d, d), dtype=np.float32)
        for b in range(B):
            for j in range(d):
                D[b, :, j] = self.vector(j, np.nonzero(adjacency[b, :, j])[0])
        return D

    def delta_add(self, adj, i, j):
        pas = np.nonzero(adj[:, j])[0]
        return self.local(j, list(pas) + [i]) - self.local(j, pas)

    def score(self, adj):
        return sum(self.local(j, np.nonzero(adj[:, j])[0]) for j in range(adj.shape[0]))


def _reach(adj):
    d = adj.shape[-1]
    R = (adj > 0) | np.eye(d, dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(d, 2)))) + 1):
        R = (R.astype(np.float32) @ R.astype(np.float32)) > 0
    return R


def env_like_mask(adj, max_parents=None):
    """Igual que GFlowNetDAGEnv: mask = 1 - (adj + closure_T), y * (num_parents < max_parents)."""
    adj = np.asarray(adj).astype(np.int_)
    d = adj.shape[0]
    closure_T = _reach(adj).T
    mask = 1 - (adj + closure_T.astype(np.int_))
    mask = np.clip(mask, 0, 1)
    mp = max_parents or d
    num_parents = adj.sum(axis=0, keepdims=True)  # (1, d) por nodo destino
    mask = mask * (num_parents < mp)
    return mask.astype(np.int_)


def backward_trajectories(feats, adj, K, rng):
    d = adj.shape[0]
    edges = np.argwhere(adj > 0)
    out = []
    for _ in range(K):
        cur = np.zeros((d, d), dtype=np.int_)
        for e in rng.permutation(len(edges)):
            i, j = edges[e]
            out.append((cur.copy(), int(i * d + j), feats.delta_add(cur, i, j)))
            cur[i, j] = 1
    return out


def boundary_transitions(feats, adj, M, rng, max_parents=None):
    d = adj.shape[0]
    cand = np.nonzero(env_like_mask(adj, max_parents).reshape(-1))[0]
    if len(cand) == 0:
        return []
    pick = rng.choice(cand, size=min(M, len(cand)), replace=False)
    return [(np.asarray(adj, dtype=np.int_).copy(), int(a), feats.delta_add(adj, a // d, a % d)) for a in pick]


def add_transitions(buffer, transitions, max_parents=None, chunk=4096):
    """Inserta [(adj, action, delta), ...] en un dag_gflownet.utils.replay_buffer.ReplayBuffer."""
    if not transitions:
        return
    d = transitions[0][0].shape[0]
    for start in range(0, len(transitions), chunk):
        part = transitions[start:start + chunk]
        n = len(part)
        A = np.stack([t[0] for t in part]).astype(np.int_)
        acts = np.array([t[1] for t in part], dtype=np.int_)
        deltas = np.array([t[2] for t in part], dtype=np.float64)
        A2 = A.copy().reshape(n, -1)
        A2[np.arange(n), acts] = 1
        A2 = A2.reshape(n, d, d)
        obs = {
            'adjacency': A,
            'mask': np.stack([env_like_mask(a, max_parents) for a in A]),
            'num_edges': A.reshape(n, -1).sum(1),
            'score': np.zeros(n),
        }
        nxt = {
            'adjacency': A2,
            'mask': np.stack([env_like_mask(a, max_parents) for a in A2]),
            'num_edges': A2.reshape(n, -1).sum(1),
            'score': np.zeros(n),
        }
        buffer.add(obs, acts.reshape(-1, 1), np.zeros((n, 1), dtype=np.int_), nxt, deltas,
                   np.zeros(n, dtype=bool))


def hill_climb(feats, adj0, max_parents=None, max_steps=200):
    """T1: Hill Climbing clásico (añadir, borrar, invertir) sobre el score del entorno."""
    adj = (np.asarray(adj0) > 0).copy()
    d = adj.shape[0]
    mp = max_parents or d
    pa = [list(np.nonzero(adj[:, j])[0]) for j in range(d)]
    loc = [feats.local(j, pa[j]) for j in range(d)]
    for _ in range(max_steps):
        R = _reach(adj)
        best, move = 1e-9, None
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if adj[i, j]:
                    nj = [p for p in pa[j] if p != i]
                    dl = feats.local(j, nj) - loc[j]
                    if dl > best:
                        best, move = dl, ('del', i, j)
                    if len(pa[i]) >= mp:
                        continue
                    adj[i, j] = False
                    blocked = _reach(adj)[i, j]
                    adj[i, j] = True
                    if blocked:
                        continue
                    dl = (feats.local(j, nj) - loc[j]) + (feats.local(i, pa[i] + [j]) - loc[i])
                    if dl > best:
                        best, move = dl, ('rev', i, j)
                elif not adj[j, i] and not R[j, i] and len(pa[j]) < mp:
                    dl = feats.local(j, pa[j] + [i]) - loc[j]
                    if dl > best:
                        best, move = dl, ('add', i, j)
        if move is None:
            break
        op, i, j = move
        if op == 'add':
            adj[i, j] = True
        elif op == 'del':
            adj[i, j] = False
        else:
            adj[i, j] = False
            adj[j, i] = True
        for k in {i, j}:
            pa[k] = list(np.nonzero(adj[:, k])[0])
            loc[k] = feats.local(k, pa[k])
    return adj.astype(np.int_)
