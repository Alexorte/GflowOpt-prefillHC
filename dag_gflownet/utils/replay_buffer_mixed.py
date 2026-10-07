"""W1: buffer experto protegido + muestreo mixto (symmetric sampling, Ball et al. 2023).

El buffer principal es el ReplayBuffer FIFO original; el experto es otro ReplayBuffer que solo
se llena con transiciones heurísticas (prefill, trayectorias hacia atrás, HC) y NO se sobrescribe
con experiencia online. Cada minilote toma una fracción rho del experto.
"""
import numpy as np

from dag_gflownet.utils.jraph_utils import to_graphs_tuple


def _decode(buffer, samples):
    adjacency = buffer.decode(samples['adjacency'], dtype=np.int_)
    next_adjacency = buffer.decode(samples['next_adjacency'], dtype=np.int_)
    return {
        'adjacency': adjacency.astype(np.float32),
        'graph': to_graphs_tuple(adjacency),
        'num_edges': samples['num_edges'],
        'actions': samples['actions'],
        'delta_scores': samples['delta_scores'],
        'mask': buffer.decode(samples['mask']),
        'next_adjacency': next_adjacency.astype(np.float32),
        'next_graph': to_graphs_tuple(next_adjacency),
        'next_mask': buffer.decode(samples['next_mask']),
    }


class MixedReplay:
    def __init__(self, main, expert):
        self.main = main
        self.expert = expert

    def __len__(self):
        return len(self.main) + len(self.expert)

    def sample(self, batch_size, rng, rho):
        n_exp = 0
        if len(self.expert) > 0:
            n_exp = batch_size if len(self.main) == 0 else int(round(rho * batch_size))
        n_main = batch_size - n_exp
        parts = []
        if n_exp > 0:
            idx = rng.choice(len(self.expert), size=n_exp, replace=len(self.expert) < n_exp)
            parts.append(self.expert._replay[idx])
        if n_main > 0:
            idx = rng.choice(len(self.main), size=n_main, replace=len(self.main) < n_main)
            parts.append(self.main._replay[idx])
        samples = np.concatenate(parts)
        return _decode(self.main, samples)
