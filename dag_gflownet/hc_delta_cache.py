"""Caché de delta-scores locales para el prefill dirigido.

La caché se utiliza durante el prefill constructivo tipo Hill-Climbing / Algorithm B
para evitar recomputar deltas locales ya evaluados.
"""

import numpy as np


class HCDeltaCache:
    """Cache de deltas locales para el prefill Hill-Climbing constructivo.

    La clave representa exactamente una transición de adición ``source -> target``
    desde un estado local concreto del nodo destino ``target``::

        (target, source, parents_before)

    En un score descomponible localmente, el delta de añadir ``source -> target``
    depende únicamente del nodo destino, de sus padres actuales y del nuevo padre.
    Por tanto, la misma clave devuelve siempre el mismo delta mientras el dataset
    y la configuración del scorer no cambien.
    """

    def __init__(self):
        self._values = {}
        self.hits = 0
        self.misses = 0
        self.stores = 0

    def make_key(self, obs, source, target):
        """Construye una clave hashable a partir de ``obs`` y de la arista.

        Parameters
        ----------
        obs : dict
            Observación del entorno con una única instancia activa.
        source : int
            Nodo origen de la arista candidata.
        target : int
            Nodo destino de la arista candidata.
        """
        adjacency = np.asarray(obs["adjacency"][0])
        source = int(source)
        target = int(target)

        # Padres actuales del nodo destino antes de añadir source -> target.
        parents = tuple(int(p) for p in np.flatnonzero(adjacency[:, target]))
        return (target, source, parents)

    def get(self, key):
        """Devuelve el delta almacenado para ``key`` o ``None`` si no existe."""
        if key in self._values:
            self.hits += 1
            return self._values[key]
        self.misses += 1
        return None

    def put(self, key, value):
        """Almacena un delta local en la caché."""
        if key not in self._values:
            self.stores += 1
        self._values[key] = float(value)

    def __len__(self):
        return len(self._values)

    def summary(self):
        """Devuelve estadísticas agregadas de uso de la caché."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "cache_size": int(len(self)),
            "cache_hits": int(self.hits),
            "cache_misses": int(self.misses),
            "cache_stores": int(self.stores),
            "cache_hit_rate": float(hit_rate),
        }