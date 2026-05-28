import numpy as np


class HCDeltaCache:
    """Cache de deltas locales para el prefill Hill-Climbing constructivo.

    La clave representa exactamente una transición de adición i -> j desde un
    estado local concreto del nodo destino j:

        (target=j, source=i, parents_before=Pa_G(j))

    En un score descomponible localmente, el delta de añadir i -> j depende solo
    del nodo destino j, de sus padres actuales y del nuevo padre i. Por tanto, la
    misma clave debe devolver siempre el mismo delta mientras el dataset y la
    configuración del scorer no cambien.
    """

    def __init__(self):
        self._values = {}
        self.hits = 0
        self.misses = 0
        self.stores = 0

    def make_key(self, obs, source, target):
        """Construye una clave hashable a partir de obs y de la arista source->target."""
        adjacency = np.asarray(obs["adjacency"][0])
        source = int(source)
        target = int(target)

        # Padres actuales del nodo destino antes de añadir source -> target.
        parents = tuple(int(p) for p in np.flatnonzero(adjacency[:, target]))
        return (target, source, parents)

    def get(self, key):
        if key in self._values:
            self.hits += 1
            return self._values[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key not in self._values:
            self.stores += 1
        self._values[key] = float(value)

    def __len__(self):
        return len(self._values)

    def summary(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "cache_size": int(len(self)),
            "cache_hits": int(self.hits),
            "cache_misses": int(self.misses),
            "cache_stores": int(self.stores),
            "cache_hit_rate": float(hit_rate),
        }

# Esta función asume env.num_envs == 1 en el entorno real.
# Se crea un batch virtual interno de tamaño K para evaluar candidatos.
#
# Básicamente, se pasa a los métodos que devuelven los deltas un batch virtual
# con K copias del mismo estado. En la posición k del batch se evalúa el
# candidato k.
#
# VALIDATED: delta_async == delta_step (see unit test test_hc.py)

def _evaluar_candidatos_sin_cache(env, obs, sources, targets):
    """Evalúa candidatos delegando directamente en el entorno de la GFlowNet."""

    K = int(len(sources))
    if K == 0:
        return np.zeros((0,), dtype=np.float64)

    # Respaldo del estado real del env para restaurarlo al final.
    estado_real = env._state
    clausura_real = env._closure_T

    # Batch virtual: duplicar el estado 1-env a K-envs.
    estado_virtual = {
        "adjacency": np.repeat(obs["adjacency"][:1], K, axis=0),
        "mask":      np.repeat(obs["mask"][:1], K, axis=0),
        "num_edges": np.repeat(obs["num_edges"][:1], K, axis=0),
        "score":     np.repeat(obs["score"][:1], K, axis=0),
        "order":     np.repeat(obs["order"][:1], K, axis=0),
    }
    clausura_virtual_T = np.repeat(env._closure_T[:1], K, axis=0)

    # Inyectar en el env.
    env._state = estado_virtual
    env._closure_T = clausura_virtual_T

    try:
        keys, local_cache, queued_data = env.local_scores_async(sources, targets)
        deltas = env.local_scores_wait(keys, local_cache, queued_data)
    finally:
        # Restaurar siempre el env real, incluso si falla el cálculo de scores.
        env._state = estado_real
        env._closure_T = clausura_real

    return np.asarray(deltas, dtype=np.float64)


def evaluar_candidatos(env, obs, sources, targets, delta_cache=None):
    """Evalúa los deltas de los candidatos, usando caché opcional.

    Si delta_cache es None, mantiene exactamente el comportamiento original.
    Si delta_cache existe, solo envía al entorno los candidatos que no estén en
    caché y recompone el vector de deltas en el orden original.
    """

    sources = np.asarray(sources, dtype=np.int_)
    targets = np.asarray(targets, dtype=np.int_)
    K = int(len(sources))

    if K == 0:
        return np.zeros((0,), dtype=np.float64)

    if delta_cache is None:
        return _evaluar_candidatos_sin_cache(env, obs, sources, targets)

    deltas = np.empty((K,), dtype=np.float64)
    missing_positions = []
    missing_sources = []
    missing_targets = []
    missing_keys = []

    for pos, (src, dst) in enumerate(zip(sources, targets)):
        key = delta_cache.make_key(obs, src, dst)
        value = delta_cache.get(key)

        if value is None:
            # Vamos agrupando candidatos que no están en caché
            missing_positions.append(pos) 
            missing_sources.append(int(src))
            missing_targets.append(int(dst))
            missing_keys.append(key)
        else:
            deltas[pos] = float(value)

    # Evaluamos candidatos que no estaban en caché.
    if missing_positions: 
        computed = _evaluar_candidatos_sin_cache(
            env=env,
            obs=obs,
            sources=np.asarray(missing_sources, dtype=np.int_),
            targets=np.asarray(missing_targets, dtype=np.int_),
        )

        for pos, key, delta in zip(missing_positions, missing_keys, computed):
            delta = float(delta)
            deltas[pos] = delta
            delta_cache.put(key, delta)

    return deltas


def seleccionar_accion(env, obs, epsilon=0.2, top_k=5, rng=None, delta_cache=None):

    if rng is None:
        rng = np.random.default_rng()

    d = env.num_variables
    accion_terminal = d * d

    # Aristas válidas según la máscara del entorno. La máscara ya excluye
    # self-loops, aristas existentes y acciones que introducirían ciclos.
    mask = obs["mask"][0]
    candidatos = np.argwhere(mask == 1)

    if len(candidatos) == 0:
        return accion_terminal, False

    sources = candidatos[:, 0]
    targets = candidatos[:, 1]

    deltas = evaluar_candidatos(
        env=env,
        obs=obs,
        sources=sources,
        targets=targets,
        delta_cache=delta_cache,
    )

    # Filtrar acciones que mejoran el score.
    mejoras = np.where(deltas > 0)[0]

    if len(mejoras) == 0:
        return accion_terminal, False

    # Índice del candidato con mayor mejora.
    mejores_idx = mejoras[np.argmax(deltas[mejoras])]
    mejor_accion = int(sources[mejores_idx] * d + targets[mejores_idx])

    if rng.random() < epsilon:
        # Exploración: elegir aleatoriamente entre las top-k mejores acciones.
        ordenadas = mejoras[np.argsort(deltas[mejoras])[::-1]]
        k = min(top_k, len(ordenadas))
        escogida = rng.choice(ordenadas[:k])
        accion = int(sources[escogida] * d + targets[escogida])
        return accion, accion != mejor_accion

    # Explotación: elegir la mejor acción.
    return mejor_accion, False


def prefill(env, replay_buffer, rng, epsilon=0.2, top_k=5, max_steps=None, delta_cache=None):
    obs = env.reset()
    steps = 0
    done = False
    cum_delta = 0.0
    num_exploration = 0

    # Estado terminal rastreado manualmente.
    final_adj = obs["adjacency"][0].copy()
    final_num_edges = int(np.sum(final_adj))
    final_score = float(obs["score"][0])

    d = env.num_variables
    accion_terminal = d * d

    while not done:
        action, is_exploration = seleccionar_accion(
            env=env,
            obs=obs,
            epsilon=epsilon,
            top_k=top_k,
            rng=rng,
            delta_cache=delta_cache,
        )
        next_obs, delta_score, done, _ = env.step(np.array([action], dtype=np.int_))

        replay_buffer.add(
            observations=obs,
            actions=np.array([action], dtype=np.int_),
            is_exploration=np.array([is_exploration], dtype=np.bool_),
            next_observations=next_obs,
            delta_scores=delta_score,
            dones=done,
        )

        steps += 1
        num_exploration += int(is_exploration)
        cum_delta += float(delta_score[0])

        # Actualizar estado final manualmente solo si no es acción terminal.
        a = int(action)
        if a != accion_terminal:
            src, dst = divmod(a, d)
            final_adj[src, dst] = 1
            final_num_edges += 1
            final_score += float(delta_score[0])

        done = bool(done[0])
        obs = next_obs

        if max_steps is not None and steps >= max_steps:
            break

    result = {
        "steps": steps,
        "exploration_steps": num_exploration,
        "acum_delta": cum_delta,
        "final_adj": final_adj.copy(),
        "final_num_edges": int(final_num_edges),
        "final_score": float(final_score),
    }

    if delta_cache is not None:
        result.update(delta_cache.summary())

    return result


def prefill_hc(
    env,
    replay_buffer,
    rng,
    num_episodes=1000,
    epsilon=0.2,
    top_k=5,
    max_steps=None,
    history_every=10,
    history_callback=None,
    use_cache=True,
    delta_cache=None,
):
    """Ejecuta el prefill HC.

    La caché se mantiene entre episodios, que es donde realmente se amortiza el
    coste: muchos episodios visitan los mismos conjuntos de padres locales.
    """

    if use_cache and delta_cache is None:
        delta_cache = HCDeltaCache()
    elif not use_cache:
        delta_cache = None

    stats = []
    history = []

    for ep in range(num_episodes):
        s = prefill(
            env=env,
            replay_buffer=replay_buffer,
            rng=rng,
            epsilon=epsilon,
            top_k=top_k,
            max_steps=max_steps,
            delta_cache=delta_cache,
        )
        stats.append(s)

        if history_callback is not None and ((ep + 1) % history_every == 0 or (ep + 1) == num_episodes):
            history.append(history_callback(stats, ep + 1))

    return stats, history
