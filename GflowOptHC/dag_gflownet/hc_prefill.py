import numpy as np

#Esta funciÃ³n asume env.num_envs == 1 en el entorno real.
#Se crea un batch virtual interno de tamaÃ±o K para evaluar candidatos.

# basicamente, se pasa a los metodos que nos devuelve los deltas
# un batch virtual con K copias del mismo estado. En el for, en la iteracion 1
# se evalua el candidato 1, en la iteracion 2 el candidato 2, etc. 

# VALIDATED: delta_async == delta_step (see unit test test_hc.py)

def evaluar_candidatos(env, obs, sources, targets):

    # numero de candidatos a evaluar
    K = int(len(sources)) 
    if K == 0:
        return np.zeros((0,), dtype=np.float64)

    # respaldo del estado real del env para restaurarlo al final
    estado_real = env._state
    clausura_real = env._closure_T

    # Batch virtual: duplicar el estado 1-env a K-envs
    estado_virtual = {
        "adjacency": np.repeat(obs["adjacency"][:1], K, axis=0),
        "mask":      np.repeat(obs["mask"][:1], K, axis=0),
        "num_edges": np.repeat(obs["num_edges"][:1], K, axis=0),
        "score":     np.repeat(obs["score"][:1], K, axis=0),
        "order":     np.repeat(obs["order"][:1], K, axis=0),
    }
    clausura_virtual_T = np.repeat(env._closure_T[:1], K, axis=0)

    # Inyectar en el env
    env._state = estado_virtual
    env._closure_T = clausura_virtual_T

    try:
        keys, local_cache, queued_data = env.local_scores_async(sources, targets)
        deltas = env.local_scores_wait(keys, local_cache, queued_data)
    finally:
        # Restaurar el env real
        env._state = estado_real
        env._closure_T = clausura_real

    return deltas

def seleccionar_accion(env, obs, epsilon=0.2, top_k=5, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    d = env.num_variables
    accion_terminal = d * d

    # aristas validas
    mask = obs["mask"][0]
    candidatos = np.argwhere(mask == 1)

    if len(candidatos) == 0:
        return accion_terminal, False
    
    sources = candidatos[:, 0]
    targets = candidatos[:, 1]

    deltas = evaluar_candidatos(env, obs, sources, targets)
    # filtrar acciones que mejoran el score
    mejoras = np.where(deltas > 0)[0]

    if len(mejoras) == 0:
        return accion_terminal, False
    
    mejores_idx = mejoras[np.argmax(deltas[mejoras])] # indice del candidato con mayor mejora
    mejor_accion = int(sources[mejores_idx] * d + targets[mejores_idx]) # convertir a accion 
    
    if rng.random() < epsilon:
        # exploracion: elegir aleatoriamente entre las mejores acciones
        ordenadas = mejoras[np.argsort(deltas[mejoras])[::-1]] # indices de mejoras ordenados por delta descendente
        k = min(top_k, len(ordenadas))
        escogida = rng.choice(ordenadas[:k]) # elegir aleatoriamente entre las top-k mejores
        accion = int(sources[escogida] * d + targets[escogida]) # convertir a accion
        return accion, accion != mejor_accion    
    else:
        # explotacion: elegir la mejor accion
        return mejor_accion, False
    
def prefill(env, replay_buffer, rng, epsilon=0.2, top_k=5, max_steps=None):
    obs = env.reset()
    steps = 0
    done = False
    cum_delta = 0.0
    num_exploration = 0

    # Estado terminal rastreado manualmente
    final_adj = obs["adjacency"][0].copy()
    final_num_edges = int(np.sum(final_adj))
    final_score = float(obs["score"][0])

    d = env.num_variables
    accion_terminal = d * d

    while not done:
        action, is_exploration = seleccionar_accion(env, obs, epsilon=epsilon, top_k=top_k, rng=rng)
        next_obs, delta_score, done, _ = env.step(np.array([action], dtype=np.int_))

        replay_buffer.add(
            observations=obs,
            actions=np.array([action], dtype=np.int_),
            is_exploration=np.array([is_exploration], dtype=np.bool_),
            next_observations=next_obs,
            delta_scores=delta_score,
            dones=done
        )

        steps += 1
        num_exploration += int(is_exploration)
        cum_delta += float(delta_score[0])

        # Actualizar estado final manualmente solo si no es acción terminal
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

    return {
        "steps": steps,
        "exploration_steps": num_exploration,
        "acum_delta": cum_delta,
        "final_adj": final_adj.copy(),
        "final_num_edges": int(final_num_edges),
        "final_score": float(final_score),
    }

def prefill_hc(env, replay_buffer, rng, num_episodes=1000, epsilon=0.2, top_k=5, max_steps=None, history_every=10, history_callback=None):

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
        )
        stats.append(s)
        if history_callback is not None and ((ep + 1) % history_every == 0 or (ep + 1) == num_episodes):
            history.append(history_callback(stats, ep + 1))
    return stats, history
