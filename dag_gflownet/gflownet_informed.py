"""DAGGFlowNet con política informada por el delta-score, templado beta y red objetivo configurable.

Uso (ver train_gflownet_v2.py):
    gfn = InformedDAGGFlowNet(features=DeltaFeatures(scorer, d), informed=True, ...)
    gfn.beta = 1.0                         # se puede cambiar en cada iteración (templado T2)
    params, state = gfn.init(subkey, optimizer, replay.dummy['graph'], replay.dummy['mask'])
    actions, key, logs, g = gfn.act(params.online, key, observations, epsilon, normalization)
    params, state, logs = gfn.step(params, state, samples, normalization)

act() y step() son métodos normales: calculan en CPU (NumPy, con caché) las matrices de delta-scores de
los estados y llaman a las versiones jit. Así posterior_estimate() y el resto del código base, que
llaman a gflownet.act(...), funcionan sin cambios.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad, random, jit, tree_util

from dag_gflownet.gflownet_c_b import DAGGFlowNet, DAGGFlowNetParameters, DAGGFlowNetState
from dag_gflownet.nets.gflownet_informed import gflownet_informed
from dag_gflownet.utils.gflownet_v2 import uniform_log_policy, detailed_balance_loss
from dag_gflownet.utils.jnp_utils import batch_random_choice


class InformedDAGGFlowNet(DAGGFlowNet):
    def __init__(self, features=None, informed=True, logit_scale_init=0.0, **kwargs):
        super().__init__(model=gflownet_informed(logit_scale_init), **kwargs)
        self.features = features
        self.informed = informed and features is not None
        self.beta = 1.0

    # ---------------------------------------------------------------- utilidades
    def _D(self, adjacency):
        if not self.informed:
            return np.zeros(np.asarray(adjacency).shape, dtype=np.float32)
        return self.features.matrix(np.asarray(adjacency))

    # ---------------------------------------------------------------- init
    def init(self, key, optimizer, graph, mask, gflownet_params=None):
        key, subkey = random.split(key)
        self._optimizer = optax.chain(optimizer, optax.zero_nans())
        if gflownet_params is not None:
            online_params = gflownet_params['params']
        else:
            D0 = jnp.zeros(mask.shape, dtype=jnp.float32)
            online_params = self.model.init(subkey, graph, mask, jnp.array(1.), D0, jnp.array(1.))
        params = DAGGFlowNetParameters(online=online_params, target=online_params)
        state = DAGGFlowNetState(optimizer=self.optimizer.init(online_params), key=key, steps=jnp.array(0))
        return params, state

    # ---------------------------------------------------------------- pérdida
    def loss(self, params, target_params, key, samples, normalization):
        beta = samples['beta']

        @partial(jax.vmap, in_axes=0)
        def _loss(log_pi_t, log_pi_tp1, actions, num_edges, scores):
            return detailed_balance_loss(log_pi_t, log_pi_tp1, actions, scores, num_edges,
                                         normalization=self.dataset_size, delta=self.delta)

        log_pi_t, rep_t = self.model.apply(params, samples['graph'], samples['mask'], normalization,
                                           samples['delta_matrix'], beta)
        log_pi_tp1, rep_tp1 = self.model.apply(target_params, samples['next_graph'], samples['next_mask'],
                                               normalization, samples['next_delta_matrix'], beta)
        cl_loss = self.contrastive_loss(rep_t, rep_tp1, rep_t.shape[0])
        outputs = _loss(log_pi_t, log_pi_tp1, samples['actions'], samples['num_edges'],
                        samples['delta_scores'] * beta)   # T2: log R = beta * score
        balance_loss, logs = tree_util.tree_map(partial(jnp.mean, axis=0), outputs)
        total_loss = balance_loss + self.contrastive_lambda * cl_loss
        logs.update({'balance_loss': balance_loss, 'contrastive_loss': cl_loss, 'total_loss': total_loss,
                     'error': outputs[1]['error']})
        return total_loss, logs

    # ---------------------------------------------------------------- step
    @partial(jit, static_argnums=(0,))
    def _step_jit(self, params, state, samples, normalization):
        key, subkey = random.split(state.key)
        grads, logs = grad(self.loss, has_aux=True)(params.online, params.target, subkey, samples, normalization)
        updates, opt_state = self.optimizer.update(grads, state.optimizer, params.online)
        state = DAGGFlowNetState(optimizer=opt_state, key=key, steps=state.steps + 1)
        online_params = optax.apply_updates(params.online, updates)
        if self.update_target_every > 0:
            target_params = optax.periodic_update(online_params, params.target, state.steps,
                                                  self.update_target_every)
        else:
            target_params = params.target
        return DAGGFlowNetParameters(online=online_params, target=target_params), state, logs

    def step(self, params, state, samples, normalization):
        samples = dict(samples)
        samples['delta_matrix'] = self._D(samples['adjacency'])
        samples['next_delta_matrix'] = self._D(samples['next_adjacency'])
        samples['beta'] = jnp.array(self.beta, dtype=jnp.float32)
        return self._step_jit(params, state, samples, normalization)

    # ---------------------------------------------------------------- act
    @partial(jit, static_argnums=(0,))
    def _act_jit(self, params, key, graphs, masks, delta_matrix, beta, epsilon, normalization):
        batch_size = masks.shape[0]
        key, subkey1, subkey2 = random.split(key, 3)
        log_pi, log_pi_global = self.model.apply(params, graphs, masks, normalization, delta_matrix, beta)
        log_uniform = uniform_log_policy(masks)
        is_exploration = random.bernoulli(subkey1, p=1. - epsilon, shape=(batch_size, 1))
        log_pi = jnp.where(is_exploration, log_uniform, log_pi)
        actions = batch_random_choice(subkey2, jnp.exp(log_pi), masks)
        return actions, key, {'is_exploration': is_exploration.astype(jnp.int32)}, log_pi_global

    def act(self, params, key, observations, epsilon, normalization):
        masks = jnp.asarray(observations['mask'], dtype=jnp.float32)
        D = self._D(observations['adjacency'])
        return self._act_jit(params, key, observations['graph'], masks, D,
                             jnp.array(self.beta, dtype=jnp.float32), epsilon, normalization)
