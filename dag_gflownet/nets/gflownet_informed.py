"""Política DAG-GFlowNet informada por el delta-score (ampliación del TFG).

Es la red original de dag_gflownet/nets/gflownet.py con tres cambios:
  1. logits(i->j) = logits_red(i->j) + beta * D[i, j], con D = delta-score de añadir i->j.
     Sin entrenar, la política ya es un "Algorithm B estocástico" (softmax sobre las mejoras de score);
     la red solo aprende la corrección.
  2. stop = stop_red * exp(log_escala) - c * logsumexp_validas(beta * D): si hay mejoras grandes
     disponibles, parar es improbable. log_escala y c son parámetros aprendibles.
  3. Si D es None se comporta como la red original (más la escala aprendible del stop).
"""
import jax
import jax.numpy as jnp
import haiku as hk
import jraph
from jax import lax

from dag_gflownet.utils.gflownet import log_policy


def gflownet_informed(logit_scale_init=0.0):
    @hk.without_apply_rng
    @hk.transform
    def _gflownet(graphs, masks, normalization, delta_matrix=None, beta=1.0):
        batch_size, num_variables = masks.shape[:2]

        node_embeddings = hk.Embed(num_variables, embed_dim=128)
        edge_embedding = hk.get_parameter('edge_embed', shape=(1, 128),
            init=hk.initializers.TruncatedNormal())
        graphs = graphs._replace(
            nodes=node_embeddings(graphs.nodes),
            edges=jnp.repeat(edge_embedding, graphs.edges.shape[0], axis=0),
            globals=jnp.zeros((graphs.n_node.shape[0], 1)),
        )

        @jraph.concatenated_args
        def update_node_fn(features):
            return hk.nets.MLP([128, 128], name='node')(features)

        @jraph.concatenated_args
        def update_edge_fn(features):
            return hk.nets.MLP([128, 128], name='edge')(features)

        @jraph.concatenated_args
        def update_global_fn(features):
            return hk.nets.MLP([128, 128], name='global')(features)

        graph_net = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
        )
        features = graph_net(graphs)

        node_features = features.nodes[:batch_size * num_variables]
        node_features = node_features.reshape(batch_size, num_variables, -1)
        node_features = hk.Linear(128 * 3, name='projection')(node_features)
        queries, keys, values = jnp.split(node_features, 3, axis=2)
        node_features = hk.MultiHeadAttention(num_heads=4, key_size=32, w_init_scale=2.)(queries, keys, values)

        features = features._replace(
            nodes=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(node_features),
            globals=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(features.globals[:batch_size])
        )

        senders = hk.nets.MLP([128, 128], name='senders')(features.nodes)
        receivers = hk.nets.MLP([128, 128], name='receivers')(features.nodes)
        logits = lax.batch_matmul(senders, receivers.transpose(0, 2, 1))
        logits = logits.reshape(batch_size, -1) * normalization
        stop = hk.nets.MLP([128, 1], name='stop')(features.globals) * normalization

        log_scale = hk.get_parameter('stop_log_scale', shape=(), init=hk.initializers.Constant(logit_scale_init))
        c_stop = hk.get_parameter('stop_lse_coef', shape=(), init=hk.initializers.Constant(1.0))
        stop = stop * jnp.exp(log_scale)

        if delta_matrix is not None:
            bD = beta * delta_matrix.reshape(batch_size, -1)
            logits = logits + bD
            valid = masks.reshape(batch_size, -1) > 0
            bD_valid = jnp.where(valid, bD, -jnp.inf)
            lse = jax.nn.logsumexp(bD_valid, axis=-1, keepdims=True)
            lse = jnp.where(jnp.isfinite(lse), lse, 0.0)
            stop = stop - c_stop * lax.stop_gradient(lse)

        return log_policy(logits, stop, masks), features.globals

    return _gflownet
