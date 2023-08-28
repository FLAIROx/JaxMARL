from typing import Tuple, Callable

import chex
import jax.numpy as jnp
import jax.experimental.checkify as checkify
import jax.random

from .base import (
    ReplayBuffer, ReplayBufferState, Item, ItemBatch, IntScalar, ItemUpdateFn,
    make_default_add_batch_fn
)
from . import utils


@chex.dataclass(frozen=True)
class ClusteredReplayBufferState:
    cluster_buffer: ReplayBuffer
    cluster_state_batch: ReplayBufferState
    clustering_fn: Callable[[Item], IntScalar]
    distribution_power: float


def clustered_replay(
    num_clusters: int,
    cluster_buffer: ReplayBuffer,
    clustering_fn: Callable[[Item], IntScalar],
    distribution_power: float = 0.0,
) -> ReplayBuffer:
    def init_fn(item_prototype: Item) -> ClusteredReplayBufferState:
        cluster_states = [cluster_buffer.init_fn(item_prototype) for _ in range(num_clusters)]
        cluster_state_batch = utils.stack_trees(*cluster_states)
        return ClusteredReplayBufferState(
            cluster_buffer=cluster_buffer,
            cluster_state_batch=cluster_state_batch,
            clustering_fn=jax.tree_util.Partial(clustering_fn),
            distribution_power=distribution_power,
        )

    def size_fn(state: ClusteredReplayBufferState) -> IntScalar:
        return jnp.sum(jax.vmap(cluster_buffer.size_fn)(state.cluster_state_batch))

    def add_fn(state: ClusteredReplayBufferState, item: Item) -> ClusteredReplayBufferState:
        cluster_state_batch = state.cluster_state_batch
        cluster_index = state.clustering_fn(item)
        item_cluster = utils.get_pytree_batch_item(cluster_state_batch, cluster_index)
        item_cluster = state.cluster_buffer.add_fn(item_cluster, item)
        cluster_state_batch = utils.set_pytree_batch_item(cluster_state_batch, cluster_index, item_cluster)
        return state.replace(cluster_state_batch=cluster_state_batch)

    def sample_fn(state: ClusteredReplayBufferState, rng: chex.PRNGKey, batch_size: int) -> ItemBatch:
        cluster_sizes = jax.vmap(state.cluster_buffer.size_fn)(state.cluster_state_batch)
        cluster_weights = jnp.where(
            cluster_sizes > 0, jnp.power(cluster_sizes, state.distribution_power), cluster_sizes)

        cluster_fractions = cluster_weights / jnp.sum(cluster_weights)
        num_samples = jnp.round(batch_size * cluster_fractions).astype(jnp.int32)
        checkify.check(jnp.sum(num_samples) == batch_size, 'Number of samples does not match batch size')

        rng, cluster_selection_key = jax.random.split(rng)
        cluster_for_sample = jax.random.categorical(
            cluster_selection_key, logits=jnp.log(cluster_weights), shape=(batch_size,))
        rng_batch = jax.random.split(rng, batch_size)

        def sample_item(cluster_index: IntScalar, rng: chex.PRNGKey) -> Item:
            chex.assert_shape(cluster_index, ())

            cluster_state = utils.get_pytree_batch_item(state.cluster_state_batch, cluster_index)
            sample = state.cluster_buffer.sample_fn(cluster_state, rng, 1)

            chex.assert_tree_shape_prefix(sample, (1,))
            return utils.get_pytree_batch_item(sample, 0)

        return jax.vmap(sample_item)(cluster_for_sample, rng_batch)

    def update_fn(state: ClusteredReplayBufferState, item_update_fn: ItemUpdateFn) -> ClusteredReplayBufferState:
        def cluster_update_fn(cluster_state: ReplayBufferState) -> ReplayBufferState:
            return state.cluster_buffer.update_fn(cluster_state, item_update_fn)
        batch_cluster_update_fn = jax.vmap(cluster_update_fn)
        updated_cluster_state_batch = batch_cluster_update_fn(state.cluster_state_batch)
        return state.replace(cluster_state_batch=updated_cluster_state_batch)

    return ReplayBuffer(
        init_fn=jax.tree_util.Partial(init_fn),
        size_fn=jax.tree_util.Partial(size_fn),
        add_fn=jax.tree_util.Partial(add_fn),
        add_batch_fn=jax.tree_util.Partial(make_default_add_batch_fn(add_fn)),
        sample_fn=jax.tree_util.Partial(sample_fn),
        update_fn=jax.tree_util.Partial(update_fn),
    )
