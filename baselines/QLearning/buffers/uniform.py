import chex
import jax
import jax.experimental.checkify as checkify

from . import circular_buffer
from .base import ReplayBuffer, Item, ItemBatch, IntScalar, ItemUpdateFn, make_default_add_batch_fn


@chex.dataclass(frozen=True)
class UniformReplayBufferState:
    storage: circular_buffer.CircularBuffer


def uniform_sample(
        buffer: circular_buffer.CircularBuffer, rng: chex.PRNGKey, batch_size: int
) -> circular_buffer.ItemBatch:
    checkify.check(circular_buffer.size(buffer) > 0, 'Cannot sample from an empty buffer')

    sample_pos = jax.random.randint(rng, minval=0, maxval=circular_buffer.size(buffer), shape=(batch_size,))
    get_at_index_batch = jax.vmap(circular_buffer.get_at_index, in_axes=(None, 0))
    return get_at_index_batch(buffer, sample_pos)


def uniform_replay(max_size: int) -> ReplayBuffer:
    def init_fn(item_prototype: Item) -> UniformReplayBufferState:
        return UniformReplayBufferState(storage=circular_buffer.init(item_prototype, max_size))

    def size_fn(state: UniformReplayBufferState) -> IntScalar:
        return circular_buffer.size(state.storage)

    def add_fn(state: UniformReplayBufferState, item: Item) -> UniformReplayBufferState:
        return state.replace(storage=circular_buffer.push(state.storage, item))

    def sample_fn(state: UniformReplayBufferState, rng: chex.PRNGKey, batch_size: int) -> ItemBatch:
        return uniform_sample(state.storage, rng, batch_size)

    def update_fn(state: UniformReplayBufferState, item_update_fn: ItemUpdateFn) -> UniformReplayBufferState:
        # TODO: there might be a faster way to make updates that does not affect all items in the buffer
        batch_update_fn = jax.vmap(item_update_fn)
        updated_data = batch_update_fn(state.storage.data)
        return state.replace(storage=state.storage.replace(data=updated_data))

    return ReplayBuffer(
        init_fn=jax.tree_util.Partial(init_fn),
        size_fn=jax.tree_util.Partial(size_fn),
        add_fn=jax.tree_util.Partial(add_fn),
        # TODO: it should be possible to make an optimized version of add_batch_fn for this buffer type
        add_batch_fn=jax.tree_util.Partial(make_default_add_batch_fn(add_fn)),
        sample_fn=jax.tree_util.Partial(sample_fn),
        update_fn=jax.tree_util.Partial(update_fn),
    )
