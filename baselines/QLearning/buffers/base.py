from typing import Callable, Any, Tuple

import chex
import jax.lax

ReplayBufferState = Any
Item = chex.ArrayTree
ItemBatch = chex.ArrayTree
IntScalar = chex.Array
ItemUpdateFn = Callable[[Item], Item]


@chex.dataclass(frozen=True)
class ReplayBuffer:
    init_fn: Callable[[Item], ReplayBufferState]
    size_fn: Callable[[ReplayBufferState], IntScalar]
    add_fn: Callable[[ReplayBufferState, Item], ReplayBufferState]
    add_batch_fn: Callable[[ReplayBufferState, ItemBatch], ReplayBufferState]
    sample_fn: Callable[[ReplayBufferState, chex.PRNGKey, int], ItemBatch]
    update_fn: Callable[[ReplayBufferState, ItemUpdateFn], ReplayBufferState]


def make_default_add_batch_fn(add_fn):
    def add_batch_fn(state: ReplayBufferState, item_batch: ItemBatch) -> ReplayBufferState:
        def scan_body(state: ReplayBufferState, item: Item) -> Tuple[ReplayBufferState, None]:
            state = add_fn(state, item)
            return state, None

        state, _ = jax.lax.scan(f=scan_body, init=state, xs=item_batch)
        return state

    return add_batch_fn


