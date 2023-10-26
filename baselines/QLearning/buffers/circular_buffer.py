import chex
import jax
import jax.numpy as jnp
import jax.experimental.checkify as checkify

from . import utils

IntScalar = chex.Array
BoolScalar = chex.Array
Item = chex.ArrayTree
ItemBatch = chex.ArrayTree


@chex.dataclass(frozen=True)
class CircularBuffer:
    data: ItemBatch
    head: IntScalar
    tail: IntScalar
    full: BoolScalar


def init(item_prototype: Item, max_size: int) -> CircularBuffer:
    chex.assert_tree_has_only_ndarrays(item_prototype)

    data = jax.tree_util.tree_map(
        lambda t: utils.tile_over_axis(t, axis=0, size=max_size), item_prototype)
    return CircularBuffer(
        data=data,
        head=utils.scalar_to_jax(0),
        tail=utils.scalar_to_jax(0),
        full=utils.scalar_to_jax(False),
    )


def max_size(buffer: CircularBuffer) -> int:
    return utils.get_pytree_axis_dim(buffer.data, axis=0)


def size(buffer: CircularBuffer) -> IntScalar:
    return jax.lax.select(
        buffer.full,
        on_true=max_size(buffer),
        on_false=jax.lax.select(
            buffer.head >= buffer.tail,
            on_true=buffer.head - buffer.tail,
            on_false=max_size(buffer) - (buffer.tail - buffer.head),
        ),
    )


def push(buffer: CircularBuffer, item: Item) -> CircularBuffer:
    chex.assert_tree_has_only_ndarrays(item)

    insert_pos = buffer.head
    new_data = utils.set_pytree_batch_item(buffer.data, insert_pos, item)
    new_head = (insert_pos + 1) % max_size(buffer)
    new_tail = jax.lax.select(
        buffer.full,
        on_true=new_head,
        on_false=buffer.tail,
    )
    new_full = new_head == new_tail

    return buffer.replace(data=new_data, head=new_head, tail=new_tail, full=new_full)


def pop(buffer: CircularBuffer) -> (Item, CircularBuffer):
    checkify.check(size(buffer) > 0, 'There are no items in storage')

    remove_pos = buffer.tail
    popped_item = utils.get_pytree_batch_item(buffer.data, remove_pos)
    new_tail = (remove_pos + 1) % max_size(buffer)
    new_full = utils.scalar_to_jax(False)

    return popped_item, buffer.replace(tail=new_tail, full=new_full)


def get_at_index(buffer: CircularBuffer, index: IntScalar) -> Item:
    chex.assert_shape(index, ())
    checkify.check(jnp.all(jnp.logical_and(index >= 0, index < size(buffer))), 'Index out of bounds')

    index = (buffer.tail + index) % max_size(buffer)
    return utils.get_pytree_batch_item(buffer.data, index)
