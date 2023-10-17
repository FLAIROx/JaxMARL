import chex
import jax.numpy as jnp
import jax.tree_util


def tile_over_axis(tensor, axis, size):
    tensor = jnp.expand_dims(tensor, axis)
    tensor = jnp.repeat(tensor, size, axis=axis)
    return tensor


def type_to_dtype(t):
    if t == int:
        return jnp.int32
    elif t == float:
        return jnp.float32
    elif t == bool:
        return jnp.bool_
    else:
        raise ValueError(f"Unsupported type: {t}")


def scalar_to_jax(scalar):
    chex.assert_scalar(scalar)
    return jnp.asarray(scalar, dtype=type_to_dtype(type(scalar)))


def assert_tree_is_batch_of_tree(tree_batch, tree):
    def cmp(tensor_batch, tensor):
        return tensor_batch.shape[1:] == tensor.shape

    def error(tensor_batch, tensor):
        return f"{tensor_batch.shape[1:]} != {tensor.shape}"

    chex.assert_trees_all_equal_comparator(cmp, error, tree_batch, tree)


def set_pytree_batch_item(tree_batch, index, tree):
    assert_tree_is_batch_of_tree(tree_batch, tree)

    return jax.tree_util.tree_map(
        lambda tb, t: tb.at[index].set(t),
        tree_batch, tree,
    )


def get_pytree_batch_item(tree_batch, index):
    return jax.tree_util.tree_map(lambda tb: tb[index], tree_batch)


def get_pytree_axis_dim(tree, axis):
    leaves = jax.tree_util.tree_leaves(tree)
    assert len(leaves) > 0
    axis_dim = leaves[0].shape[axis]
    assert all(leave.shape[axis] == axis_dim for leave in leaves)
    return axis_dim


def concatenate_trees(*trees, axis=0):
    return jax.tree_util.tree_map(lambda *t: jnp.concatenate(t, axis=axis), *trees)


def stack_trees(*trees, axis=0):
    return jax.tree_util.tree_map(lambda *t: jnp.stack(t, axis=axis), *trees)
