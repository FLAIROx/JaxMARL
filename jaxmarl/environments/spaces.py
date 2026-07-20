"""Built off Gymnax spaces.py, this module contains jittable classes for action and observation spaces."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import (  # type: ignore[attr-defined]
    Array,
    Bool,
    Float,
    Int,
    Num,
    PRNGKeyArray,
)


class Space(object):
    """Abstract base class for JAX-compatible spaces."""

    def sample(self, rng: PRNGKeyArray) -> Num[Array, "..."]:
        raise NotImplementedError

    def contains(self, x: Num[Array, "..."]) -> Bool[Array, ""]:
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete gymnax spaces.

    TODO: For now this is a 1d space. Make composable for multi-discrete.
    """

    def __init__(self, num_categories: int, dtype=jnp.int32):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = dtype

    def sample(self, rng: PRNGKeyArray) -> Int[Array, ""]:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: Int[Array, ""]) -> Bool[Array, ""]:
        """Check whether ``x`` is within the space."""
        return jnp.logical_and(x >= 0, x < self.n)


class MultiDiscrete(Space):
    """Multi-dimensional discrete space.

    Each dimension has its own number of categories. For example,
    ``[2, 3, 4]`` defines a space with shape ``(3,)`` where the first
    dimension has 2 categories, the second has 3, and the third has 4.
    """

    def __init__(self, num_categories: Sequence[int]):
        """Initialise the multi-discrete space.

        Args:
            num_categories: Number of categories for each discrete dimension.
        """
        self.num_categories = jnp.array(num_categories)
        self.shape = (len(num_categories),)
        self.dtype = jnp.int32

    def sample(self, rng: PRNGKeyArray) -> Int[Array, "num_categories"]:  # noqa: F821
        """Sample a random value uniformly for each discrete dimension."""
        return jax.random.randint(
            rng,
            shape=self.shape,
            minval=0,
            maxval=self.num_categories,
            dtype=self.dtype,
        )

    def contains(self, x: Int[Array, "num_categories"]) -> Bool[Array, ""]:  # noqa: F821
        """Check whether ``x`` is valid for every discrete dimension."""
        return jnp.all(jnp.logical_and(x >= 0, x < self.num_categories))


class Box(Space):
    """Array-shaped continuous space with lower and upper bounds.

    Samples are drawn uniformly from ``[low, high]``.
    TODO: Add unboundedness - sampling from other distributions, etc.
    """

    def __init__(
        self,
        low: jax.typing.ArrayLike,
        high: jax.typing.ArrayLike,
        shape: tuple[int, ...],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng: PRNGKeyArray) -> Float[Array, "..."]:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            rng, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: Num[Array, "..."]) -> Bool[Array, ""]:
        """Check whether all entries of ``x`` lie within the box bounds."""
        return jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))


class Dict(Space):
    """Dictionary space composed of named subspaces."""

    def __init__(self, spaces: Mapping[str, Space]):
        """Initialise the dictionary space.

        Args:
            spaces: Mapping from names to subspaces.
        """
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, rng: PRNGKeyArray) -> OrderedDict[str, Any]:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(rng, self.num_spaces)
        return OrderedDict(
            [
                (k, self.spaces[k].sample(key_split[i]))
                for i, k in enumerate(self.spaces)
            ]
        )

    def contains(self, x: Mapping[str, Any]) -> Bool[Array, ""]:
        """Check whether dimensions of object are within subspace."""
        out_of_space: Array = jnp.array(0)
        for k, space in self.spaces.items():
            out_of_space = out_of_space + (1 - space.contains(x[k]))
        return out_of_space == 0


class Tuple(Space):
    """Tuple/product space composed of multiple subspaces."""

    def __init__(self, spaces: tuple[Space, ...] | list[Space]):
        """Initialise the tuple space.

        Args:
            spaces: Ordered collection of subspaces.
        """
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, rng: PRNGKeyArray) -> tuple[Any, ...]:
        """Sample independently from each subspace."""
        key_split = jax.random.split(rng, self.num_spaces)
        return tuple(
            [space.sample(key_split[i]) for i, space in enumerate(self.spaces)]
        )

    def contains(self, x: tuple[Any, ...]) -> Bool[Array, ""]:
        """Check whether dimensions of object are within subspace."""
        out_of_space: Array = jnp.array(0)
        for i, space in enumerate(self.spaces):
            out_of_space = out_of_space + (1 - space.contains(x[i]))
        return out_of_space == 0
