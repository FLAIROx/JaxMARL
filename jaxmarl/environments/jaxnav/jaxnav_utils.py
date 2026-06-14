"""
Utility functions for simulators
"""

import chex
import jax
import jax.numpy as jnp


### --- MATHS UTILS ---
def pol2cart(rho: float, phi: float) -> chex.Array:
    """Convert polar coordinates into cartesian"""
    x = rho * jnp.cos(phi)
    y = rho * jnp.sin(phi)
    return jnp.array([x, y])


def cart2pol(x, y) -> chex.Array:
    rho = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    return jnp.array([rho, phi])


def unitvec(theta) -> chex.Array:
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def wrap(angle):
    """Ensure angle lies in the range [-pi, pi]"""

    def large(x):
        return x - 2 * jnp.pi

    def small(x):
        return x + 2 * jnp.pi

    def no_change(x):
        return x

    wrapped_angle = jax.lax.cond(angle >= jnp.pi, large, no_change, angle)
    wrapped_angle = jax.lax.cond(angle < -jnp.pi, small, no_change, wrapped_angle)

    return wrapped_angle


def euclid_dist(x, y):
    return jnp.norm(x - y)


def rot_mat(theta):
    """2x2 rotation matrix for 2D about the origin"""
    return jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    ).squeeze()
