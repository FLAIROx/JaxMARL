import chex
import jax
import jax.numpy as jnp
from functools import partial


class Distribution:
    """Base class for all the start position and unit type generation"""

    def __init__(self, n_allies, n_enemies, map_width, map_height):
        self.n_allies = n_allies
        self.n_enemies = n_enemies
        self.map_width = map_width
        self.map_height = map_height

    @partial(jax.jit, static_argnums=(0,))
    def generate(self, key: chex.PRNGKey):
        raise NotImplementedError


class ReflectPositionDistribution(Distribution):
    @partial(jax.jit, static_argnums=(0,))
    def generate(self, key: chex.PRNGKey):
        key, ally_key = jax.random.split(key)
        ally_pos = jax.random.uniform(
            ally_key,
            shape=(self.n_allies, 2),
            minval=jnp.array([0.0, 0.0]),
            maxval=jnp.array([self.map_width / 2, self.map_height]),
        )
        enemy_pos = jnp.zeros((self.n_enemies, 2))
        min_size = min(self.n_enemies, self.n_allies)
        enemy_pos = enemy_pos.at[:min_size, 0].set(self.map_width - ally_pos[:, 0])
        enemy_pos = enemy_pos.at[:min_size, 1].set(ally_pos[:, 1])
        enemy_pos = jax.lax.select(
            min_size == self.n_enemies,
            enemy_pos,
            enemy_pos.at[self.n_allies :, :].set(
                jax.random.uniform(
                    key,
                    shape=(self.n_enemies - self.n_allies, 2),
                    minval=jnp.array([self.map_width / 2, 0.0]),
                    maxval=jnp.array([self.map_width, self.map_height]),
                )
            ),
        )
        return jnp.concatenate([ally_pos, enemy_pos])


class SurroundPositionDistribution(Distribution):
    @partial(jax.jit, static_argnums=(0,))
    def generate(self, key):
        # issue: want to randomly decide the centre and outside teams.
        # issue: don't know what that mean
        def draw_positions(key_, n_inside, n_outside):
            centre_pos = jnp.zeros((n_inside, 2))
            key_, centre_noise_key = jax.random.split(key_)
            centre_start_noise = jax.random.uniform(
                centre_noise_key, shape=(n_inside, 2), minval=-2, maxval=2
            )
            centre_pos = centre_pos.at[:, :].set(
                jnp.array([self.map_width / 2, self.map_height / 2])
                + centre_start_noise
            )
            n_groups = 4
            key_, key_groups = jax.random.split(key_)
            group_assignments = jax.random.categorical(
                key_groups,
                jnp.log(jnp.ones((n_groups,)) / n_groups),
                shape=(n_outside,),
            )
            centre = jnp.array(
                [[self.map_width / 2.0, self.map_height / 2.0]] * n_groups
            )
            edges = jnp.array(
                [
                    [0.0, 0.0],
                    [0.0, self.map_height],
                    [self.map_width, 0],
                    [self.map_width, self.map_height],
                ]
            )
            key_, t_key = jax.random.split(key_)
            t = jax.random.uniform(t_key, shape=(n_groups, 1), minval=0.0, maxval=1.0)
            group_positions = t * centre + (1 - t) * edges
            outside_pos = group_positions[group_assignments]
            key_, outside_noise_key = jax.random.split(key_)
            outside_pos_noise = jax.random.uniform(
                outside_noise_key, shape=(n_outside, 2), minval=-2, maxval=2
            )
            outside_pos = outside_pos + outside_pos_noise
            return {"outside": outside_pos, "inside": centre_pos}

        key, ally_key, enemy_key = jax.random.split(key, num=3)
        ally_inside_positions = draw_positions(ally_key, self.n_allies, self.n_enemies)
        ally_inside_positions = jnp.concatenate(
            [ally_inside_positions["inside"], ally_inside_positions["outside"]]
        )
        enemy_inside_positions = draw_positions(
            enemy_key, self.n_enemies, self.n_allies
        )
        enemy_inside_positions = jnp.concatenate(
            [enemy_inside_positions["outside"], enemy_inside_positions["inside"]]
        )
        ally_inside = jax.random.randint(key, shape=(), minval=0, maxval=2)
        return jax.lax.select(
            ally_inside, ally_inside_positions, enemy_inside_positions
        )


class SurroundAndReflectPositionDistribution(Distribution):
    def __init__(self, n_allies, n_enemies, map_width, map_height):
        super().__init__(n_allies, n_enemies, map_width, map_height)
        self.surround_distribution = SurroundPositionDistribution(
            n_allies, n_enemies, map_width, map_height
        )
        self.reflect_distribution = ReflectPositionDistribution(
            n_allies, n_enemies, map_width, map_height
        )

    def generate(self, key):
        key_draw, key_surround, key_reflect = jax.random.split(key, num=3)
        val = jax.random.uniform(key_draw)
        return jax.lax.select(
            val < 0.5,
            self.surround_distribution.generate(key_surround),
            self.reflect_distribution.generate(key_reflect),
        )


class UniformUnitTypeDistribution(Distribution):
    def __init__(self, n_allies, n_enemies, map_width, map_height, n_unit_types):
        super().__init__(n_allies, n_enemies, map_width, map_height)
        self.n_unit_types = n_unit_types

    def generate(self, key):
        enemy_key, ally_key = jax.random.split(key)
        ally_unit_types = jax.random.categorical(
            ally_key,
            jnp.log(jnp.ones((self.n_unit_types,)) / self.n_unit_types),
            shape=(self.n_allies,),
        ).astype(jnp.uint8)
        enemy_unit_types = jnp.zeros((self.n_enemies,), dtype=jnp.uint8)
        min_size = min(self.n_allies, self.n_enemies)
        enemy_unit_types = enemy_unit_types.at[:min_size].set(ally_unit_types)

        enemy_unit_types = jax.lax.select(
            min_size == self.n_enemies,
            enemy_unit_types,
            enemy_unit_types.at[min_size:].set(
                jax.random.categorical(
                    enemy_key,
                    jnp.log(jnp.ones((self.n_unit_types)) / self.n_unit_types),
                    shape=(self.n_enemies - self.n_allies,),
                ).astype(jnp.uint8)
            ),
        )
        return jnp.concatenate([ally_unit_types, enemy_unit_types], dtype=jnp.uint8)
