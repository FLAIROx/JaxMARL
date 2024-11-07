import jax
from jax import numpy as jnp
from flax import struct
import chex
import numpy as np
from functools import partial


@struct.dataclass
class OneParticleState:
    x: float
    y: float
    theta: float
    vel_x: float
    vel_y: float


@struct.dataclass
class ParticlesState:
    particles: OneParticleState
    weights: chex.Array


class ParticleFilter:

    def __init__(
        self,
        num_particles,
        std_range=10,  # m (standard deviation error of the range measurements)
        mu_init_vel=0.1,  # m/s
        std_init_vel=0.1,  # m/s
        turn_noise=0.1,  # rad
        vel_noise=0.05,  # m/s
        min_weight=0.01,
    ):
        self.num_particles = num_particles
        self.std_range = std_range
        self.mu_init_vel = mu_init_vel
        self.std_init_vel = std_init_vel
        self.turn_noise = turn_noise
        self.vel_noise = vel_noise
        self.min_weight = min_weight

    @partial(jax.jit, static_argnums=0)
    def reset(self, key, position, range_obs):
        """
        Resets particles from a single observation.
        - key: rng key
        - position: position of the observer
        - range_obs: range of the observer
        """

        def init_particle(rng):
            # Randomly sample the initial position and velocity in the range around observer
            rng_a, rng_r, rng_v, rng_o = jax.random.split(rng, 4)
            angle = jax.random.uniform(rng_a, minval=0.0, maxval=2 * jnp.pi)
            r = jax.random.uniform(rng_r) * self.std_range - self.std_range + range_obs
            vel = jax.random.normal(rng_v) * self.std_init_vel + self.mu_init_vel
            orientation = jax.random.uniform(rng_o, minval=0, maxval=2 * jnp.pi)
            return OneParticleState(
                x=position[0] + r * jnp.cos(angle),
                y=position[1] + r * jnp.sin(angle),
                theta=orientation,
                vel_x=vel * jnp.cos(orientation),
                vel_y=vel * jnp.sin(orientation),
            )

        particles = jax.vmap(init_particle)(jax.random.split(key, self.num_particles))
        weights = jnp.ones(self.num_particles)

        return ParticlesState(particles=particles, weights=weights)

    @partial(jax.jit, static_argnums=0)
    def step_and_predict(self, rng, state, pos, obs, mask):
        """
        Step of the particle filter.
        - state: ParticlesState
        - pos: positions of the observers (num_observers, x, y)
        - obs: observations (num_observers, range)
        - mask: mask for the observations (num_observers,)
        """

        key_update, key_resample = jax.random.split(rng, 2)

        # Update particles
        state = self.update_particles(key_update, state)

        # Update weights
        state = self.update_weights(state, pos, obs, mask)

        # Resample or reinit particles if the weights are too low
        reinit_cond = jnp.isnan(state.weights).any() # reinit if the max weight is too low or if there are NaNs
        state = jax.lax.cond(
            reinit_cond,
            lambda _: self.reset(
                key_update, pos[0], obs[0]
            ),  # reset with the first observation (TODO: take into account masking here)
            lambda _: self.resample(key_resample, state),
            operand=None,
        )

        # Estimate position
        pos_est = self.estimate_pos(state)

        return state, pos_est

    @partial(jax.jit, static_argnums=0)
    def update_particles(self, key, state, dt=30.0):
        """
        Updates the particles with a simple model.
        - key: rng key
        - state: ParticlesState
        - dt: time step in seconds
        """

        def update_particle(rng, particle):
            # Update particle position and velocity with noise and a simple model
            rng_t, rng_v = jax.random.split(rng, 2)
            turn = jnp.arctan2(particle.vel_y, particle.vel_x)
            orientation = (
                turn + jax.random.uniform(rng_t) * self.turn_noise * 2 - self.turn_noise
            )
            velocity = jnp.sqrt(particle.vel_x**2 + particle.vel_y**2)
            velocity = (
                velocity
                + jax.random.uniform(rng_v) * self.vel_noise * 2
                - self.vel_noise
            ).clip(0)
            forward = velocity * dt
            particle = OneParticleState(
                x=particle.x + jnp.cos(orientation) * forward,
                y=particle.y + jnp.sin(orientation) * forward,
                theta=orientation,
                vel_x=jnp.cos(orientation) * velocity,
                vel_y=jnp.sin(orientation) * velocity,
            )
            return particle

        particles = jax.vmap(update_particle)(
            jax.random.split(key, self.num_particles), state.particles
        )
        return state.replace(particles=particles)

    @partial(jax.jit, static_argnums=0)
    def update_weights(self, state, pos, obs, mask):
        """
        Updates the weights of the particles based on the observations.
        - key: rng key
        - state: ParticlesState
        - pos: positions of the observers (num_observers, x, y)
        - obs: observations (num_observers, range)
        - mask: mask for the observations (num_observers,)
        """

        def gaussian_pdf(x, mu, sigma):
            return jnp.exp(-((mu - x) ** 2) / (sigma**2) / 2.0) / jnp.sqrt(
                2.0 * jnp.pi * (sigma**2)
            )

        def get_prob(single_particle, single_pos, single_obs):
            # Compute the weight of the particle based on one observation
            dist = jnp.sqrt(
                (single_particle.x - single_pos[0]) ** 2
                + (single_particle.y - single_pos[1]) ** 2
            )
            return gaussian_pdf(dist, single_obs, self.std_range)

        def get_probs(particle):
            # Compute the weight of the particle based on all the observations
            return jax.vmap(get_prob, in_axes=(None, 0, 0))(
                particle, pos, obs
            )  # (num_observers,)

        probs = jax.vmap(get_probs)(state.particles)  # (num_particles, num_observers)
        probs = jnp.where(
            mask[np.newaxis], probs, 1.0
        )  # don't use the masks (num_particles, num_observers)
        weights = probs.prod(axis=1)  # (num_particles,)

        return state.replace(weights=weights)

    @partial(jax.jit, static_argnums=0)
    def resample(self, key, state):
        """
        Resampling of particles based on weights using systematic resampling.
        - key: rng key
        - state: ParticlesState
        """

        # Normalize weights
        weights = state.weights / state.weights.sum()

        cumulative_sum = jnp.cumsum(weights)

        # Create evenly spaced positions with a random start point
        positions = (
            jax.random.uniform(key) + jnp.arange(self.num_particles)
        ) / self.num_particles

        # Use searchsorted to find the index for each position
        indexes = jnp.searchsorted(cumulative_sum, positions, side="right")

        # Gather resampled particles based on the computed indexes
        resampled_particles = jax.tree_map(lambda x: x[indexes], state.particles)

        return ParticlesState(particles=resampled_particles, weights=state.weights)

    @partial(jax.jit, static_argnums=0)
    def estimate_pos(self, state):
        """
        Estimates the position of the observer based on the particles.
        - state: ParticlesState
        """
        p = jax.tree_map(
            lambda x: (x * state.weights).sum() / state.weights.sum(), state.particles
        )
        p = p.replace(theta=jnp.arctan2(p.vel_y, p.vel_x))
        return jnp.array([p.x, p.y])
