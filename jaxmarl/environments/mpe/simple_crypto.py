import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from flax import struct
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from gymnax.environments.spaces import Box, Discrete

SPEAKER = "alice_0"
LISTENER = "bob_0"
ADVERSARY = "eve_0"
SPEAKER_IDX = 1
ADVERSARY_NAMES = ["eve_0"]
OBS_COLOUR = jnp.array([[255, 0, 0, 0], [0, 255, 0, 0]])


@struct.dataclass
class CryptoState(State):
    """State for the simple crypto environment."""

    goal_colour: chex.Array = None
    private_key: chex.Array = None


class SimpleCryptoMPE(SimpleMPE):
    """
    JAX Compatible version of simple_crypto_v2 PettingZoo environment.
    Source code: https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/mpe/simple_crypto/simple_crypto.py
    Note, currently only have continuous actions implemented.
    """

    def __init__(self, num_agents=3, num_landmarks=2, action_type=DISCRETE_ACT):
        assert num_agents == 3, "Simple Crypto only supports 3 agents"
        assert num_landmarks == 2, "Simple Crypto only supports 2 landmarks"

        dim_c = 4  # Communication channel dimension

        num_landmarks = num_landmarks
        num_entities = num_landmarks + num_agents

        self.num_good_agents, self.num_adversaries = 2, 1
        self.num_agents = num_agents
        self.adversaries = [ADVERSARY]
        self.good_agents = [SPEAKER, LISTENER]

        assert self.num_agents == (self.num_good_agents + self.num_adversaries)
        assert len(self.adversaries) == self.num_adversaries
        assert len(self.good_agents) == self.num_good_agents

        agents = self.adversaries + self.good_agents
        assert agents[SPEAKER_IDX] == "alice_0"

        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        # Action and observation spaces
        if action_type == DISCRETE_ACT:
            action_spaces = {i: Discrete(4) for i in agents}
        elif action_type == CONTINUOUS_ACT:
            action_spaces = {i: Box(0.0, 1.0, (4,)) for i in agents}
        else:
            raise NotImplementedError("Action type not implemented")

        observation_spaces = {i: Box(-jnp.inf, jnp.inf, (4,)) for i in self.adversaries}
        observation_spaces.update(
            {i: Box(-jnp.inf, jnp.inf, (8,)) for i in self.good_agents}
        )

        colour = (
            [ADVERSARY_COLOUR] * self.num_adversaries
            + [AGENT_COLOUR] * self.num_good_agents
            + list(OBS_COLOUR)
        )

        # Parameters
        rad = jnp.concatenate(
            [
                jnp.full((self.num_adversaries), 0.05),
                jnp.full((self.num_good_agents), 0.05),
                jnp.full((num_landmarks), 0.05),
            ]
        )
        moveable = jnp.full((num_entities), False)
        silent = jnp.full((num_agents), 0)
        collide = jnp.full((num_entities), False)

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            action_spaces=action_spaces,
            observation_spaces=observation_spaces,
            dim_c=dim_c,
            colour=colour,
            rad=rad,
            moveable=moveable,
            silent=silent,
            collide=collide,
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, CryptoState]:
        key_a, key_l, key_g, key_k = jax.random.split(key, 4)

        p_pos = jnp.concatenate(
            [
                jax.random.uniform(key_a, (self.num_agents, 2), minval=-1, maxval=+1),
                jax.random.uniform(
                    key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0
                ),
            ]
        )

        g_idx = jax.random.randint(key_g, (1,), minval=0, maxval=self.num_landmarks)
        k_idx = jax.random.randint(key_k, (1,), minval=0, maxval=self.num_landmarks)

        state = CryptoState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            goal_colour=jnp.array(
                OBS_COLOUR[g_idx], dtype=jnp.float32
            ).flatten(),  # set to float to be same as zoo env
            private_key=jnp.array(OBS_COLOUR[k_idx], dtype=jnp.float32).flatten(),
        )

        return self.get_obs(state), state

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_continuous_action(
        self, a_idx: int, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Communication action"""
        u = jnp.zeros((self.dim_p,))
        c = action
        return u, c

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_discrete_action(
        self, a_idx: int, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Communication action"""
        u = jnp.zeros((self.dim_p,))
        c = jnp.zeros((self.dim_c,))
        c = c.at[action].set(1.0)
        return u, c

    def get_obs(self, state: CryptoState) -> Dict[str, chex.Array]:
        goal_colour = state.goal_colour
        comm = state.c[SPEAKER_IDX]

        def _speaker():
            return jnp.concatenate(
                [
                    goal_colour,
                    state.private_key,
                ]
            )

        def _listener():
            return jnp.concatenate(
                [
                    state.private_key.flatten(),
                    comm,
                ]
            )

        def _adversary():
            return comm

        obs = {SPEAKER: _speaker(), LISTENER: _listener(), ADVERSARY: _adversary()}
        return obs

    def rewards(self, state: CryptoState) -> Dict[str, float]:
        comm_diff = jnp.sum(
            jnp.square(state.c - state.goal_colour), axis=1
        )  # check axis

        comm_zeros = ~jnp.all(state.c == 0)  # Ensure communication has happend

        mask = jnp.array([1, 0, -1])
        mask *= comm_zeros

        def _good():
            return jnp.sum(comm_diff * mask)

        def _adversary(idx):
            return -1 * jnp.sum(comm_diff[idx]) * comm_zeros

        rew = {a: _adversary(i) for i, a in enumerate(self.adversaries)}
        rew.update({a: _good() for i, a in enumerate(self.good_agents)})
        return rew
