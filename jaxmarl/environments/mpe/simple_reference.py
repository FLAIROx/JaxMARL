import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from gymnax.environments.spaces import Box, Discrete

# Obstacle Colours
OBS_COLOUR = [(191, 64, 64), (64, 191, 64), (64, 64, 191)]

class SimpleReferenceMPE(SimpleMPE):
    def __init__(
        self,
        num_agents=2,
        num_landmarks=3,
        local_ratio=0.5,
        action_type=DISCRETE_ACT,
    ):
        assert num_agents == 2, "SimpleReferenceMPE only supports 2 agents"
        assert num_landmarks == 3, "SimpleReferenceMPE only supports 3 landmarks"

        num_entites = num_agents + num_landmarks

        self.local_ratio = local_ratio

        dim_c = 10

        agents = ["agent_{}".format(i) for i in range(num_agents)]

        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        # Action and observation spaces
        if action_type == DISCRETE_ACT:
            action_spaces = {i: Discrete(50) for i in agents}
        elif action_type == CONTINUOUS_ACT:
            action_spaces = {i: Box(0.0, 1.0, (15,)) for i in agents}
        else:
            raise NotImplementedError("Action type not implemented")

        observation_spaces = {i: Box(-jnp.inf, jnp.inf, (21,)) for i in agents}
        colour = [AGENT_COLOUR] * num_agents + OBS_COLOUR

        silent = jnp.full((num_agents), 0)
        collide = jnp.full((num_entites), False)

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
            silent=silent,
            collide=collide,
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        key_a, key_l, key_g = jax.random.split(key, 3)

        p_pos = jnp.concatenate(
            [
                jax.random.uniform(key_a, (self.num_agents, 2), minval=-1, maxval=+1),
                jax.random.uniform(
                    key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0
                ),
            ]
        )

        g_idx = jax.random.randint(key_g, (2,), minval=0, maxval=self.num_landmarks)

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            goal=g_idx,
        )

        return self.get_obs(state), state

    def get_obs(
        self,
        state: State,
    ) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0, None))
        def _common_stats(aidx: int, state: State):
            """Values needed in all observations"""

            landmark_pos = (
                state.p_pos[self.num_agents :] - state.p_pos[aidx]
            )  # Landmark positions in agent reference frame

            return landmark_pos

        landmark_pos = _common_stats(self.agent_range, state)

        def _agent(aidx):
            other_idx = (aidx + 1) % 2
            colour = jnp.full((3,), 0.25)
            colour = colour.at[state.goal[other_idx]].set(0.75)
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 3, 2
                    colour.flatten(),  # 3
                    state.c[other_idx].flatten(),  # 10
                ]
            )

        obs = {a: _agent(i) for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_discrete_action(
        self, a_idx: int, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.zeros((self.dim_p,))
        c = jnp.zeros((self.dim_c,))
        u_act = action % 5
        c_act = action // 5
        idx = jax.lax.select(u_act <= 2, 0, 1)
        u_val = jax.lax.select(u_act % 2 == 0, 1.0, -1.0) * (u_act != 0)
        u = u.at[idx].set(u_val)
        u = u * self.accel[a_idx] * self.moveable[a_idx]
        c = c.at[c_act].set(1.0)
        return u, c

    def rewards(self, state: State) -> Dict[str, float]:
        @partial(jax.vmap, in_axes=(0, None))
        def _agent(aidx, state):
            other_idx = (aidx + 1) % 2
            return -1 * jnp.linalg.norm(
                state.p_pos[other_idx]
                - state.p_pos[self.num_agents + state.goal[other_idx]]
            )

        agent_rew = _agent(self.agent_range, state)
        global_rew = jnp.sum(agent_rew) / self.num_agents
        rew = {
            a: global_rew * (1 - self.local_ratio) + agent_rew[i] * self.local_ratio
            for i, a in enumerate(self.agents)
        }
        return rew
