import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from jaxmarl.environments.mpe.simple import State, SimpleMPE
from jaxmarl.environments.mpe.default_params import *
from gymnax.environments.spaces import Box


class SimpleAdversaryMPE(SimpleMPE):
    def __init__(
        self,
        num_good_agents=2,
        num_adversaries=1,
        num_obs=2,
        action_type=DISCRETE_ACT,
    ):
        dim_c = 2  # NOTE follows code rather than docs

        num_agents = num_good_agents + num_adversaries
        num_landmarks = num_obs
        num_entities = num_landmarks + num_agents

        self.num_good_agents, self.num_adversaries = num_good_agents, num_adversaries

        self.adversaries = ["adversary_{}".format(i) for i in range(num_adversaries)]
        self.good_agents = ["agent_{}".format(i) for i in range(num_good_agents)]
        agents = self.adversaries + self.good_agents

        landmarks = ["landmark {}".format(i) for i in range(num_obs)]

        # Action and observation spaces
        observation_spaces = {i: Box(-jnp.inf, jnp.inf, (8,)) for i in self.adversaries}
        observation_spaces.update(
            {i: Box(-jnp.inf, jnp.inf, (10,)) for i in self.good_agents}
        )

        colour = (
            [ADVERSARY_COLOUR] * num_adversaries
            + [AGENT_COLOUR] * num_good_agents
            + [OBS_COLOUR] * num_obs
        )

        # Parameters
        rad = jnp.concatenate(
            [jnp.full((num_agents), 0.15), jnp.full((num_landmarks), 0.08)]
        )
        collide = jnp.full((num_entities), False)

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=dim_c,
            colour=colour,
            rad=rad,
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

        g_idx = jax.random.randint(key_g, (), minval=0, maxval=self.num_landmarks)

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            goal=g_idx,
        )

        return self.get_obs(state), state

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0, None))
        def _common_stats(aidx, state: State):
            """Values needed in all observations"""

            landmark_pos = (
                state.p_pos[self.num_agents :] - state.p_pos[aidx]
            )  # Landmark positions in agent reference frame

            # Zero out unseen agents with other_mask
            other_pos = state.p_pos[: self.num_agents] - state.p_pos[aidx]

            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]

            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)

            return landmark_pos, other_pos

        landmark_pos, other_pos = _common_stats(self.agent_range, state)

        def _good(aidx):
            goal_rel_pos = state.p_pos[state.goal + self.num_agents] - state.p_pos[aidx]

            return jnp.concatenate(
                [
                    goal_rel_pos.flatten(),
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                ]
            )

        def _adversary(aidx):
            return jnp.concatenate(
                [
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                ]
            )

        obs = {a: _adversary(i) for i, a in enumerate(self.adversaries)}
        obs.update(
            {a: _good(i + self.num_adversaries) for i, a in enumerate(self.good_agents)}
        )
        return obs

    def rewards(
        self,
        state: State,
    ) -> Dict[str, float]:
        adv_rew = jnp.sum(
            jnp.linalg.norm(
                state.p_pos[: self.num_adversaries]
                - state.p_pos[state.goal + self.num_agents],
                axis=1,
            )
        )
        pos_rew = -1 * jnp.min(
            jnp.linalg.norm(
                state.p_pos[self.num_adversaries : self.num_agents]
                - state.p_pos[state.goal + self.num_agents],
                axis=1,
            )
        )
        good_rew = adv_rew + pos_rew

        def _ad(aidx):
            return -1 * jnp.linalg.norm(
                state.p_pos[aidx] - state.p_pos[state.goal + self.num_agents]
            )

        rew = {a: _ad(i) for i, a in enumerate(self.adversaries)}
        rew.update({a: good_rew for a in self.good_agents})
        return rew
