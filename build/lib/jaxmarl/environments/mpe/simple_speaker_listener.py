import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from gymnax.environments.spaces import Box, Discrete

SPEAKER = "speaker_0"
LISTENER = "listener_0"
AGENT_NAMES = [SPEAKER, LISTENER]

OBS_COLOURS = [(166, 38, 38), (38, 166, 38), (38, 38, 166)]

class SimpleSpeakerListenerMPE(SimpleMPE):
    def __init__(
        self,
        num_agents=2,
        num_landmarks=3,
        action_type=DISCRETE_ACT,
    ):
        assert num_agents == 2, "SimpleSpeakerListnerMPE only supports 2 agents"
        assert num_landmarks == 3, "SimpleSpeakerListnerMPE only supports 3 landmarks"

        num_entities = num_agents + num_landmarks

        dim_c = 3
        # collaborative bool .. ?

        agents = AGENT_NAMES

        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        # Action and observation spaces

        if action_type == DISCRETE_ACT:
            action_spaces = {
                SPEAKER: Discrete(3),
                LISTENER: Discrete(5),
            }
        elif action_type == CONTINUOUS_ACT:
            action_spaces = {
                SPEAKER: Box(0.0, 1.0, (3,)),
                LISTENER: Box(0.0, 1.0, (5,)),
            }
        else:
            raise NotImplementedError("Action type not implemented")

        observation_spaces = {
            SPEAKER: Box(-jnp.inf, jnp.inf, (3,)),
            LISTENER: Box(-jnp.inf, jnp.inf, (11,)),
        }

        colour = (
            [ADVERSARY_COLOUR]
            + [AGENT_COLOUR]
            + OBS_COLOURS
        )

        # Parameters
        rad = jnp.concatenate(
            [jnp.full((num_agents), 0.075), jnp.full((num_landmarks), 0.04)]
        )
        moveable = jnp.concatenate(
            [jnp.array([False]), jnp.array([True]), jnp.full((num_landmarks), False)]
        )
        silent = jnp.array([0, 1])
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

    def set_actions(self, actions: Dict):
        """Extract u and c actions for all agents from actions Dict."""

        """actions = jnp.array([actions[i] for i in self.agents]).reshape((self.num_agents, -1))"""

        return self.action_decoder(None, actions)

    def _decode_continuous_action(
        self,
        a_idx: int,
        action: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.zeros((self.num_agents, self.dim_p))
        c = jnp.zeros((self.num_agents, self.dim_c))
        c = c.at[0].set(action[SPEAKER])

        u_act = action[LISTENER]

        u_act = jnp.array([u_act[2] - u_act[1], u_act[4] - u_act[3]]) * self.accel[1]
        u = u.at[1].set(u_act)

        return u, c

    def _decode_discrete_action(
        self,
        a_idx: int,
        action: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.zeros((self.num_agents, self.dim_p))
        c = jnp.zeros((self.num_agents, self.dim_c))

        c = c.at[0, action[SPEAKER]].set(1.0)

        idx = jax.lax.select(action[LISTENER] <= 2, 0, 1)
        u_val = jax.lax.select(action[LISTENER] % 2 == 0, 1.0, -1.0) * (
            action[LISTENER] != 0
        )
        u = u.at[1, idx].set(u_val)
        u = u * self.accel[1] * self.moveable[1]
        return u, c

    def rewards(
        self,
        state: State,
    ) -> Dict[str, float]:
        r = -1 * jnp.sum(
            jnp.square(state.p_pos[1] - state.p_pos[state.goal + self.num_agents])
        )
        return {a: r for a in self.agents}

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        goal_colour = jnp.full((3,), 0.15)
        goal_colour = goal_colour.at[state.goal].set(0.65)

        dist = state.p_pos[self.num_agents :] - state.p_pos[1]
        comm = state.c[0]

        def _speaker():
            return goal_colour

        def _listener():
            return jnp.concatenate([state.p_vel[1], dist.flatten(), comm])

        return {SPEAKER: _speaker(), LISTENER: _listener()}
