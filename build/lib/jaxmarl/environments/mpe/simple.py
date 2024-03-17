""" 
Base class for MPE PettingZoo envs.

TODO: viz for communication env, e.g. crypto
"""

import jax
import jax.numpy as jnp
import numpy as onp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.mpe.default_params import *
import chex
from gymnax.environments.spaces import Box, Discrete
from flax import struct
from typing import Tuple, Optional, Dict
from functools import partial

import matplotlib.pyplot as plt
import matplotlib

@struct.dataclass
class State:
    """Basic MPE State"""

    p_pos: chex.Array  # [num_entities, [x, y]]
    p_vel: chex.Array  # [n, [x, y]]
    c: chex.Array  # communication state [num_agents, [dim_c]]
    done: chex.Array  # bool [num_agents, ]
    step: int  # current step
    goal: int = None  # index of target landmark, used in: SimpleSpeakerListenerMPE, SimpleReferenceMPE, SimplePushMPE, SimpleAdversaryMPE


class SimpleMPE(MultiAgentEnv):
    def __init__(
        self,
        num_agents=1,
        action_type=DISCRETE_ACT,
        agents=None,
        num_landmarks=1,
        landmarks=None,
        action_spaces=None,
        observation_spaces=None,
        colour=None,
        dim_c=0,
        dim_p=2,
        max_steps=MAX_STEPS,
        dt=DT,
        **kwargs,
    ):
        # Agent and entity constants
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_entities = num_agents + num_landmarks
        self.agent_range = jnp.arange(num_agents)
        self.entity_range = jnp.arange(self.num_entities)

        # Setting, and sense checking, entity names and agent action spaces
        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert (
                len(agents) == num_agents
            ), f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}
        self.classes = self.create_agent_classes()

        if landmarks is None:
            self.landmarks = [f"landmark {i}" for i in range(num_landmarks)]
        else:
            assert (
                len(landmarks) == num_landmarks
            ), f"Number of landmarks {len(landmarks)} does not match number of landmarks {num_landmarks}"
            self.landmarks = landmarks
        self.l_to_i = {l: i + self.num_agents for i, l in enumerate(self.landmarks)}

        if action_spaces is None:
            if action_type == DISCRETE_ACT:
                self.action_spaces = {i: Discrete(5) for i in self.agents}
            elif action_type == CONTINUOUS_ACT:
                self.action_spaces = {i: Box(0.0, 1.0, (5,)) for i in self.agents}
        else:
            assert (
                len(action_spaces.keys()) == num_agents
            ), f"Number of action spaces {len(action_spaces.keys())} does not match number of agents {num_agents}"
            self.action_spaces = action_spaces

        if observation_spaces is None:
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (4,)) for i in self.agents
            }
        else:
            assert (
                len(observation_spaces.keys()) == num_agents
            ), f"Number of observation spaces {len(observation_spaces.keys())} does not match number of agents {num_agents}"
            self.observation_spaces = observation_spaces

        self.colour = (
            colour
            if colour is not None
            else [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks
        )

        # Action type
        if action_type == DISCRETE_ACT:
            self.action_decoder = self._decode_discrete_action
        elif action_type == CONTINUOUS_ACT:
            self.action_decoder = self._decode_continuous_action
        else:
            raise NotImplementedError(f"Action type: {action_type} is not supported")

        # World dimensions
        self.dim_c = dim_c  # communication channel dimensionality
        self.dim_p = dim_p  # position dimensionality

        # Environment parameters
        self.max_steps = max_steps
        self.dt = dt
        if "rad" in kwargs:
            self.rad = kwargs["rad"]
            assert (
                len(self.rad) == self.num_entities
            ), f"Rad array length {len(self.rad)} does not match number of entities {self.num_entities}"
            assert jnp.all(self.rad > 0), f"Rad array must be positive, got {self.rad}"
        else:
            self.rad = jnp.concatenate(
                [jnp.full((self.num_agents), 0.15), jnp.full((self.num_landmarks), 0.2)]
            )

        if "moveable" in kwargs:
            self.moveable = kwargs["moveable"]
            assert (
                len(self.moveable) == self.num_entities
            ), f"Moveable array length {len(self.moveable)} does not match number of entities {self.num_entities}"
            assert (
                self.moveable.dtype == bool
            ), f"Moveable array must be boolean, got {self.moveable}"
        else:
            self.moveable = jnp.concatenate(
                [
                    jnp.full((self.num_agents), True),
                    jnp.full((self.num_landmarks), False),
                ]
            )

        if "silent" in kwargs:
            self.silent = kwargs["silent"]
            assert (
                len(self.silent) == self.num_agents
            ), f"Silent array length {len(self.silent)} does not match number of agents {self.num_agents}"
        else:
            self.silent = jnp.full((self.num_agents), 1)

        if "collide" in kwargs:
            self.collide = kwargs["collide"]
            assert (
                len(self.collide) == self.num_entities
            ), f"Collide array length {len(self.collide)} does not match number of entities {self.num_entities}"
        else:
            self.collide = jnp.full((self.num_entities), False)

        if "mass" in kwargs:
            self.mass = kwargs["mass"]
            assert (
                len(self.mass) == self.num_entities
            ), f"Mass array length {len(self.mass)} does not match number of entities {self.num_entities}"
            assert jnp.all(
                self.mass > 0
            ), f"Mass array must be positive, got {self.mass}"
        else:
            self.mass = jnp.full((self.num_entities), 1.0)

        if "accel" in kwargs:
            self.accel = kwargs["accel"]
            assert (
                len(self.accel) == self.num_agents
            ), f"Accel array length {len(self.accel)} does not match number of agents {self.num_agents}"
            assert jnp.all(
                self.accel > 0
            ), f"Accel array must be positive, got {self.accel}"
        else:
            self.accel = jnp.full((self.num_agents), 5.0)

        if "max_speed" in kwargs:
            self.max_speed = kwargs["max_speed"]
            assert (
                len(self.max_speed) == self.num_entities
            ), f"Max speed array length {len(self.max_speed)} does not match number of entities {self.num_entities}"
        else:
            self.max_speed = jnp.concatenate(
                [jnp.full((self.num_agents), -1), jnp.full((self.num_landmarks), 0.0)]
            )

        if "u_noise" in kwargs:
            self.u_noise = kwargs["u_noise"]
            assert (
                len(self.u_noise) == self.num_agents
            ), f"U noise array length {len(self.u_noise)} does not match number of agents {self.num_agents}"
        else:
            self.u_noise = jnp.full((self.num_agents), 0)

        if "c_noise" in kwargs:
            self.c_noise = kwargs["c_noise"]
            assert (
                len(self.c_noise) == self.num_agents
            ), f"C noise array length {len(self.c_noise)} does not match number of agents {self.num_agents}"
        else:
            self.c_noise = jnp.full((self.num_agents), 0)

        if "damping" in kwargs:
            self.damping = kwargs["damping"]
            assert (
                self.damping >= 0
            ), f"Damping must be non-negative, got {self.damping}"
        else:
            self.damping = DAMPING

        if "contact_force" in kwargs:
            self.contact_force = kwargs["contact_force"]
        else:
            self.contact_force = CONTACT_FORCE

        if "contact_margin" in kwargs:
            self.contact_margin = kwargs["contact_margin"]
        else:
            self.contact_margin = CONTACT_MARGIN

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        u, c = self.set_actions(actions)
        if (
            c.shape[1] < self.dim_c
        ):  # This is due to the MPE code carrying around 0s for the communication channels
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1
            )

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, self.c_noise, self.silent)
        done = jnp.full((self.num_agents), state.step >= self.max_steps)

        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)

        obs = self.get_obs(state)

        info = {}

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Initialise with random positions"""

        key_a, key_l = jax.random.split(key)

        p_pos = jnp.concatenate(
            [
                jax.random.uniform(
                    key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
                ),
                jax.random.uniform(
                    key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0
                ),
            ]
        )

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return dictionary of agent observations"""

        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> jnp.ndarray:
            """Return observation for agent i."""
            landmark_rel_pos = state.p_pos[self.num_agents :] - state.p_pos[aidx]

            return jnp.concatenate(
                [state.p_vel[aidx].flatten(), landmark_rel_pos.flatten()]
            )

        obs = _observation(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    def rewards(self, state: State) -> Dict[str, float]:
        """Assign rewards for all agents"""

        @partial(jax.vmap, in_axes=[0, None])
        def _reward(aidx: int, state: State):
            return -1 * jnp.sum(
                jnp.square(state.p_pos[aidx] - state.p_pos[self.num_agents :])
            )

        r = _reward(self.agent_range, state)
        return {agent: r[i] for i, agent in enumerate(self.agents)}

    def set_actions(self, actions: Dict):
        """Extract u and c actions for all agents from actions Dict."""

        actions = jnp.array([actions[i] for i in self.agents]).reshape(
            (self.num_agents, -1)
        )

        return self.action_decoder(self.agent_range, actions)

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_continuous_action(
        self, a_idx: int, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.array([action[2] - action[1], action[4] - action[3]])
        u = u * self.accel[a_idx] * self.moveable[a_idx]
        c = action[5:]
        return u, c

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_discrete_action(
        self, a_idx: int, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.zeros((self.dim_p,))
        idx = jax.lax.select(action <= 2, 0, 1)
        u_val = jax.lax.select(action % 2 == 0, 1.0, -1.0) * (action != 0)
        u = u.at[idx].set(u_val)
        u = u * self.accel[a_idx] * self.moveable[a_idx]
        return u, jnp.zeros((self.dim_c,))

    def _world_step(self, key: chex.PRNGKey, state: State, u: chex.Array):
        p_force = jnp.zeros((self.num_agents, 2))

        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_agents)
        p_force = self._apply_action_force(
            key_noise, p_force, u, self.u_noise, self.moveable[: self.num_agents]
        )
        # jax.debug.print('jax p_force post agent {p_force}', p_force=p_force)

        # apply environment forces
        p_force = jnp.concatenate([p_force, jnp.zeros((self.num_landmarks, 2))])
        p_force = self._apply_environment_force(p_force, state)
        # print('p_force post apply env force', p_force)
        # jax.debug.print('jax p_force final: {p_force}', p_force=p_force)

        # integrate physical state
        p_pos, p_vel = self._integrate_state(
            p_force, state.p_pos, state.p_vel, self.mass, self.moveable, self.max_speed
        )

        return p_pos, p_vel

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def _apply_comm_action(
        self, key: chex.PRNGKey, c: chex.Array, c_noise: int, silent: int
    ) -> chex.Array:
        silence = jnp.zeros(c.shape)
        noise = jax.random.normal(key, shape=c.shape) * c_noise
        return jax.lax.select(silent, silence, c + noise)

    # gather agent action forces
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def _apply_action_force(
        self,
        key: chex.PRNGKey,
        p_force: chex.Array,
        u: chex.Array,
        u_noise: int,
        moveable: bool,
    ):
        noise = jax.random.normal(key, shape=u.shape) * u_noise
        return jax.lax.select(moveable, u + noise, p_force)

    def _apply_environment_force(self, p_force_all: chex.Array, state: State):
        """gather physical forces acting on entities"""

        @partial(jax.vmap, in_axes=[0])
        def __env_force_outer(idx: int):
            @partial(jax.vmap, in_axes=[None, 0])
            def __env_force_inner(idx_a: int, idx_b: int):
                l = idx_b <= idx_a
                l_a = jnp.zeros((2, 2))

                collision_force = self._get_collision_force(idx_a, idx_b, state)

                xx = jax.lax.select(l, l_a, collision_force)
                # jax.debug.print('{a} {b} {f}', a=idx_a, b=idx_b, f=xx)
                return xx

            p_force_t = __env_force_inner(idx, self.entity_range)

            p_force_a = jnp.sum(p_force_t[:, 0], axis=0)  # ego force from other agents
            p_force_o = p_force_t[:, 1]
            p_force_o = p_force_o.at[idx].set(p_force_a)

            return p_force_o

        p_forces = __env_force_outer(self.entity_range)
        p_forces = jnp.sum(p_forces, axis=0)

        return p_forces + p_force_all

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0, 0])
    def _integrate_state(self, p_force, p_pos, p_vel, mass, moveable, max_speed):
        """integrate physical state"""

        p_pos += p_vel * self.dt
        p_vel = p_vel * (1 - self.damping)

        p_vel += (p_force / mass) * self.dt * moveable

        speed = jnp.sqrt(jnp.square(p_vel[0]) + jnp.square(p_vel[1]))
        over_max = (
            p_vel / jnp.sqrt(jnp.square(p_vel[0]) + jnp.square(p_vel[1])) * max_speed
        )

        p_vel = jax.lax.select((speed > max_speed) & (max_speed >= 0), over_max, p_vel)

        return p_pos, p_vel

    # get collision forces for any contact between two entities BUG
    def _get_collision_force(self, idx_a: int, idx_b: int, state: State):
        dist_min = self.rad[idx_a] + self.rad[idx_b]
        delta_pos = state.p_pos[idx_a] - state.p_pos[idx_b]

        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))

        # softmax penetration
        k = self.contact_margin
        penetration = jnp.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force * self.moveable[idx_a]
        force_b = -force * self.moveable[idx_b]
        force = jnp.array([force_a, force_b])

        c = (~self.collide[idx_a]) | (~self.collide[idx_b]) | (idx_a == idx_b)
        c_force = jnp.zeros((2, 2))
        return jax.lax.select(c, c_force, force)

    def create_agent_classes(self):
        if hasattr(self, "leader"):
            return {
                "leadadversary": self.leader,
                "adversaries": self.adversaries,
                "agents": self.good_agents,
            }
        elif hasattr(self, "adversaries"):
            return {
                "adversaries": self.adversaries,
                "agents": self.good_agents,
            }
        else:
            return {
                "agents": self.agents,
            }

    def agent_classes(self) -> Dict[str, list]:
        return self.classes

    ### === UTILITIES === ###
    def is_collision(self, a: int, b: int, state: State):
        """check if two entities are colliding"""
        dist_min = self.rad[a] + self.rad[b]
        delta_pos = state.p_pos[a] - state.p_pos[b]
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        return (dist < dist_min) & (self.collide[a] & self.collide[b]) & (a != b)

    @partial(jax.vmap, in_axes=(None, 0))
    def map_bounds_reward(self, x: float):
        """vmap over x, y coodinates"""
        w = x < 0.9
        m = x < 1.0
        mr = (x - 0.9) * 10
        br = jnp.min(jnp.array([jnp.exp(2 * x - 2), 10]))
        return jax.lax.select(m, mr, br) * ~w   


if __name__ == "__main__":
    from jaxmarl.environments.mpe import MPEVisualizer

    num_agents = 3
    key = jax.random.PRNGKey(0)

    env = SimpleMPE(num_agents)

    obs, state = env.reset(key)

    mock_action = jnp.array([[1.0, 1.0, 0.1, 0.1, 0.0]])

    actions = jnp.repeat(mock_action[None], repeats=num_agents, axis=0).squeeze()

    actions = {agent: mock_action for agent in env.agents}
    a = env.agents
    a.reverse()
    print("a", a)
    actions = {agent: mock_action for agent in a}
    print("actions", actions)

    # env.enable_render()

    state_seq = []
    print("state", state)
    print("action spaces", env.action_spaces)

    for _ in range(25):
        state_seq.append(state)
        key, key_act = jax.random.split(key)
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        obs, state, rew, dones, _ = env.step_env(key, state, actions)

    viz = MPEVisualizer(env, state_seq)
    viz.animate(None, view=True)
