import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, Tuple, Any
from collections import namedtuple, OrderedDict
import chex
from functools import partial
from flax import struct
from typing import Tuple, Optional
from enum import IntEnum
from multi_agent_env import MultiAgentEnv
from spaces import Box, Discrete
import numpy as np

# from multiagentgymnax.environments.multi_agent_env import MultiAgentEnv
# from multiagentgymnax.environments.spaces import Box, Discrete

from ma_common import (
    OBJECT_TO_INDEX,
    COLORS,
    COLOR_TO_INDEX,
    DIR_TO_VEC,
    make_maze_map)


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Toggle/activate an object
    toggle = 3
    # Done completing task
    done = 4


@struct.dataclass
class State:
    agents_pos: chex.Array
    agents_dir: chex.Array
    agents_dir_idx: chex.Array
    goals_pos: chex.Array
    last_goals: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    goals_map: chex.Array
    agents_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


@struct.dataclass
class EnvParams:
    num_agents: int
    height: int
    width: int
    n_walls: int
    n_goals: int
    agent_view_size: int
    replace_wall_pos: bool
    see_through_walls: bool
    see_agent: bool
    normalize_obs: bool
    sample_n_walls: bool
    max_episode_steps: int
    singleton_seed: int


class MAGoalCycle(MultiAgentEnv):
    def __init__(
            self,
            num_agents=2,
            height=13,
            width=13,
            n_walls=25,
            n_goals=3,
            agent_view_size=5,
            replace_wall_pos=False,
            see_through_walls=False,
            sample_n_walls=False,
            see_agent=False,
            max_episode_steps=250,
            normalize_obs=False,
            singleton_seed=-1
    ):
        super().__init__(num_agents=num_agents)

        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_range = jnp.arange(self.num_agents)

        self.obs_shape = (agent_view_size, agent_view_size, 3)
        self.action_set = jnp.array([
            Actions.left,
            Actions.right,
            Actions.forward,
            Actions.toggle,
            Actions.done
        ])
        self.observation_spaces = {i: Box(0, 255, self.obs_shape) for i in self.agents}
        self.action_spaces = {i: Discrete(len(self.action_set), dtype=jnp.uint32) for i in self.agents}

        self.params = EnvParams(
            num_agents=num_agents,
            height=height,
            width=width,
            n_walls=n_walls,
            n_goals=n_goals,
            agent_view_size=agent_view_size,
            replace_wall_pos=replace_wall_pos and not sample_n_walls,
            see_through_walls=see_through_walls,
            sample_n_walls=sample_n_walls,
            see_agent=see_agent,
            max_episode_steps=max_episode_steps,
            normalize_obs=normalize_obs,
            singleton_seed=-1,
        )

    @property
    def default_params(self) -> EnvParams:
        return self.params

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: dict
    ) -> Tuple[chex.Array, State, chex.Array, bool, dict]:
        """Perform single timestep state transition."""
        print(actions)
        a = jnp.array([self.action_set[action] for action in actions])
        state, rewards = self.step_agents(key, state, a)
        # Check game condition & no. steps for termination condition
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        # new_episode_return = state.episode_returns + reward.squeeze()
        # new_episode_length = state.episode_lengths + 1
        # state = state.replace(episode_returns=new_episode_return * (1 - done),
        #                       episode_lengths=new_episode_length * (1 - done),
        #                       returned_episode_returns=state.returned_episode_returns * (
        #                                   1 - done) + new_episode_return * done,
        #                       returned_episode_lengths=state.returned_episode_lengths * (
        #                                   1 - done) + new_episode_length * done)
        info = {}

        # info["returned_episode_returns"] = state.returned_episode_returns
        # info["returned_episode_lengths"] = state.returned_episode_lengths
        # info["returned_episode"] = done

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            rewards,
            done,
            info,
        )

    def reset_env(
            self,
            key: chex.PRNGKey,
    ) -> tuple[dict, Any]:
        """Reset environment state by resampling contents of maze_map
        - initial agent position
        - goal position
        - wall positions
        """
        params = self.params
        h = params.height
        w = params.width
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        # Reset wall map, with shape H x W, and value of 1 at (i,j) iff there is a wall at (i,j)
        key, subkey = jax.random.split(key)
        wall_idx = jax.random.choice(
            subkey, all_pos,
            shape=(params.n_walls,),
            replace=params.replace_wall_pos)

        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

        # Reset agent position + dir
        key, subkey = jax.random.split(key)
        agents_idx = jax.random.choice(subkey, all_pos, shape=(self.num_agents,),
                                      p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32))
        occupied_mask = occupied_mask.at[agents_idx].set(1)
        agents_pos = jnp.transpose(jnp.array([agents_idx % w, agents_idx // w], dtype=jnp.uint32))

        agents_mask = jnp.zeros_like(all_pos)
        agents_mask = agents_mask.at[agents_idx].set(1)
        agents_map = agents_mask.reshape(h, w).astype(jnp.bool_)

        key, subkey = jax.random.split(key)
        agents_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.uint8),
                                           shape=(self.num_agents,))
        agents_dir = DIR_TO_VEC.at[agents_dir_idx].get()

        # Reset goal position
        key, subkey = jax.random.split(key)
        goals_idx = jax.random.choice(subkey, all_pos, shape=(params.n_goals,),
                                     p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32))
        goals_pos = jnp.array([goals_idx % w, goals_idx // w], dtype=jnp.uint32)

        goals_mask = jnp.zeros_like(all_pos)
        goals_mask = goals_mask.at[goals_idx].set(1)
        goals_map = goals_mask.reshape(h, w).astype(jnp.bool_)

        key, subkey = jax.random.split(key)
        last_goals = jax.random.choice(subkey, jnp.arange(params.n_goals, dtype=jnp.uint8),
                                           shape=(self.num_agents,))

        maze_map = make_maze_map(
            params,
            wall_map,
            goals_pos,
            agents_pos,
            agents_dir_idx,
            pad_obs=True)

        state = State(
            agents_pos=agents_pos,
            agents_dir=agents_dir,
            agents_dir_idx=agents_dir_idx,
            goals_pos=goals_pos,
            last_goals=last_goals,
            wall_map=wall_map.astype(jnp.bool_),
            agents_map=agents_map.astype(jnp.bool_),
            goals_map=goals_map.astype(jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
            episode_returns=0,
            episode_lengths=0,
            returned_episode_returns=0,
            returned_episode_lengths=0,
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict:

        # @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> chex.Array:
            """Return limited grid view ahead of agent."""
            obs = jnp.zeros(self.obs_shape, dtype=jnp.uint8)

            obs_fwd_bound1 = state.agents_pos[aidx]
            obs_fwd_bound2 = state.agents_pos[aidx] + state.agents_dir[aidx] * (self.obs_shape[0] - 1)

            side_offset = self.obs_shape[0] // 2
            obs_side_bound1 = state.agents_pos[aidx] + (state.agents_dir[aidx] == 0) * side_offset
            obs_side_bound2 = state.agents_pos[aidx] - (state.agents_dir[aidx] == 0) * side_offset

            all_bounds = jnp.stack([obs_fwd_bound1, obs_fwd_bound2, obs_side_bound1, obs_side_bound2])

            # Clip obs to grid bounds appropriately
            padding = obs.shape[0] - 1
            obs_bounds_min = np.min(all_bounds, 0) + padding
            obs_range_x = jnp.arange(obs.shape[0]) + obs_bounds_min[1]
            obs_range_y = jnp.arange(obs.shape[0]) + obs_bounds_min[0]

            meshgrid = jnp.meshgrid(obs_range_y, obs_range_x)
            coord_y = meshgrid[1].flatten()
            coord_x = meshgrid[0].flatten()

            obs = state.maze_map.at[
                  coord_y, coord_x, :].get().reshape(obs.shape[0], obs.shape[1], 3)

            obs = (state.agents_dir_idx[aidx] == 0) * jnp.rot90(obs, 1) + \
                  (state.agents_dir_idx[aidx] == 1) * jnp.rot90(obs, 2) + \
                  (state.agents_dir_idx[aidx] == 2) * jnp.rot90(obs, 3) + \
                  (state.agents_dir_idx[aidx] == 3) * jnp.rot90(obs, 4)

            if not self.params.see_agent:
                obs = obs.at[-1, side_offset].set(
                    jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
                )

            image = obs.astype(jnp.uint8)
            if self.params.normalize_obs:
                image = image / 10.0

            obs_dict = dict(
                image=image,
                agent_dir=state.agents_dir_idx[aidx]
            )

            # return OrderedDict(obs_dict)

            return image

        obs = jnp.array([_observation(aidx, state) for aidx in range(self.num_agents)])
        return {a: obs[i] for i, a in enumerate(self.agents)}
        # obs = _observation(0, state)
        # return obs

    def step_agents(self, key: chex.PRNGKey, state: State, actions: chex.Array) -> Tuple[State, chex.Array]:
        params = self.params

        def _step(aidx, carry):
            actions, rewards, state = carry
            action = actions[aidx]
            fwd = (action == Actions.forward)
            # Update agent position (forward action)
            fwd_pos = jnp.minimum(
                jnp.maximum(state.agents_pos[aidx] + (action == Actions.forward) * state.agents_dir[aidx], 0),
                jnp.array((params.width - 1, params.height - 1), dtype=jnp.uint32))

            # Can't go past wall or goal
            fwd_pos_has_wall = state.wall_map.at[fwd_pos[1], fwd_pos[0]].get()
            fwd_pos_has_goal = state.goals_map.at[fwd_pos[1], fwd_pos[0]].get()
            fwd_pos_has_agent = state.agents_map.at[fwd_pos[1], fwd_pos[0]].get()

            last_idx = state.last_goals[aidx]
            next_idx = last_idx + 1
            target_idx = state.goals_pos[:, next_idx % params.n_goals]
            hit_target = jnp.logical_and((fwd_pos[1] == target_idx[1]), (fwd_pos[0] == target_idx[0]))
            last_goals = state.last_goals
            last_goals = last_goals.at[aidx].set((last_idx + hit_target) % params.n_goals)

            fwd_pos_blocked = jnp.logical_or(jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal), fwd_pos_has_agent)

            agent_pos_prev = jnp.array(state.agents_pos[aidx])
            agent_pos = (fwd_pos_blocked * state.agents_pos[aidx] + (~fwd_pos_blocked) * fwd_pos).astype(jnp.uint32)
            agents_pos = state.agents_pos
            agents_pos = agents_pos.at[aidx].set(agent_pos)

            # Update agent direction (left_turn or right_turn action)
            agent_dir_offset = \
                0 \
                + (action == Actions.left) * (-1) \
                + (action == Actions.right) * 1

            agent_dir_idx = (state.agents_dir_idx[aidx] + agent_dir_offset) % 4
            agents_dir_idx = state.agents_dir_idx
            agents_dir_idx = agents_dir_idx.at[aidx].set(agent_dir_idx)
            agent_dir = DIR_TO_VEC[agents_dir_idx[aidx]]
            agents_dir = state.agents_dir
            agents_dir = agents_dir.at[aidx].set(agent_dir)

            # Update agent component in maze_map
            empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
            agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], agent_dir_idx], dtype=jnp.uint8)
            padding = self.obs_shape[0] - 1
            maze_map = state.maze_map
            maze_map = maze_map.at[padding + agent_pos_prev[1], padding + agent_pos_prev[0], :].set(empty)
            maze_map = maze_map.at[padding + agent_pos[1], padding + agent_pos[0], :].set(agent)
            agents_map = state.agents_map
            agents_map = agents_map.at[agent_pos_prev[1], agent_pos_prev[0]].set(0)
            agents_map = agents_map.at[agent_pos[1], agent_pos[0]].set(1)

            reward = hit_target
            rewards = rewards.at[aidx].set(reward)

            return (actions, rewards, state.replace(
                                        agents_pos=agents_pos,
                                        agents_dir_idx=agents_dir_idx,
                                        agents_dir=agents_dir,
                                        agents_map=agents_map,
                                        maze_map=maze_map,
                                        last_goals=last_goals))

        rewards = jnp.zeros(self.num_agents)
        actions, rewards, state = jax.lax.fori_loop(0, self.num_agents, _step, (actions, rewards, state))

        return (state, rewards)

    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.params.max_episode_steps
        return jnp.logical_or(done_steps, state.terminal)

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats['return'] > 0

        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return "Multi-Agent Goal Cycle"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def observation_space(self, agent: str):
        """ Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """ Action space for a given agent."""
        return self.action_spaces[agent]