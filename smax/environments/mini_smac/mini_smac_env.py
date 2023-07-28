import jax.numpy as jnp
import jax
from smax.environments.multi_agent_env import MultiAgentEnv
from smax.environments.spaces import Box, Discrete
import chex
from typing import Tuple, Dict, Optional
from flax.struct import dataclass
from enum import IntEnum
from functools import partial


@dataclass
class State:
    unit_positions: chex.Array
    unit_alive: chex.Array
    unit_teams: chex.Array
    unit_health: chex.Array
    unit_types: chex.Array
    prev_actions: chex.Array
    time: int
    terminal: bool


@dataclass
class EnvParams:
    num_agents_per_team: int
    map_width: float
    map_height: float
    world_steps_per_env_step: int
    time_per_step: float
    unit_velocity: float
    unit_type_attacks: chex.Array
    unit_type_attack_ranges: chex.Array
    unit_type_sight_ranges: chex.Array
    unit_type_radiuses: chex.Array
    time_per_step: float
    max_steps: int
    won_battle_bonus: float


@dataclass
class WorldDelta:
    """Encapsulates the effect of an agent's action"""

    pos: chex.Array
    attacked_idx: int
    health_diff: float


class MiniSMAC(MultiAgentEnv):
    def __init__(
        self,
        num_agents_per_team=5,
        world_steps_per_env_step=8,
    ) -> None:
        self.num_agents_per_team = num_agents_per_team
        self.num_agents = num_agents_per_team * 2
        self.num_movement_actions = 4
        self.world_steps_per_env_step = world_steps_per_env_step
        self.agents = [f"ally_{i}" for i in range(self.num_agents_per_team)] + [
            f"enemy_{i}" for i in range(self.num_agents_per_team)
        ]
        self.agent_ids = {agent: i for i, agent in enumerate(self.agents)}
        self.teams = jnp.zeros((self.num_agents,), dtype=jnp.uint8)
        self.teams = self.teams.at[self.num_agents_per_team :].set(1)
        self.own_features = ["health", "position_x", "position_y"]
        self.unit_features = ["health", "position_x", "position_y", "last_action"]
        self.obs_size = (
            len(self.unit_features) * (self.num_agents_per_team - 1)
            + len(self.unit_features) * self.num_agents_per_team
            + len(self.own_features)
        )
        self.state_size = (len(self.unit_features) + 2) * self.num_agents
        self.observation_spaces = {
            i: Box(low=0.0, high=1.0, shape=(self.obs_size,)) for i in self.agents
        }
        self.num_actions = self.num_agents_per_team + self.num_movement_actions
        self.action_spaces = {
            i: Discrete(num_categories=self.num_actions) for i in self.agents
        }

    @property
    def default_params(self):
        return EnvParams(
            num_agents_per_team=5,
            map_width=32,
            map_height=32,
            world_steps_per_env_step=8,
            unit_velocity=5.0,
            unit_type_attacks=jnp.array([0.02]),
            time_per_step=1.0 / 16,
            won_battle_bonus=1,
            unit_type_attack_ranges=jnp.array([3.0]),
            unit_type_sight_ranges=jnp.array([4.0]),
            unit_type_radiuses=jnp.array([0.5]),
            max_steps=100,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Environment-specific reset."""
        key, team_0_key, team_1_key = jax.random.split(key, num=3)
        team_0_start = jnp.stack([jnp.array([8.0, 16.0])] * self.num_agents_per_team)
        team_0_start_noise = jax.random.uniform(
            team_0_key, shape=(self.num_agents_per_team, 2), minval=-2, maxval=2
        )
        team_0_start = team_0_start + team_0_start_noise
        team_1_start = jnp.stack([jnp.array([24.0, 16.0])] * self.num_agents_per_team)
        team_1_start_noise = jax.random.uniform(
            team_1_key, shape=(self.num_agents_per_team, 2), minval=-2, maxval=2
        )
        team_1_start = team_1_start + team_1_start_noise
        unit_positions = jnp.concatenate([team_0_start, team_1_start])
        unit_teams = jnp.zeros((self.num_agents,))
        unit_teams = unit_teams.at[self.num_agents_per_team :].set(1)
        state = State(
            unit_positions=unit_positions,
            unit_alive=jnp.ones((self.num_agents,), dtype=jnp.bool_),
            unit_teams=unit_teams,
            unit_health=jnp.ones((self.num_agents,)),
            unit_types=jnp.zeros(
                (self.num_agents,), dtype=jnp.uint8
            ),  # only one unit type for now
            prev_actions=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            time=0,
            terminal=False,
        )
        state = self._push_units_away(state, params)
        obs = self.get_obs(state, params)
        world_state = self.get_world_state(state, params)
        obs["world_state"] = jax.lax.stop_gradient(world_state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        params: EnvParams,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        actions = jnp.array([actions[i] for i in self.agents])
        health_before = jnp.copy(state.unit_health)

        def world_step_fn(carry, _):
            carry = partial(self._world_step, actions=actions, params=params)(carry)
            carry = self._update_dead_agents(carry)
            carry = self._push_units_away(carry, params)
            return carry, None

        state, _ = jax.lax.scan(
            world_step_fn, init=state, xs=None, length=self.world_steps_per_env_step
        )
        health_after = state.unit_health
        state = state.replace(
            terminal=self.is_terminal(state, params), prev_actions=actions
        )
        obs = self.get_obs(state, params)
        dones = {
            agent: ~state.unit_alive[self.agent_ids[agent]] for agent in self.agents
        }
        rewards = self.compute_reward(state, params, health_before, health_after)
        dones["__all__"] = state.terminal
        world_state = self.get_world_state(state, params)
        infos = {}
        obs["world_state"] = jax.lax.stop_gradient(world_state)
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            rewards,
            dones,
            infos,
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_reward(self, state, params, health_before, health_after):
        def compute_team_reward(team_idx):
            # compute how much the enemy team health has decreased
            other_team_idx = jnp.logical_not(team_idx).astype(jnp.uint32)
            other_team_start_idx = jnp.array([0, self.num_agents_per_team])[
                other_team_idx
            ]
            enemy_health_decrease = jnp.sum(
                jax.lax.dynamic_slice_in_dim(
                    health_after - health_before,
                    other_team_start_idx,
                    self.num_agents_per_team,
                )
            )
            enemy_health_decrease_reward = (
                jnp.abs(enemy_health_decrease) / self.num_agents_per_team
            )
            won_battle = jnp.all(
                jnp.logical_not(
                    jax.lax.dynamic_slice_in_dim(
                        state.unit_alive, other_team_start_idx, self.num_agents_per_team
                    )
                )
            )
            won_battle_bonus = jax.lax.cond(
                won_battle, lambda: params.won_battle_bonus, lambda: 0
            )
            return enemy_health_decrease_reward + won_battle_bonus

        # agents still get reward when they are dead to allow for noble sacrifice
        team_rewards = jax.vmap(compute_team_reward)(jnp.arange(2))
        return {
            agent: team_rewards[self.agent_ids[agent] // self.num_agents_per_team]
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state, params):
        all_dead = jnp.all(
            jnp.logical_not(state.unit_alive[: self.num_agents_per_team])
        )
        all_enemy_dead = jnp.all(
            jnp.logical_not(state.unit_alive[self.num_agents_per_team :])
        )
        over_time_limit = state.time >= params.max_steps
        return all_dead | all_enemy_dead | over_time_limit

    def _update_dead_agents(
        self,
        state: State,
    ):
        unit_alive = state.unit_health > 0
        return state.replace(unit_alive=unit_alive)

    def _push_units_away(self, state: State, params: EnvParams, firmness: float = 1.0):
        delta_matrix = state.unit_positions[:, None] - state.unit_positions[None, :]
        dist_matrix = (
            jnp.linalg.norm(delta_matrix, axis=-1)
            + jnp.identity(self.num_agents)
            + 1e-6
        )
        radius_matrix = (
            params.unit_type_radiuses[state.unit_types][:, None]
            + params.unit_type_radiuses[state.unit_types][None, :]
        )
        overlap_term = jax.nn.relu(radius_matrix / dist_matrix - 1.0)
        unit_positions = (
            state.unit_positions
            + firmness * jnp.sum(delta_matrix * overlap_term[:, :, None], axis=1) / 2
        )
        return state.replace(unit_positions=unit_positions)

    @partial(jax.jit, static_argnums=(0,))
    def _world_step(
        self,
        state: State,
        actions: Dict[str, chex.Array],
        params: EnvParams,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        def update_position(idx, action):
            # Compute the movements slightly strangely.
            # The velocities below are for diagonal directions
            # because these are easier to encode as actions than the four
            # diagonal directions. Then rotate the velocity 45
            # degrees anticlockwise to compute the movement.
            pos = state.unit_positions[idx]
            vec = jax.lax.cond(
                action > self.num_movement_actions - 1,
                lambda: jnp.zeros((2,)),
                lambda: jnp.array(
                    [
                        (-1) ** (action // 2) * (1.0 / jnp.sqrt(2)),
                        (-1) ** (action // 2 + action % 2) * (1.0 / jnp.sqrt(2)),
                    ]
                ),
            )
            rotation = jnp.array(
                [
                    [1.0 / jnp.sqrt(2), -1.0 / jnp.sqrt(2)],
                    [1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)],
                ]
            )
            vec = rotation @ vec
            new_pos = pos + vec * params.unit_velocity * params.time_per_step
            # avoid going out of bounds
            new_pos = jnp.maximum(
                jnp.minimum(new_pos, jnp.array([params.map_width, params.map_height])),
                jnp.zeros((2,)),
            )
            return WorldDelta(new_pos, idx, 0.0)

        def update_agent_health(idx, action):
            # for team 1, their attack actions are labelled in
            # reverse order because that is the order they are
            # observed in
            attacked_idx = jax.lax.cond(
                idx < self.num_agents_per_team,
                lambda: action + self.num_agents_per_team - self.num_movement_actions,
                lambda: self.num_agents_per_team
                - 1
                - (action - self.num_movement_actions),
            )
            attack_valid = (
                (
                    jnp.linalg.norm(
                        state.unit_positions[idx] - state.unit_positions[attacked_idx]
                    )
                    < params.unit_type_attack_ranges[state.unit_types[idx]]
                )
                & state.unit_alive[idx]
                & state.unit_alive[attacked_idx]
            )
            health_diff = jax.lax.select(
                attack_valid,
                -params.unit_type_attacks[state.unit_types[idx]],
                0.0,
            )
            return WorldDelta(state.unit_positions[idx], attacked_idx, health_diff)

        def perform_agent_action(idx, action):
            return jax.lax.cond(
                actions[idx] > self.num_movement_actions - 1,
                lambda: update_agent_health(idx, action),
                lambda: update_position(idx, action),
            )

        deltas = jax.vmap(perform_agent_action)(jnp.arange(self.num_agents), actions)

        def update_health(unit_health: chex.Array, delta: WorldDelta):
            unit_health = unit_health.at[delta.attacked_idx].set(
                jnp.maximum(unit_health[delta.attacked_idx] + delta.health_diff, 0.0)
            )
            return unit_health, None

        unit_health, _ = jax.lax.scan(update_health, state.unit_health, deltas)
        state = state.replace(
            unit_health=unit_health,
            unit_positions=deltas.pos,
        )
        return state

    def get_world_state(self, state: State, params: EnvParams) -> chex.Array:
        # get the features of every unit, as well as the teams that they belong to.
        def get_features(i):
            empty_features = jnp.zeros(shape=(len(self.own_features),))
            features = empty_features.at[0].set(state.unit_health[i])
            features = features.at[1:3].set(state.unit_positions[i])
            return jax.lax.cond(
                state.unit_alive[i], lambda: features, lambda: empty_features
            )

        get_all_features = jax.vmap(get_features)
        unit_obs = get_all_features(jnp.arange(self.num_agents)).reshape(-1)
        unit_teams = state.unit_teams
        unit_types = state.unit_types
        return jnp.concatenate([unit_obs, unit_teams, unit_types], axis=-1)

    def get_obs(self, state: State, params: EnvParams) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        actions = state.prev_actions

        def get_features(i, j):
            """Get features of unit j as seen from unit i"""
            # Can just keep them symmetrical for now.
            # j here means 'the jth unit that is not i'
            # The observation is such that allies are always first
            # so for units in the second team we count in reverse.
            j = jax.lax.cond(
                i < params.num_agents_per_team,
                lambda: j,
                lambda: self.num_agents - j - 1,
            )
            offset = jax.lax.cond(i < params.num_agents_per_team, lambda: 1, lambda: -1)
            j_idx = jax.lax.cond(
                ((j < i) & (i < params.num_agents_per_team))
                | ((j > i) & (i >= params.num_agents_per_team)),
                lambda: j,
                lambda: j + offset,
            )
            empty_features = jnp.zeros(shape=(len(self.unit_features),))
            features = empty_features.at[0].set(state.unit_health[j_idx])
            features = features.at[1:3].set(
                (state.unit_positions[j_idx] - state.unit_positions[i])
                / params.unit_type_sight_ranges[state.unit_types[i]]
            )
            # TODO encode as one hot?
            features = features.at[4].set(actions[j_idx])
            visible = (
                jnp.linalg.norm(state.unit_positions[j_idx] - state.unit_positions[i])
                < params.unit_type_sight_ranges[state.unit_types[i]]
            )
            return jax.lax.cond(
                visible & state.unit_alive[i] & state.unit_alive[j_idx],
                lambda: features,
                lambda: empty_features,
            )

        def get_self_features(i):
            empty_features = jnp.zeros(shape=(len(self.own_features),))
            features = empty_features.at[0].set(state.unit_health[i])
            features = features.at[1:3].set(
                state.unit_positions[i]
                / jnp.array([params.map_width, params.map_height])
            )
            return jax.lax.cond(
                state.unit_alive[i], lambda: features, lambda: empty_features
            )

        get_all_features_for_unit = jax.vmap(get_features, in_axes=(None, 0))
        get_all_features = jax.vmap(get_all_features_for_unit, in_axes=(0, None))
        other_unit_obs = get_all_features(
            jnp.arange(self.num_agents), jnp.arange(self.num_agents - 1)
        )
        other_unit_obs = other_unit_obs.reshape((self.num_agents, -1))
        get_all_self_features = jax.vmap(get_self_features)
        own_unit_obs = get_all_self_features(jnp.arange(self.num_agents))
        obs = jnp.concatenate([other_unit_obs, own_unit_obs], axis=-1)
        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}

    def init_render(
        self,
        ax,
        state: Tuple[State, Dict],
        step: int,
        env_step: int,
        params: Optional[EnvParams] = None,
    ):
        from matplotlib.patches import Circle, Rectangle
        import numpy as np

        _, state, actions = state
        if params is None:
            params = self.default_params

        # work out which agents are being shot
        def agent_being_shot(shooter_idx, action):
            attacked_idx = jax.lax.cond(
                shooter_idx < self.num_agents_per_team,
                lambda: action + self.num_agents_per_team - self.num_movement_actions,
                lambda: params.num_agents_per_team
                - 1
                - (action - self.num_movement_actions),
            )
            return attacked_idx

        def agent_can_shoot(shooter_idx, action):
            attacked_idx = agent_being_shot(shooter_idx, action)
            dist = jnp.linalg.norm(
                state.unit_positions[shooter_idx] - state.unit_positions[attacked_idx]
            )
            return (
                state.unit_alive[shooter_idx]
                & state.unit_alive[attacked_idx]
                & (dist < params.unit_type_attack_ranges[state.unit_types[shooter_idx]])
            )

        attacked_agents = set(
            int(agent_being_shot(i, actions[agent]))
            for i, agent in enumerate(self.agents)
            if actions[agent] > self.num_movement_actions - 1
            and agent_can_shoot(i, actions[agent])
        )
        # render circles
        ax.clear()
        ax.set_xlim([0.0, params.map_width])
        ax.set_ylim([0.0, params.map_height])
        ax.set_title(f"Step {env_step}")
        for i in range(self.num_agents_per_team):
            if state.unit_alive[i]:
                color = "blue" if i not in attacked_agents else "red"
                c = Circle(
                    state.unit_positions[i],
                    params.unit_type_radiuses[state.unit_types[i]],
                    color=color,
                )
                ax.add_patch(c)
            if state.unit_alive[i + self.num_agents_per_team]:
                color = "green" if i not in attacked_agents else "red"
                c = Circle(
                    state.unit_positions[i + params.num_agents_per_team],
                    params.unit_type_radiuses[
                        state.unit_types[i + params.num_agents_per_team]
                    ],
                    color=color,
                )
                ax.add_patch(c)

        # render bullets
        for agent in self.agents:
            i = self.agent_ids[agent]
            attacked_idx = agent_being_shot(i, actions[agent])
            if actions[agent] < self.num_movement_actions or not agent_can_shoot(
                i, actions[agent]
            ):
                continue
            frac = step / self.world_steps_per_env_step
            bullet_pos = (1 - frac) * state.unit_positions[
                i
            ] + frac * state.unit_positions[attacked_idx]
            r = Rectangle(bullet_pos, 0.5, 0.5, color="gray")
            ax.add_patch(r)

        canvas = ax.figure.canvas
        canvas.draw()

        rgb_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        rgb_array = rgb_array.reshape(canvas.get_width_height()[::-1] + (3,))
        im = ax.imshow(rgb_array)

        return im

    def update_render(
        self,
        im,
        state: State,
        step: int,
        env_step: int,
        params: Optional[EnvParams] = None,
    ):
        ax = im.axes
        return self.init_render(ax, state, step, env_step, params)
