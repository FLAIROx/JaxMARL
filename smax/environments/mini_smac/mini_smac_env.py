import jax.numpy as jnp
import jax
from smax.environments.multi_agent_env import MultiAgentEnv
from smax.environments.spaces import Box, Discrete
import chex
from typing import Tuple, Dict
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
    unit_type_attacks: float
    unit_type_attack_ranges: float
    unit_type_sight_ranges: float
    time_per_step: float
    max_steps: int
    won_battle_bonus: float


class MiniSMAC(MultiAgentEnv):
    def __init__(
        self,
        num_agents_per_team: int = 5,
        map_width: float = 32,
        map_height: float = 32,
        world_steps_per_env_step: int = 8,
        unit_velocity: float = 1.0,
        time_per_step: float = 1.0 / 16,
        unit_type_attacks: chex.Array = jnp.array([0.02]),
        unit_type_attack_ranges: chex.Array = jnp.array([3.0]),
        unit_type_sight_ranges: chex.Array = jnp.array([5.0]),
        won_battle_bonus: float = 5,
        max_steps: int = 100,
    ) -> None:
        self.num_agents_per_team = num_agents_per_team
        self.num_agents = num_agents_per_team * 2
        self.num_movement_actions = 4
        self.params = EnvParams(
            num_agents_per_team=num_agents_per_team,
            map_width=map_width,
            map_height=map_height,
            world_steps_per_env_step=world_steps_per_env_step,
            unit_velocity=unit_velocity,
            unit_type_attacks=unit_type_attacks,
            time_per_step=time_per_step,
            won_battle_bonus=won_battle_bonus,
            unit_type_attack_ranges=unit_type_attack_ranges,
            unit_type_sight_ranges=unit_type_sight_ranges,
            max_steps=max_steps,
        )
        self.world_steps_per_env_step = world_steps_per_env_step
        self.agents = [f"ally_{i}" for i in range(self.num_agents_per_team)] + [
            f"enemy_{i}" for i in range(self.num_agents_per_team)
        ]
        self.agent_ids = {agent: i for i, agent in enumerate(self.agents)}
        self.teams = jnp.zeros((self.num_agents,), dtype=jnp.uint8)
        self.teams = self.teams.at[self.num_agents_per_team :].set(1)
        self.own_features = ["health", "position_x", "position_y"]
        self.unit_features = [
            "health",
            "position_x",
            "position_y",
        ]
        self.obs_size = (
            len(self.unit_features) * (self.num_agents_per_team - 1)
            + len(self.unit_features) * self.num_agents_per_team
            + len(self.own_features)
        )
        self.observation_spaces = {
            i: Box(low=0.0, high=1.0, shape=(self.obs_size,)) for i in self.agents
        }
        self.action_spaces = {i: Discrete(num_categories=5) for i in self.agents}

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
            won_battle_bonus=5,
            unit_type_attack_ranges=jnp.array([3.0]),
            unit_type_sight_ranges=jnp.array([4.0]),
            max_steps=100,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Environment-specific reset."""
        team_0_start = jnp.stack([jnp.array([8.0, 16.0])] * self.num_agents_per_team)
        team_1_start = jnp.stack([jnp.array([24.0, 16.0])] * self.num_agents_per_team)
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
            time=0,
            terminal=False,
        )
        return self.get_obs(state, params), state

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
            return carry, None

        state, _ = jax.lax.scan(
            world_step_fn, init=state, xs=None, length=self.world_steps_per_env_step
        )
        health_after = state.unit_health
        state = state.replace(terminal=self.is_terminal(state, params))
        obs = self.get_obs(state, params)
        dones = {
            agent: ~state.unit_alive[self.agent_ids[agent]] for agent in self.agents
        }
        rewards = self.compute_reward(state, params, health_before, health_after)
        dones["__all__"] = state.terminal
        return (
            jax.lax.stop_gradient(obs),
            jax.lax.stop_gradient(state),
            rewards,
            dones,
            {},
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

    @partial(jax.jit, static_argnums=(0,))
    def _world_step(
        self,
        state: State,
        actions: Dict[str, chex.Array],
        params: EnvParams,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        def update_agent_pos(state, params, idx, action):
            # MOVE
            pos = state.unit_positions[idx]
            # Compute the movements slightly strangely.
            # The velocities below are for diagonal directions
            # because these are easier to encode as actions than the four
            # diagonal directions. Then rotate the velocity 45
            # degrees anticlockwise to compute the movement.
            vel = jax.lax.cond(
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
            vel = rotation @ vel
            new_pos = pos + vel * params.time_per_step
            # avoid going out of bounds
            new_pos = jnp.maximum(
                jnp.minimum(new_pos, jnp.array([params.map_width, params.map_height])),
                jnp.zeros((2,)),
            )
            unit_positions = state.unit_positions.at[idx].set(new_pos)
            state = state.replace(unit_positions=unit_positions)
            return state

        def update_agent_health(state, params, idx, action):
            attacked_idx = jax.lax.cond(
                idx < self.num_agents_per_team,
                lambda: action + self.num_agents_per_team - self.num_movement_actions,
                lambda: action - self.num_movement_actions,
            )
            attack_valid = (
                jnp.linalg.norm(
                    state.unit_positions[idx] - state.unit_positions[attacked_idx]
                )
                < params.unit_type_attack_ranges[state.unit_types[idx]]
            )
            unit_health = state.unit_health.at[attacked_idx].set(
                jax.lax.cond(
                    attack_valid,
                    lambda: jnp.maximum(
                        state.unit_health[attacked_idx]
                        - params.unit_type_attacks[state.unit_types[idx]],
                        0,
                    ),
                    lambda: state.unit_health[attacked_idx],
                )
            )
            state = state.replace(unit_health=unit_health)
            return state

        def perform_agent_action(carry, idx):
            state, params, actions = carry
            # do nothing if the agent is dead

            updated_state = jax.lax.cond(
                actions[idx] > self.num_movement_actions - 1,
                lambda: update_agent_health(state, params, idx, actions[idx]),
                lambda: update_agent_pos(state, params, idx, actions[idx]),
            )
            state = jax.lax.cond(
                state.unit_alive[idx], lambda: updated_state, lambda: state
            )
            return (state, params, actions), None

        (state, params, actions), _ = jax.lax.scan(
            perform_agent_action,
            init=(state, params, actions),
            xs=jnp.arange(self.num_agents),
        )
        return state

    def get_obs(self, state: State, params: EnvParams) -> Dict[str, chex.Array]:
        """Applies observation function to state."""

        def get_features(i, j):
            """Get features of unit j as seen from unit i"""
            # much easier if features are symmetrical. Then can VMAP.
            # Can just keep them symmetrical for now.
            # j here means 'the jth unit that is not i'
            j_idx = jax.lax.cond(j < i, lambda: j, lambda: j + 1)
            empty_features = jnp.zeros(shape=(len(self.unit_features),))
            features = empty_features.at[0].set(state.unit_health[j_idx])
            features = features.at[1:3].set(state.unit_positions[j_idx])
            visible = (
                jnp.linalg.norm(state.unit_positions[j_idx] - state.unit_positions[i])
                < params.unit_type_sight_ranges[state.unit_types[i]]
            )
            return jax.lax.cond(
                visible & state.unit_alive[i], lambda: features, lambda: empty_features
            )

        def get_self_features(i):
            empty_features = jnp.zeros(shape=(len(self.own_features),))
            features = empty_features.at[0].set(state.unit_health[i])
            features = features.at[1:3].set(state.unit_positions[i])
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
