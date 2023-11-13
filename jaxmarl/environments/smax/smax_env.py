import jax.numpy as jnp
import jax
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete
from jaxmarl.environments.smax.distributions import (
    SurroundAndReflectPositionDistribution,
    UniformUnitTypeDistribution,
)
import chex
from typing import Tuple, Dict, Optional
from flax.struct import dataclass
from enum import IntEnum
from functools import partial
import io


@dataclass
class State:
    unit_positions: chex.Array
    unit_alive: chex.Array
    unit_teams: chex.Array
    unit_health: chex.Array
    unit_types: chex.Array
    unit_weapon_cooldowns: chex.Array
    prev_actions: chex.Array
    time: int
    terminal: bool


@dataclass
class WorldDelta:
    """Encapsulates the effect of an agent's action"""

    pos: chex.Array
    attacked_idx: int
    cooldown_diff: float
    health_diff: float


@dataclass
class Scenario:
    unit_types: chex.Array
    num_allies: int
    num_enemies: int
    smacv2_position_generation: bool
    smacv2_unit_type_generation: bool


MAP_NAME_TO_SCENARIO = {
    # name: (unit_types, n_allies, n_enemies, SMACv2 position generation, SMACv2 unit generation)
    "3m": Scenario(jnp.zeros((6,), dtype=jnp.uint8), 3, 3, False, False),
    "2s3z": Scenario(
        jnp.array([2, 2, 3, 3, 3] * 2, dtype=jnp.uint8), 5, 5, False, False
    ),
    "25m": Scenario(jnp.zeros((50,), dtype=jnp.uint8), 25, 25, False, False),
    "3s5z": Scenario(
        jnp.array(
            [
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
            ]
            * 2,
            dtype=jnp.uint8,
        ),
        8,
        8,
        False,
        False,
    ),
    "8m": Scenario(jnp.zeros((16,), dtype=jnp.uint8), 8, 8, False, False),
    "5m_vs_6m": Scenario(jnp.zeros((11,), dtype=jnp.uint8), 5, 6, False, False),
    "10m_vs_11m": Scenario(jnp.zeros((21,), dtype=jnp.uint8), 10, 11, False, False),
    "27m_vs_30m": Scenario(jnp.zeros((57,), dtype=jnp.uint8), 27, 30, False, False),
    "3s5z_vs_3s6z": Scenario(
        jnp.concatenate(
            [
                jnp.array([2, 2, 2, 3, 3, 3, 3, 3], dtype=jnp.uint8),
                jnp.array([2, 2, 2, 3, 3, 3, 3, 3, 3], dtype=jnp.uint8),
            ]
        ),
        8,
        9,
        False,
        False,
    ),
    "3s_vs_5z": Scenario(
        jnp.array([2, 2, 2, 3, 3, 3, 3, 3], dtype=jnp.uint8), 3, 5, False, False
    ),
    "6h_vs_8z": Scenario(
        jnp.array([5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3], dtype=jnp.uint8),
        6,
        8,
        False,
        False,
    ),
    "smacv2_5_units": Scenario(jnp.zeros((10,), dtype=jnp.uint8), 5, 5, True, True),
    "smacv2_10_units": Scenario(jnp.zeros((20,), dtype=jnp.uint8), 10, 10, True, True),
    "smacv2_20_units": Scenario(jnp.zeros((40,), dtype=jnp.uint8), 20, 20, True, True),
}


def map_name_to_scenario(map_name):
    """maps from smac map names to a scenario array"""
    return MAP_NAME_TO_SCENARIO[map_name]


def register_scenario(map_name, scenario):
    MAP_NAME_TO_SCENARIO[map_name] = scenario


class SMAX(MultiAgentEnv):
    def __init__(
        self,
        num_allies=5,
        num_enemies=5,
        map_width=32,
        map_height=32,
        world_steps_per_env_step=8,
        time_per_step=1.0 / 16,
        scenario=None,
        unit_type_names=[
            "marine",
            "marauder",
            "stalker",
            "zealot",
            "zergling",
            "hydralisk",
        ],
        unit_type_shorthands=["m", "M", "s", "Z", "z", "h"],
        unit_type_velocities=jnp.array([3.15, 2.25, 4.13, 3.15, 4.13, 3.15]),
        unit_type_attacks=jnp.array([9.0, 10.0, 13.0, 8.0, 5.0, 12.0]),
        unit_type_attack_ranges=jnp.array([5.0, 6.0, 6.0, 2.0, 2.0, 5.0]),
        unit_type_sight_ranges=jnp.array([9.0, 10.0, 10.0, 9.0, 8.0, 9.0]),
        unit_type_radiuses=jnp.array([0.375, 0.5625, 0.625, 0.5, 0.375, 0.625]),
        unit_type_health=jnp.array([45.0, 125.0, 160, 150, 35, 80]),
        unit_type_weapon_cooldowns=jnp.array([0.61, 1.07, 1.87, 0.86, 0.5, 0.59]),
        use_self_play_reward=False,
        see_enemy_actions=True,
        won_battle_bonus=1.0,
        walls_cause_death=True,
        max_steps=100,
        smacv2_position_generation=False,
        smacv2_unit_type_generation=False,
    ) -> None:
        self.num_allies = num_allies if scenario is None else scenario.num_allies
        self.num_enemies = num_enemies if scenario is None else scenario.num_enemies
        self.num_agents = self.num_allies + self.num_enemies
        self.walls_cause_death = walls_cause_death
        self.unit_type_names = unit_type_names
        self.unit_type_shorthands = unit_type_shorthands
        self.num_movement_actions = 5  # 5 cardinal directions + stop
        self.world_steps_per_env_step = world_steps_per_env_step
        self.map_width = map_width
        self.map_height = map_height
        self.scenario = scenario if scenario is None else scenario.unit_types
        self.use_self_play_reward = use_self_play_reward
        self.time_per_step = time_per_step
        self.unit_type_velocities = unit_type_velocities
        self.unit_type_weapon_cooldowns = unit_type_weapon_cooldowns
        self.unit_type_attacks = unit_type_attacks
        self.unit_type_attack_ranges = unit_type_attack_ranges
        self.unit_type_sight_ranges = unit_type_sight_ranges
        self.unit_type_radiuses = unit_type_radiuses
        self.unit_type_health = unit_type_health
        self.unit_type_bits = len(self.unit_type_names)
        self.max_steps = max_steps
        self.won_battle_bonus = won_battle_bonus
        self.see_enemy_actions = see_enemy_actions
        self.smacv2_unit_type_generation = (
            smacv2_unit_type_generation
            if scenario is None
            else scenario.smacv2_unit_type_generation
        )
        self.smacv2_position_generation = (
            smacv2_position_generation
            if scenario is None
            else scenario.smacv2_position_generation
        )
        self.position_generator = SurroundAndReflectPositionDistribution(
            self.num_allies, self.num_enemies, self.map_width, self.map_height
        )
        self.unit_type_generator = UniformUnitTypeDistribution(
            self.num_allies,
            self.num_enemies,
            self.map_width,
            self.map_height,
            len(self.unit_type_names),
        )
        self.agents = [f"ally_{i}" for i in range(self.num_allies)] + [
            f"enemy_{i}" for i in range(self.num_enemies)
        ]
        self.agent_ids = {agent: i for i, agent in enumerate(self.agents)}
        self.teams = jnp.zeros((self.num_agents,), dtype=jnp.uint8)
        self.teams = self.teams.at[self.num_allies :].set(1)
        self.own_features = ["health", "position_x", "position_y", "weapon_cooldown"]
        self.own_features += [f"unit_type_bit_{i}" for i in range(self.unit_type_bits)]
        self.unit_features = [
            "health",
            "position_x",
            "position_y",
            "last_action",
            "weapon_cooldown",
        ]
        self.unit_features += [
            f"unit_type_bits_{i}" for i in range(self.unit_type_bits)
        ]
        self.obs_size = (
            len(self.unit_features) * (self.num_allies - 1)
            + len(self.unit_features) * self.num_enemies
            + len(self.own_features)
        )
        self.state_size = (len(self.own_features) + 2) * self.num_agents
        self.observation_spaces = {
            i: Box(low=0.0, high=1.0, shape=(self.obs_size,)) for i in self.agents
        }
        self.num_ally_actions = self.num_enemies + self.num_movement_actions
        self.num_enemy_actions = self.num_allies + self.num_movement_actions
        self.action_spaces = {
            agent: Discrete(
                num_categories=self.num_ally_actions
                if i < self.num_allies
                else self.num_enemy_actions
            )
            for i, agent in enumerate(self.agents)
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Environment-specific reset."""
        key, team_0_key, team_1_key = jax.random.split(key, num=3)
        team_0_start = jnp.stack([jnp.array([8.0, 16.0])] * self.num_allies)
        team_0_start_noise = jax.random.uniform(
            team_0_key, shape=(self.num_allies, 2), minval=-2, maxval=2
        )
        team_0_start = team_0_start + team_0_start_noise
        team_1_start = jnp.stack([jnp.array([24.0, 16.0])] * self.num_enemies)
        team_1_start_noise = jax.random.uniform(
            team_1_key, shape=(self.num_enemies, 2), minval=-2, maxval=2
        )
        team_1_start = team_1_start + team_1_start_noise
        unit_positions = jnp.concatenate([team_0_start, team_1_start])
        key, pos_key = jax.random.split(key)
        generated_unit_positions = self.position_generator.generate(pos_key)
        unit_positions = jax.lax.select(
            self.smacv2_position_generation, generated_unit_positions, unit_positions
        )
        unit_teams = jnp.zeros((self.num_agents,))
        unit_teams = unit_teams.at[self.num_allies :].set(1)
        unit_weapon_cooldowns = jnp.zeros((self.num_agents,))
        # default behaviour spawn all marines
        unit_types = (
            jnp.zeros((self.num_agents,), dtype=jnp.uint8)
            if self.scenario is None
            else self.scenario
        )
        key, unit_type_key = jax.random.split(key)
        generated_unit_types = self.unit_type_generator.generate(unit_type_key)
        unit_types = jax.lax.select(
            self.smacv2_unit_type_generation, generated_unit_types, unit_types
        )
        unit_health = self.unit_type_health[unit_types]
        state = State(
            unit_positions=unit_positions,
            unit_alive=jnp.ones((self.num_agents,), dtype=jnp.bool_),
            unit_teams=unit_teams,
            unit_health=unit_health,
            unit_types=unit_types,
            prev_actions=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            time=0,
            terminal=False,
            unit_weapon_cooldowns=unit_weapon_cooldowns,
        )
        state = self._push_units_away(state)
        obs = self.get_obs(state)
        world_state = self.get_world_state(state)
        obs["world_state"] = jax.lax.stop_gradient(world_state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        actions = jnp.array([actions[i] for i in self.agents])
        health_before = jnp.copy(state.unit_health)

        def world_step_fn(carry, _):
            state, step_key = carry
            step_key, world_step_key = jax.random.split(step_key)
            state = partial(self._world_step, actions=actions)(
                key=world_step_key, state=state
            )
            state = self._kill_agents_touching_walls(state)
            state = self._update_dead_agents(state)
            state = self._push_units_away(state)
            return (state, step_key), None

        (state, _), _ = jax.lax.scan(
            world_step_fn,
            init=(state, key),
            xs=None,
            length=self.world_steps_per_env_step,
        )
        health_after = state.unit_health
        state = state.replace(
            terminal=self.is_terminal(state), prev_actions=actions, time=state.time + 1
        )
        obs = self.get_obs(state)
        dones = {
            agent: ~state.unit_alive[self.agent_ids[agent]] for agent in self.agents
        }
        rewards = self.compute_reward(state, health_before, health_after)
        dones["__all__"] = state.terminal
        world_state = self.get_world_state(state)
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
    def compute_reward(self, state, health_before, health_after):
        @partial(jax.jit, static_argnums=(0,))
        def compute_team_reward(team_idx):
            # compute how much the enemy team health has decreased
            other_team_idx = jnp.logical_not(team_idx).astype(jnp.uint32)
            other_team_start_idx = jnp.array([0, self.num_allies])[other_team_idx]
            team_start_idx = jnp.array([0, self.num_allies])[team_idx]

            team_size = self.num_allies if team_idx == 0 else self.num_enemies

            enemy_team_size = self.num_enemies if team_idx == 0 else self.num_allies

            enemy_health_decrease = jnp.sum(
                jax.lax.dynamic_slice_in_dim(
                    (health_after - health_before)
                    / self.unit_type_health[state.unit_types],
                    other_team_start_idx,
                    enemy_team_size,
                )
            )
            enemy_health_decrease_reward = (
                jnp.abs(enemy_health_decrease) / enemy_team_size
            )
            enemy_health_decrease_reward = jax.lax.select(
                self.use_self_play_reward, 0.0, enemy_health_decrease_reward
            )
            won_battle = jnp.all(
                jnp.logical_not(
                    jax.lax.dynamic_slice_in_dim(
                        state.unit_alive, other_team_start_idx, enemy_team_size
                    )
                )
            )
            lost_battle = jnp.all(
                jnp.logical_not(
                    jax.lax.dynamic_slice_in_dim(
                        state.unit_alive, team_start_idx, team_size
                    )
                )
            )
            # have a lost battle bonus in addition to the won bonus in
            # order to make the game zero-sum in self-play and therefore prevent any
            # collaboration.
            lost_battle_bonus = jax.lax.cond(
                lost_battle & self.use_self_play_reward & ~won_battle,
                lambda: -self.won_battle_bonus,
                lambda: 0.0,
            )
            # only award the won_battle_bonus when all the enemy is dead
            # AND there is at least one ally alive. Otherwise it's a draw.
            # This can't happen in SC2 because actions happen in a random order,
            # but I'd rather VMAP over events where possible, which means we
            # can get draws.
            won_battle_bonus = jax.lax.cond(
                won_battle & ~lost_battle, lambda: self.won_battle_bonus, lambda: 0.0
            )
            return enemy_health_decrease_reward + won_battle_bonus + lost_battle_bonus

        # agents still get reward when they are dead to allow for noble sacrifice
        team_rewards = [compute_team_reward(i) for i in range(2)]
        return {
            agent: team_rewards[int(self.agent_ids[agent] >= self.num_allies)]
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state):
        all_dead = jnp.all(jnp.logical_not(state.unit_alive[: self.num_allies]))
        all_enemy_dead = jnp.all(jnp.logical_not(state.unit_alive[self.num_allies :]))
        over_time_limit = state.time >= self.max_steps
        return all_dead | all_enemy_dead | over_time_limit

    def _update_dead_agents(
        self,
        state: State,
    ):
        unit_alive = state.unit_health > 0
        return state.replace(unit_alive=unit_alive)

    def _kill_agents_touching_walls(self, state: State):
        units_touching_walls = jnp.logical_or(
            jnp.any(state.unit_positions <= 0.0, axis=-1),
            jnp.any(
                state.unit_positions >= jnp.array([self.map_width, self.map_height]),
                axis=-1,
            ),
        )
        unit_health = jnp.where(units_touching_walls, 0.0, state.unit_health)
        unit_health = jax.lax.select(
            self.walls_cause_death, unit_health, state.unit_health
        )
        return state.replace(unit_health=unit_health)

    def _push_units_away(self, state: State, firmness: float = 1.0):
        delta_matrix = state.unit_positions[:, None] - state.unit_positions[None, :]
        dist_matrix = (
            jnp.linalg.norm(delta_matrix, axis=-1)
            + jnp.identity(self.num_agents)
            + 1e-6
        )
        radius_matrix = (
            self.unit_type_radiuses[state.unit_types][:, None]
            + self.unit_type_radiuses[state.unit_types][None, :]
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
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        def update_position(idx, action):
            # Compute the movements slightly strangely.
            # The velocities below are for diagonal directions
            # because these are easier to encode as actions than the four
            # diagonal directions. Then rotate the velocity 45
            # degrees anticlockwise to compute the movement.
            pos = state.unit_positions[idx]
            vec = jax.lax.cond(
                # action is an attack action OR stop (action 4)
                action >= self.num_movement_actions - 1,
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
            new_pos = (
                pos
                + vec
                * self.unit_type_velocities[state.unit_types[idx]]
                * self.time_per_step
            )
            # avoid going out of bounds
            new_pos = jnp.maximum(
                jnp.minimum(new_pos, jnp.array([self.map_width, self.map_height])),
                jnp.zeros((2,)),
            )
            return WorldDelta(new_pos, idx, -self.time_per_step, 0.0)

        def update_agent_health(idx, action, key):
            # for team 1, their attack actions are labelled in
            # reverse order because that is the order they are
            # observed in
            attacked_idx = jax.lax.cond(
                idx < self.num_allies,
                lambda: action + self.num_allies - self.num_movement_actions,
                lambda: self.num_allies - 1 - (action - self.num_movement_actions),
            )
            attack_valid = (
                (
                    jnp.linalg.norm(
                        state.unit_positions[idx] - state.unit_positions[attacked_idx]
                    )
                    < self.unit_type_attack_ranges[state.unit_types[idx]]
                )
                & state.unit_alive[idx]
                & state.unit_alive[attacked_idx]
            )
            attack_valid = attack_valid & (state.unit_weapon_cooldowns[idx] <= 0.0)
            health_diff = jax.lax.select(
                attack_valid,
                -self.unit_type_attacks[state.unit_types[idx]],
                0.0,
            )
            # design choice based on the pysc2 randomness details.
            # See https://github.com/deepmind/pysc2/blob/master/docs/environment.md#determinism-and-randomness

            cooldown_deviation = jax.random.uniform(
                key, minval=-self.time_per_step, maxval=2 * self.time_per_step
            )
            cooldown = (
                self.unit_type_weapon_cooldowns[state.unit_types[idx]]
                + cooldown_deviation
            )
            cooldown_diff = jax.lax.select(
                attack_valid,
                cooldown - state.unit_weapon_cooldowns[idx],
                -self.time_per_step,
            )
            return WorldDelta(
                state.unit_positions[idx],
                attacked_idx,
                cooldown_diff,
                health_diff,
            )

        def perform_agent_action(idx, action, key):
            return jax.lax.cond(
                actions[idx] > self.num_movement_actions - 1,
                lambda: update_agent_health(idx, action, key),
                lambda: update_position(idx, action),
            )

        keys = jax.random.split(key, num=self.num_agents)
        deltas = jax.vmap(perform_agent_action)(
            jnp.arange(self.num_agents), actions, keys
        )

        def update_health(carry: chex.Array, delta: WorldDelta):
            unit_health, unit_weapon_cooldowns, idx = carry
            unit_health = unit_health.at[delta.attacked_idx].set(
                jnp.maximum(unit_health[delta.attacked_idx] + delta.health_diff, 0.0)
            )
            unit_weapon_cooldowns = unit_weapon_cooldowns.at[idx].set(
                state.unit_weapon_cooldowns[idx] + delta.cooldown_diff
            )
            return (unit_health, unit_weapon_cooldowns, idx + 1), None

        # TODO, just sum the deltas and take the max w/ 0
        (unit_health, unit_weapon_cooldowns, _), _ = jax.lax.scan(
            update_health, (state.unit_health, state.unit_weapon_cooldowns, 0), deltas
        )
        state = state.replace(
            unit_health=unit_health,
            unit_positions=deltas.pos,
            unit_weapon_cooldowns=unit_weapon_cooldowns,
        )
        return state

    def get_world_state(self, state: State) -> chex.Array:
        # get the features of every unit, as well as the teams that they belong to.
        def get_features(i):
            empty_features = jnp.zeros(shape=(len(self.own_features),))
            features = empty_features.at[0].set(
                state.unit_health[i] / self.unit_type_health[state.unit_types[i]]
            )
            features = features.at[1:3].set(state.unit_positions[i])
            features = features.at[3].set(state.unit_weapon_cooldowns[i])
            features = features.at[4 + state.unit_types[i]].set(1)
            return jax.lax.cond(
                state.unit_alive[i], lambda: features, lambda: empty_features
            )

        get_all_features = jax.vmap(get_features)
        unit_obs = get_all_features(jnp.arange(self.num_agents)).reshape(-1)
        unit_teams = state.unit_teams
        unit_types = state.unit_types
        return jnp.concatenate([unit_obs, unit_teams, unit_types], axis=-1)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        actions = state.prev_actions

        def get_features(i, j):
            """Get features of unit j as seen from unit i"""
            # Can just keep them symmetrical for now.
            # j here means 'the jth unit that is not i'
            # The observation is such that allies are always first
            # so for units in the second team we count in reverse.
            team_i_idx = (i >= self.num_allies).astype(jnp.int32)
            j = jax.lax.cond(
                i < self.num_allies,
                lambda: j,
                lambda: self.num_agents - j - 1,
            )
            offset = jax.lax.cond(i < self.num_allies, lambda: 1, lambda: -1)
            j_idx = jax.lax.cond(
                ((j < i) & (i < self.num_allies)) | ((j > i) & (i >= self.num_allies)),
                lambda: j,
                lambda: j + offset,
            )
            team_j_idx = (j_idx >= self.num_allies).astype(jnp.int32)
            empty_features = jnp.zeros(shape=(len(self.unit_features),))
            features = empty_features.at[0].set(
                state.unit_health[j_idx]
                / self.unit_type_health[state.unit_types[j_idx]]
            )
            features = features.at[1:3].set(
                (state.unit_positions[j_idx] - state.unit_positions[i])
                / self.unit_type_sight_ranges[state.unit_types[i]]
            )
            # TODO encode as one hot?
            action_obs = jax.lax.select(
                (team_i_idx == team_j_idx) | self.see_enemy_actions, actions[j_idx], 0
            )
            features = features.at[3].set(action_obs)
            features = features.at[4].set(state.unit_weapon_cooldowns[j_idx])
            features = features.at[5 + state.unit_types[j_idx]].set(1)
            visible = (
                jnp.linalg.norm(state.unit_positions[j_idx] - state.unit_positions[i])
                < self.unit_type_sight_ranges[state.unit_types[i]]
            )
            return jax.lax.cond(
                visible & state.unit_alive[i] & state.unit_alive[j_idx],
                lambda: features,
                lambda: empty_features,
            )

        def get_self_features(i):
            empty_features = jnp.zeros(shape=(len(self.own_features),))
            features = empty_features.at[0].set(
                state.unit_health[i] / self.unit_type_health[state.unit_types[i]]
            )
            features = features.at[1:3].set(
                state.unit_positions[i] / jnp.array([self.map_width, self.map_height])
            )
            features = features.at[3].set(state.unit_weapon_cooldowns[i])
            features = features.at[4 + state.unit_types[i]].set(1)
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

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        @partial(jax.jit, static_argnums=(1,))
        def get_individual_avail_actions(i, team):
            num_actions = {0: self.num_ally_actions, 1: self.num_enemy_actions}[team]
            is_alive = state.unit_alive[i]
            mask = jnp.zeros((num_actions,), dtype=jnp.uint8)
            # always can take the stop action
            mask = mask.at[self.num_movement_actions - 1].set(1)
            mask = mask.at[: self.num_movement_actions - 1].set(
                jax.lax.select(
                    is_alive,
                    jnp.ones((self.num_movement_actions - 1,), dtype=jnp.uint8),
                    jnp.zeros((self.num_movement_actions - 1,), dtype=jnp.uint8),
                )
            )
            shootable_mask = (
                jnp.linalg.norm(state.unit_positions - state.unit_positions[i], axis=-1)
                < self.unit_type_attack_ranges[state.unit_types[i]]
            ) & state.unit_alive
            shootable_mask = shootable_mask if team == 0 else shootable_mask[::-1]
            shootable_mask = (
                shootable_mask[self.num_allies :]
                if team == 0
                else shootable_mask[self.num_enemies :]
            )
            shootable_mask = jax.lax.select(
                is_alive, shootable_mask, jnp.zeros_like(shootable_mask)
            )
            mask = mask.at[self.num_movement_actions :].set(shootable_mask)
            return mask

        ally_avail_actions_masks = jax.vmap(
            get_individual_avail_actions, in_axes=(0, None)
        )(jnp.arange(self.num_allies), 0)
        enemy_avail_actions_masks = jax.vmap(
            get_individual_avail_actions, in_axes=(0, None)
        )(jnp.arange(self.num_allies, self.num_agents), 1)
        return {
            agent: ally_avail_actions_masks[i]
            if i < self.num_allies
            else enemy_avail_actions_masks[i - self.num_allies]
            for i, agent in enumerate(self.agents)
        }

    def expand_state_seq(self, state_seq):
        expanded_state_seq = []
        for key, state, actions in state_seq:
            agents = self.agents
            for _ in range(self.world_steps_per_env_step):
                expanded_state_seq.append((key, state, actions))
                world_actions = jnp.array([actions[i] for i in agents])
                key, step_key = jax.random.split(key)
                state = self._world_step(step_key, state, world_actions)
                state = self._kill_agents_touching_walls(state)
                state = self._update_dead_agents(state)
                state = self._push_units_away(state)
            state = state.replace(terminal=self.is_terminal(state))
        return expanded_state_seq

    def init_render(
        self,
        ax,
        state: Tuple[State, Dict],
        step: int,
        env_step: int,
    ):
        from matplotlib.patches import Circle, Rectangle
        import matplotlib.pyplot as plt
        import numpy as np

        _, state, actions = state

        # work out which agents are being shot
        def agent_being_shot(shooter_idx, action):
            attacked_idx = jax.lax.cond(
                shooter_idx < self.num_allies,
                lambda: action + self.num_allies - self.num_movement_actions,
                lambda: self.num_allies - 1 - (action - self.num_movement_actions),
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
                & (dist < self.unit_type_attack_ranges[state.unit_types[shooter_idx]])
            )

        attacked_agents = set(
            int(agent_being_shot(i, actions[agent]))
            for i, agent in enumerate(self.agents)
            if actions[agent] > self.num_movement_actions - 1
            and agent_can_shoot(i, actions[agent])
        )
        # render circles
        ax.clear()
        ax.set_xlim([0.0, self.map_width])
        ax.set_ylim([0.0, self.map_height])
        ax.set_title(f"Step {env_step}")
        for i in range(self.num_allies):
            if state.unit_alive[i]:
                color = "blue" if i not in attacked_agents else "cornflowerblue"
                c = Circle(
                    state.unit_positions[i],
                    self.unit_type_radiuses[state.unit_types[i]],
                    color=color,
                )
                ax.add_patch(c)
                ax.text(
                    state.unit_positions[i][0]
                    - (1.0 / jnp.sqrt(2))
                    * self.unit_type_radiuses[state.unit_types[i]],
                    state.unit_positions[i][1]
                    - (1.0 / jnp.sqrt(2))
                    * self.unit_type_radiuses[state.unit_types[i]],
                    self.unit_type_shorthands[state.unit_types[i]],
                    fontsize="xx-small",
                    color="white",
                )
        for i in range(self.num_enemies):
            idx = i + self.num_allies
            if state.unit_alive[idx]:
                color = "green" if idx not in attacked_agents else "limegreen"
                c = Circle(
                    state.unit_positions[idx],
                    self.unit_type_radiuses[state.unit_types[idx]],
                    color=color,
                )
                ax.add_patch(c)
                ax.text(
                    state.unit_positions[idx][0]
                    - (1.0 / jnp.sqrt(2))
                    * self.unit_type_radiuses[state.unit_types[idx]],
                    state.unit_positions[idx][1]
                    - (1.0 / jnp.sqrt(2))
                    * self.unit_type_radiuses[state.unit_types[idx]],
                    self.unit_type_shorthands[state.unit_types[idx]],
                    fontsize="xx-small",
                    color="white",
                )

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

        with io.BytesIO() as buff:
            ax.figure.savefig(buff, format="raw")
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = ax.figure.canvas.get_width_height()
        im = data.reshape((w, h, -1))

        return ax.imshow(im)

    def update_render(
        self,
        im,
        state: State,
        step: int,
        env_step: int,
    ):
        ax = im.axes
        return self.init_render(ax, state, step, env_step)
