import jax.numpy as jnp
import jax
import chex


def create_heuristic_policy(env, team: int, shoot: bool = True):
    num_unit_features = len(env.unit_features)
    num_move_actions = env.num_movement_actions

    def get_heuristic_action(key: jax.random.PRNGKey, obs: chex.Array):
        """Generate a heuristic action based on an observation.
        Follows the following scheme:
            -- If you can attack:
                -- Find all the enemies that are in range
                -- Attack one at random
            -- If you can't attack:
                -- Go to just past the middle of the enemy's half
        """
        attack_range = env.unit_type_attack_ranges[0]
        first_enemy_idx = (env.num_agents_per_team - 1) * num_unit_features
        own_feats_idx = (env.num_agents_per_team * 2 - 1) * num_unit_features

        def scaled_position_to_map(position, x_scale, y_scale):
            return position * jnp.array([x_scale, y_scale])

        own_position = scaled_position_to_map(
            obs[own_feats_idx + 1 : own_feats_idx + 3],
            env.map_width,
            env.map_height,
        )
        enemy_positions = jnp.zeros((env.num_agents_per_team, 2))
        enemy_positions = enemy_positions.at[:, 0].set(
            obs[first_enemy_idx + 1 : own_feats_idx : num_unit_features],
        )
        enemy_positions = enemy_positions.at[:, 1].set(
            obs[first_enemy_idx + 2 : own_feats_idx : num_unit_features]
        )
        enemy_positions = scaled_position_to_map(
            enemy_positions,
            env.unit_type_sight_ranges[0],
            env.unit_type_sight_ranges[0],
        )
        # visible if health is > 0. Otherwise out of range or dead
        visible_enemy_mask = obs[first_enemy_idx:own_feats_idx:num_unit_features] > 0
        shootable_enemy_mask = (
            jnp.linalg.norm(enemy_positions, axis=-1) < attack_range
        ) & visible_enemy_mask
        can_shoot = jnp.any(shootable_enemy_mask)
        key, key_attack = jax.random.split(key)
        attack_action = jax.random.choice(
            key_attack,
            jnp.arange(num_move_actions, env.num_agents_per_team + num_move_actions),
            p=(shootable_enemy_mask / jnp.sum(shootable_enemy_mask)),
        )
        # compute the correct movement action.
        random_enemy_target = jax.random.choice(
            key,
            enemy_positions + own_position,
            p=(visible_enemy_mask / jnp.sum(visible_enemy_mask)),
        )
        can_see = jnp.any(visible_enemy_mask)
        team_0_target = jax.lax.cond(
            can_see, lambda: random_enemy_target, lambda: jnp.array([28.0, 16.0])
        )
        team_1_target = jax.lax.cond(
            can_see, lambda: random_enemy_target, lambda: jnp.array([4.0, 16.0])
        )
        target = jax.lax.cond(team == 0, lambda: team_0_target, lambda: team_1_target)
        vector_to_target = target - own_position
        action_vectors = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        similarity = jnp.dot(action_vectors, vector_to_target)
        move_action = jnp.argmax(similarity)
        return jax.lax.cond(
            can_shoot & shoot, lambda: attack_action, lambda: move_action
        )

    return get_heuristic_action
