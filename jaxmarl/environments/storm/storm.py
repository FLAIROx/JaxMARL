'''
    A generalised implementation of JaxMARL's STORM.

    NOTE: 
        - Change in the observation space:
            - Items at each location are encoded as a mixed one-hot vector;
            - Each vector is the concatenation of the following form:
                - [
                    present item
                    relative angle/orientation of other agent,
                    pick-up eligibility of other agent,
                    inventory of other agent
                    frozen
                ]
              where:
                1. "present item" is a length-6 one-hot vector for each of:
                wall, interact, red coin, blue coin, observing agent and other
                agent
                2. "relative angle/orientation" is a length-4 one-hot vector,
                for each of up, right, down, left (relative)
                3. "pick-up eligibility" is length-1 1 hot vector for
                eligibility of the "other" agent to interact
                4. inventory is a length-2 vector, for the amount of red and
                blue coins held by the "other" agent (visibility dependent)
                5. "frozen" is a length-1 1 hot vector for whether the "other"
                agent is frozen
'''
from enum import IntEnum
import math
from typing import Any, Optional, Tuple, Union, Dict
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass
import colorsys

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces


from jaxmarl.environments.storm.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

NUM_TYPES = 5  # empty (0), red (1), blue, red coin, blue coin, wall, interact
NUM_COIN_TYPES = 2
INTERACT_THRESHOLD = 0


@dataclass
class State:
    agent_locs: jnp.ndarray
    agent_invs: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    red_coins: jnp.ndarray
    blue_coins: jnp.ndarray

    freeze: jnp.ndarray

@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    freeze_penalty: int

class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    stay = 3
    zap_forward = 4
    zap_ahead = 5
    zap_right = 6
    zap_left = 7

class Items(IntEnum):
    empty = 0
    wall = 1
    interact = 2
    red_coin = 3
    blue_coin = 4

ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # forward
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap forward
        [0, 0, 0],  # zap ahead
        [0, 0, 0],  # zap right
        [0, 0, 0],  # zap left
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [0, 1, 0],  # up
        [1, 0, 0],  # right
        [0, -1, 0],  # down
        [-1, 0, 0],  # left
    ],
    dtype=jnp.int8,
)

def generate_agent_colors(num_agents):
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)  # Saturation and Value set to 0.8
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)
###################################################

class InTheMatrix(MultiAgentEnv):
    """
    JAX Compatible n-agent version of *inTheMatix environment.
    """

    # used for caching
    tile_cache: Dict[Tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps=152,
        num_outer_steps=1,
        fixed_coin_location=True,
        num_agents=2,
        payoff_matrix=jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]]),
        freeze_penalty=5,
        grid_size=8,
        obs_size=5,
        num_coins=6,
        cnn=True
    ):

        super().__init__(num_agents=num_agents)
        self.agents = list(range(num_agents))#, dtype=jnp.int16)
        self._agents = jnp.array(self.agents, dtype=jnp.int16) + len(Items)

        self.PLAYER_COLOURS = generate_agent_colors(num_agents)
        self.GRID_SIZE = grid_size
        self.OBS_SIZE = obs_size
        self.PADDING = self.OBS_SIZE - 1
        self.NUM_COINS = num_coins  # per type
        NUM_OBJECTS = (
            2 + NUM_COIN_TYPES * self.NUM_COINS + 1
        )  # red, blue, 2 red coin, 2 blue coin

        GRID = jnp.zeros(
            (self.GRID_SIZE + 2 * self.PADDING, self.GRID_SIZE + 2 * self.PADDING),
            dtype=jnp.int16,
        )

        # First layer of padding is Wall
        GRID = GRID.at[self.PADDING - 1, :].set(5)
        GRID = GRID.at[self.GRID_SIZE + self.PADDING, :].set(5)
        GRID = GRID.at[:, self.PADDING - 1].set(5)
        self.GRID = GRID.at[:, self.GRID_SIZE + self.PADDING].set(5)

        self.SPAWNS = jnp.concat(
            jax.vmap(
                jax.vmap(
                    lambda x, y: jnp.array([x, y], dtype=jnp.int16),
                    in_axes=(0, None)
                ),
                in_axes=(None, 0)
            )(
                jnp.array(range(self.GRID_SIZE)),
                jnp.array(range(self.GRID_SIZE))
            ), 
            axis=0
            )

        def rand_interaction(
                key: int,
                conflicts: jnp.ndarray,
                conflicts_matrix: jnp.ndarray,
                step_arr: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function for randomly choosing between conflicting interactions.
            
            Args:
                - key: jax PRNGKey for randomisation.
                - conflicts: jnp.ndarray of bools where True if agent is in a
                conflicting interaction, False otherwise.
                - conflicts_matrix: jnp.ndarray matrix of bools of agents in
                conflicting interactions.
                - step_arr: jnp.ndarray, where each index is the index of an
                agent, and the element at each index is the item found at that
                agent's respective target location in the grid.

                
            Returns:
                - jnp.ndarray array of final interactions, where each index is
                an agent, and each element is caught in its interaction beam.
            '''
            def scan_fn(
                    state,
                    idx
            ):

                key, conflicts, conflicts_matrix, step_arr = state

                return jax.lax.cond(
                    conflicts[idx] > 0,
                    lambda: _rand_interaction(
                        key,
                        conflicts,
                        conflicts_matrix,
                        step_arr
                    ),
                    lambda: (state, step_arr.astype(jnp.int16))
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, conflicts, conflicts_matrix, step_arr.astype(jnp.int16)),
                jnp.arange(self.num_agents)
            )

            final_itxs = ys[-1]
            return final_itxs

        def _rand_interaction(
                key: int,
                conflicts: jnp.ndarray,
                conflicts_matrix: jnp.ndarray,
                step_arr: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function for randomly choosing between conflicting interactions.
            
            Args:
                - key: jax PRNGKey for randomisation.
                - conflicts: jnp.ndarray of bools where True if agent is in a
                conflicting interaction, False otherwise.
                - conflicts_matrix: jnp.ndarray matrix of bools of agents in
                conflicting interactions.
                - step_arr: jnp.ndarray, where each index is the index of an
                agent, and the element at each index is the item found at that
                agent's respective target location in the grid.

                
            Returns:
                - jnp.ndarray array of final interactions, where each index is
                an agent, and each element is caught in its interaction beam.
            '''
            conflict_idx = jnp.nonzero(
                conflicts,
                size=self.num_agents,
                fill_value=-1
            )[0][0]

            agent_conflicts = conflicts_matrix[conflict_idx]

            agent_conflicts_idx = jnp.nonzero(
                agent_conflicts,
                size=self.num_agents,
                fill_value=-1
            )[0]
            max_rand = jnp.sum(agent_conflicts_idx > -1)

            # preparing random agent selection
            k1, k2 = jax.random.split(key, 2)
            random_number = jax.random.randint(
                k1,
                (1,),
                0,
                max_rand
            )

            # index in main matrix of agent of successful interaction
            rand_agent_idx = agent_conflicts_idx[random_number]

            # set that agent's bool to False, for inversion later
            new_agent_conflict = agent_conflicts.at[rand_agent_idx].set(False)

            # set all remaining True agents' values as the "empty" item
            step_arr = jnp.where(
                new_agent_conflict,
                Items.empty,
                step_arr
            ).astype(jnp.int16)

            # update conflict bools to reflect the post-conflict state
            _conflicts = conflicts.at[agent_conflicts_idx].set(0)
            conflicts = jnp.where(
                agent_conflicts,
                0,
                conflicts
            )
            conflicts = conflicts.at[conflict_idx].set(0)
            conflicts_matrix = jax.vmap(
                lambda c, x: jnp.where(c, x, jnp.array([False]*conflicts.shape[0]))
            )(conflicts, conflicts_matrix)

            # deal with next conflict
            return ((
                k2,
                conflicts,
                conflicts_matrix,
                step_arr
            ), step_arr)
        
        def fix_interactions(
            key: jnp.ndarray,
            all_interacts: jnp.ndarray,
            actions: jnp.ndarray,
            state: State,
            one_step: jnp.ndarray,
            two_step: jnp.ndarray,
            right: jnp.ndarray,
            left: jnp.ndarray
        ) -> jnp.ndarray:
            '''
            Function defining multi-interaction logic.
            
            Args:
                - key: jax key for randomisation
                - all_interacts: jnp.ndarray of bools, provisional interaction
                indicators, subject to interaction logic
                - actions: jnp.ndarray of ints, actions taken by the agents
                - state: State, env state object
                - one_step, two_step, right, left: jnp.ndarrays, all of the
                same length, where each index is the index of an agent, and
                the element at each index is the item found at that agent's
                respective target location in the grid.

                
            Returns:
                - (jnp.ndarray, jnp.ndarray) - Tuple, where index 0 contains
                the array of the final agent interactions and index 1 contains
                a multi-dimensional array of the updated freeze penalty
                matrix.
            '''
            # distance-based priority, else random

            # fix highest priority interactions; 2 one-step zaps on the same
            # agent - random selection of winner
            forward = jnp.logical_and(
                actions == Actions.zap_forward,
                all_interacts
            )

            forward = jnp.where(
                forward,
                one_step,
                Items.empty
            )

            conflicts_matrix = check_interaction_conflict(forward) * (forward != Items.empty)[:, None]

            conflicts = jnp.clip(
                jnp.sum(conflicts_matrix, axis=-1),
                0,
                1
            )

            k1, k2, k3  = jax.random.split(key, 3)
            one_step = rand_interaction(
                k2,
                conflicts,
                conflicts_matrix,
                forward
            )

            # if an interaction 2 steps away (diagonally or straight ahead)
            # is simultaneously another agent's one-step interaction, clear
            # it for the agent that is further:

            # right targets
            _right = jnp.logical_and(
                actions == Actions.zap_right,
                all_interacts
            )

            _right = jnp.where(
                _right,
                right,
                Items.empty
            )

            right = jnp.where(
                jnp.logical_and(
                    _right == one_step,
                    jnp.isin(_right, self._agents)
                ),
                Items.empty,
                _right
            )

            # left targets
            _left = jnp.logical_and(
                actions == Actions.zap_left,
                all_interacts
            )

            _left = jnp.where(
                _left,
                left,
                Items.empty
            )

            left = jnp.where(
                jnp.logical_and(
                    _left == one_step,
                    jnp.isin(_left, self._agents)
                ),
                Items.empty,
                _left
            )

            # two step targets
            _ahead = jnp.logical_and(
                actions == Actions.zap_ahead,
                all_interacts
            )

            _ahead = jnp.where(
                _ahead,
                two_step,
                Items.empty
            )
            two_step = jnp.where(
                jnp.logical_and(
                    _ahead == one_step,
                    jnp.isin(_ahead, self._agents)
                ),
                Items.empty,
                _ahead
            )

            # if any interactions occur between equi-distant pairs of agents,
            # choose randomly:
            all_two_step = jnp.concatenate(
                [
                    jnp.expand_dims(two_step, axis=-1),
                    jnp.expand_dims(right, axis=-1),
                    jnp.expand_dims(left, axis=-1),
                ],
                axis=-1
            )
            same_dist = jnp.logical_and(
                jnp.isin(
                    actions,
                    jnp.array([
                        Actions.zap_ahead,
                        Actions.zap_right,
                        Actions.zap_left
                    ]),
                ),
                all_interacts
            )

            same_dist_targets = jax.vmap(
                lambda s, ats, a: jnp.where(
                    s,
                    ats[a-Actions.zap_ahead],
                    Items.empty
                ).astype(jnp.int16)
            )(same_dist, all_two_step, actions)

            two_step_conflicts_matrix = check_interaction_conflict(
                same_dist_targets
            ) * (same_dist_targets != Items.empty)[:, None]

            two_step_conflicts = jnp.clip(
                jnp.sum(two_step_conflicts_matrix, axis=-1),
                0,
                1
            )

            k1, k2  = jax.random.split(k3, 2)
            two_step_interacts = rand_interaction(
                k2,
                two_step_conflicts,
                two_step_conflicts_matrix,
                same_dist_targets
            )

            # combine one & two step interactions
            final_interactions = jnp.where(
                actions == Actions.zap_forward, # if action was a 1-step zap
                one_step, # use the one-step item list after it was filtered
                two_step_interacts # else use two-step item after filtering
            ) * (state.freeze.max(axis=-1) <= 0)
            
            # note, both arrays were
            # qualified for:
            # 1. having actually zapped,
            # 2. zapping an actual agent,
            # 3. the zapping agent being not-frozen,
            # 4. the zapping agent meets
            # the zapping inventory threshold,
            # 5. the target agent meets the inventory threshold,
            # 6. agents 2 steps away surrender to agents 1 step away from the
            # same target
            # 7. equidistant agents succeed randomly

            # if the zapped agent is frozen, disallow the interaction
            itx_bool = jnp.isin(final_interactions, self._agents)
            frzs = state.freeze.max(axis=-1)
            final_interactions = jax.vmap(lambda i, itx: jnp.where(
                    jnp.logical_and(
                        i, frzs[itx-len(Items)] > 0
                    ),
                    Items.empty,
                    itx
                ).astype(jnp.int16)
            )(itx_bool, final_interactions)

            # update freeze matrix
            new_freeze_idx_bool = jnp.isin(
                    final_interactions,
                    self._agents
            )

            new_freeze = jax.vmap(lambda i, i_t, itx: jnp.where(
                    i_t,
                    state.freeze[i].at[itx-len(Items)].set(freeze_penalty),
                    state.freeze[i]
                ).astype(jnp.int16)
            )(self._agents-len(Items), new_freeze_idx_bool, final_interactions)

            new_freeze = new_freeze.transpose()
            new_freeze = jax.vmap(lambda i, i_t, itx: jnp.where(
                    i_t,
                    new_freeze[i].at[itx-len(Items)].set(freeze_penalty),
                    new_freeze[i]
                ).astype(jnp.int16)
            )(self._agents-len(Items), new_freeze_idx_bool, final_interactions)

            return jnp.concatenate(
                [jnp.expand_dims(final_interactions, axis=0),
                new_freeze.transpose()],
                axis=0
            )

        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        def check_collision(
                new_agent_locs: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check agent collisions.
            
            Args:
                - new_agent_locs: jnp.ndarray, the agent locations at the 
                current time step.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.all(x[:2] == y[:2]),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(new_agent_locs, new_agent_locs)

            return collisions
        
        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        def check_interaction_conflict(
                items: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check conflicting interaction targets.
            
            Args:
                - items: jnp.ndarray, the agent itemss at the interaction
                targets.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.logical_and(
                    jnp.all(x == y),
                    jnp.isin(x, self._agents)
                ),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(items, items)

            return collisions

        def fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Function defining multi-collision logic.

            Args:
                - key: jax key for randomisation
                - collided_moved: jnp.ndarray, the agents which moved in the
                last time step and caused collisions.
                - collision_matrix: jnp.ndarray, the agents currently in
                collisions
                - agent_locs: jnp.ndarray, the agent locations at the previous
                time step.
                - new_agent_locs: jnp.ndarray, the agent locations at the
                current time step.

            Returns:
                - jnp.ndarray of the final positions after collisions are
                managed.
            """
            def scan_fn(
                    state,
                    idx
            ):
                key, collided_moved, collision_matrix, agent_locs, new_agent_locs = state

                return jax.lax.cond(
                    collided_moved[idx] > 0,
                    lambda: _fix_collisions(
                        key,
                        collided_moved,
                        collision_matrix,
                        agent_locs,
                        new_agent_locs
                    ),
                    lambda: (state, new_agent_locs)
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, collided_moved, collision_matrix, agent_locs, new_agent_locs),
                jnp.arange(self.num_agents)
            )

            final_locs = ys[-1]

            return final_locs

        def _fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> Tuple[Tuple, jnp.ndarray]:
            def select_random_true_index(key, array):
                # Calculate the cumulative sum of True values
                cumsum_array = jnp.cumsum(array)

                # Count the number of True values
                true_count = cumsum_array[-1]

                # Generate a random index in the range of the number of True
                # values
                rand_index = jax.random.randint(
                    key,
                    (1,),
                    0,
                    true_count
                )

                # Find the position of the random index within the cumulative
                # sum
                chosen_index = jnp.argmax(cumsum_array > rand_index)

                return chosen_index
            # Pick one from all who collided & moved
            colliders_idx = jnp.argmax(collided_moved)

            collisions = collision_matrix[colliders_idx]

            # Check whether any of collision participants didn't move
            collision_subjects = jnp.where(
                collisions,
                collided_moved,
                collisions
            )
            collision_mask = collisions == collision_subjects
            stayed = jnp.all(collision_mask)
            stayed_mask = jnp.logical_and(~stayed, ~collision_mask)
            stayed_idx = jnp.where(
                jnp.max(stayed_mask) > 0,
                jnp.argmax(stayed_mask),
                0
            )

            # Prepare random agent selection
            k1, k2 = jax.random.split(key, 2)
            rand_idx = select_random_true_index(k1, collisions)
            collisions_rand = collisions.at[rand_idx].set(False) # <<<< PROBLEM LINE        
            new_locs_rand = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_rand,
                agent_locs,
                new_agent_locs
            )

            collisions_stayed = jax.lax.select(
                jnp.max(stayed_mask) > 0,
                collisions.at[stayed_idx].set(False),
                collisions_rand
            )
            new_locs_stayed = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_stayed,
                agent_locs,
                new_agent_locs
            )

            # Choose between the two scenarios - revert positions if
            # non-mover exists, otherwise choose random agent if all moved
            new_agent_locs = jnp.where(
                stayed,
                new_locs_rand,
                new_locs_stayed
            )

            # Update move bools to reflect the post-collision positions
            collided_moved = jnp.clip(collided_moved - collisions, 0, 1)
            collision_matrix = collision_matrix.at[colliders_idx].set(
                [False] * collisions.shape[0]
            )
            return ((k2, collided_moved, collision_matrix, agent_locs, new_agent_locs), new_agent_locs)

        def to_dict(
                agent: int,
                obs: jnp.ndarray,
                agent_invs: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                inv_to_show: jnp.ndarray
            ) -> dict:
            '''
            Function to produce observation/state dictionaries.
            
            Args:
                - agent: int, number identifying agent
                - obs: jnp.ndarray, the combined grid observations for each
                agent
                - agent_invs: jnp.ndarray of current agents' inventories
                - agent_pickups: boolean indicators of interaction
                - inv_to_show: jnp.ndarray inventory to show to other agents
                
            Returns:
                - dictionary of full state observation.
            '''
            idx = agent - len(Items)
            state_dict = {
                "observation": obs,
                "inventory": {
                    "agent_inv": agent_invs,
                    "agent_pickups": agent_pickups,
                    "invs_to_show": jnp.delete(
                        inv_to_show,
                        idx,
                        assume_unique_indices=True
                    )
                }
            }

            return state_dict
        
        def combine_channels(
                grid: jnp.ndarray,
                agent: int,
                angles: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                state: State,
            ):
            '''
            Function to enforce symmetry in observations & generate final
            feature representation; current agent is permuted to first 
            position in the feature dimension.
            
            Args:
                - grid: jax ndarray of current agent's obs grid
                - agent: int, an index indicating current agent number
                - angles: jnp.ndarray of current agents' relative orientation
                in current agent's obs grid
                - agent_pickups: jnp.ndarray of agents able to interact
                - state: State, the env state obj
            Returns:
                - grid with current agent [x] permuted to 1st position (after
                the first 4 "Items" features, so 5th position overall) in the
                feature dimension, "other" [x] agent 2nd, angle [x, x, x, x]
                3rd, pick-up [x] bool 4th, inventory [x, x] 5th, frozen [x]
                6th, for a final obs grid of shape (5, 5, 14) (additional 4
                one-hot places for 5 possible items)
            '''
            def move_and_collapse(
                    x: jnp.ndarray,
                    angle: jnp.ndarray,
                ) -> jnp.ndarray:

                # get agent's one-hot
                agent_element = jnp.array([jnp.int8(x[agent])])

                # mask to check if any other agent exists there
                mask = x[len(Items)-1:] > 0

                # does an agent exist which is not the subject?
                other_agent = jnp.int8(
                    jnp.logical_and(
                        jnp.any(mask),
                        jnp.logical_not(
                            agent_element
                        )
                    )
                )

                # what is the class of the item in cell
                item_idx = jnp.where(
                    x,
                    size=1
                )[0]

                # check if agent is frozen and can observe inventories
                show_inv_bool = jnp.logical_and(
                        state.freeze[
                            agent-len(Items)
                        ].max(axis=-1) > 0,
                        item_idx >= len(Items)
                )

                show_inv_idxs = jnp.where(
                    state.freeze[agent],
                    size=12, # since, in a setting where simultaneous interac-
                    fill_value=-1 # -tions can happen, only a max of 12 can
                )[0] # happen at once (zap logic), regardless of pop size

                inv_to_show = jnp.where(
                    jnp.logical_or(
                        jnp.logical_and(
                            show_inv_bool,
                            jnp.isin(item_idx-len(Items), show_inv_idxs),
                        ),
                        agent_element
                    ),
                    state.agent_invs[item_idx - len(Items)],
                    jnp.array([0, 0], dtype=jnp.int8)
                )[0]

                # check if agent is not the subject & is frozen & therefore
                # not possible to interact with
                frozen = jnp.where(
                    other_agent,
                    state.freeze[
                        item_idx-len(Items)
                    ].max(axis=-1) > 0,
                    0
                )

                # get pickup/inv info
                pick_up_idx = jnp.where(
                    jnp.any(mask),
                    jnp.nonzero(mask, size=1)[0],
                    jnp.int8(-1)
                )
                picked_up = jnp.where(
                    pick_up_idx > -1,
                    agent_pickups[pick_up_idx],
                    jnp.int8(0)
                )

                # build extension
                extension = jnp.concatenate(
                    [
                        agent_element,
                        other_agent,
                        angle,
                        picked_up,
                        inv_to_show,
                        frozen
                    ],
                    axis=-1
                )

                # build final feature vector
                final_vec = jnp.concatenate(
                    [x[:len(Items)-1], extension],
                    axis=-1
                )

                return final_vec

            new_grid = jax.vmap(
                jax.vmap(
                    move_and_collapse
                )
            )(grid, angles)
            return new_grid
        
        def check_relative_orientation(
                agent: int,
                agent_locs: jnp.ndarray,
                grid: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Check's relative orientations of all other agents in view of
            current agent.
            
            Args:
                - agent: int, an index indicating current agent number
                - agent_locs: jax ndarray of agent locations (x, y, direction)
                - grid: jax ndarray of current agent's obs grid
                
            Returns:
                - grid with 1) int -1 in places where no agent exists, or
                where the agent is the current agent, and 2) int in range
                0-3 in cells of opposing agents indicating relative
                orientation to current agent.
            '''
            # we decrement by num of Items when indexing as we incremented by
            # 5 in constructor call (due to 5 non-agent Items enum & locations
            # are indexed from 0)
            idx = agent - len(Items)
            agents = jnp.delete(
                self._agents,
                idx,
                assume_unique_indices=True
            )
            curr_agent_dir = agent_locs[idx, 2]

            def calc_relative_direction(cell):
                cell_agent = cell - len(Items)
                cell_direction = agent_locs[cell_agent, 2]
                return (cell_direction - curr_agent_dir) % 4

            angle = jnp.where(
                jnp.isin(grid, agents),
                jax.vmap(calc_relative_direction)(grid),
                -1
            )

            return angle
        
        def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
            '''
            Rotates agent's observation grid k * 90 degrees, depending on agent's
            orientation.

            Args:
                - agent_loc: jax ndarray of agent's x, y, direction
                - grid: jax ndarray of agent's obs grid

            Returns:
                - jnp.ndarray of new rotated grid.

            '''
            grid = jnp.where(
                agent_loc[2] == 1,
                jnp.rot90(grid, k=1, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 2,
                jnp.rot90(grid, k=2, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 3,
                jnp.rot90(grid, k=3, axes=(0, 1)),
                grid,
            )

            return grid

        def _get_obs_point(agent_loc: jnp.ndarray) -> jnp.ndarray:
            '''
            Obtain the position of top-left corner of obs map using
            agent's current location & orientation.

            Args: 
                - agent_loc: jnp.ndarray, agent x, y, direction.
            Returns:
                - x, y: ints of top-left corner of agent's obs map.
            '''
            
            x, y, direction = agent_loc

            x, y = x + self.PADDING, y + self.PADDING
            x = jnp.where(direction == 0, x - (self.OBS_SIZE // 2), x)
            x = jnp.where(direction == 2, x - (self.OBS_SIZE // 2), x)
            x = jnp.where(direction == 3, x - (self.OBS_SIZE - 1), x)

            y = jnp.where(direction == 1, y - (self.OBS_SIZE // 2), y)
            y = jnp.where(direction == 2, y - (self.OBS_SIZE - 1), y)
            y = jnp.where(direction == 3, y - (self.OBS_SIZE // 2), y)
            return x, y

        def _get_obs(state: State) -> jnp.ndarray:
            '''
            Obtain the agent's observation of the grid.

            Args: 
                - state: State object containing env state.
            Returns:
                - jnp.ndarray of grid observation.
            '''
            # create state
            grid = jnp.pad(
                state.grid,
                ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
                constant_values=Items.wall,
            )

            # obtain all agent obs-points
            agent_start_idxs = jax.vmap(_get_obs_point)(state.agent_locs)

            dynamic_slice = partial(
                jax.lax.dynamic_slice,
                operand=grid,
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE)
            )

            # obtain agent obs grids
            grids = jax.vmap(dynamic_slice)(start_indices=agent_start_idxs)

            # rotate agent obs grids
            grids = jax.vmap(rotate_grid)(state.agent_locs, grids)

            angles = jax.vmap(
                check_relative_orientation,
                in_axes=(0, None, 0)
            )(
                self._agents,
                state.agent_locs,
                grids
            )

            angles = jax.nn.one_hot(angles, 4)

            # one-hot (drop first channel as its empty blocks)
            grids = jax.nn.one_hot(
                grids - 1,
                num_agents + len(Items) - 1, # will be collapsed into a
                dtype=jnp.int8 # [Items, self, other, extra features] representation
            )

            # check agents that can interact
            inventory_sum = jnp.sum(state.agent_invs, axis=-1)
            agent_pickups = jnp.where(
                inventory_sum > INTERACT_THRESHOLD,
                True,
                False
            )

            # make index len(Item) always the current agent
            # and sum all others into an "other" agent
            grids = jax.vmap(
                combine_channels,
                in_axes=(0, 0, 0, None, None)
            )(
                grids,
                self._agents,
                angles,
                agent_pickups,
                state
            )

            return grids

        def _get_reward(
                state: State,
                agent1: int,
                agent2: int
            ) -> jnp.ndarray:
            '''
            Obtain the reward for a matrix game.

            Args: 
                - state: State object of env.
                - agent1: int of position of first agent in interaction
                - agent2: int of position of first agent in interaction
            Returns:
                - jnp.ndarray of rewards of agents.
            '''
            # calc proportion of each coin type
            inv_sum = jnp.where(
                state.agent_invs.sum(axis=-1) > 0,
                state.agent_invs.sum(axis=-1),
                1)
            invs = state.agent_invs / jnp.expand_dims(
                inv_sum,
                axis=-1
            )

            # get agent inventory proportions
            inv1 = invs[agent1]
            inv2 = invs[agent2]

            # obtain rewards via payoff matrices
            r1 = inv1 @ jnp.array(payoff_matrix[0]) @ inv2.T
            r2 = inv1 @ jnp.array(payoff_matrix[1]) @ inv2.T

            return jnp.array([r1, r2])

        def _interact(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, State, jnp.ndarray]:
            '''
            Main interaction logic entry point.

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
            Returns:
                - (jnp.ndarray, State, jnp.ndarray) - Tuple where index 0 is
                the array of rewards obtained, index 2 is the new env State,
                and index 3 is the new freeze penalty matrix.
            '''
            # if interact
            zaps = jnp.isin(actions,
                jnp.array(
                    [
                        Actions.zap_forward,
                        Actions.zap_ahead,
                        Actions.zap_right,
                        Actions.zap_left
                    ]
                )
            )

            interact_idx = jnp.int16(Items.interact)

            # remove old interacts
            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int16(Items.empty), state.grid
            ))

            # calculate pickups
            agent_pickups = state.agent_invs.sum(axis=-1) > INTERACT_THRESHOLD

            def check_valid_zap(t, i):
                '''
                Check target agent exists, isn't frozen, can interact, and
                is not the zapping-agent's self.
                '''
                agnt_bool = jnp.isin(state.grid[t[0], t[1]], self._agents)
                self_bool = state.grid[t[0], t[1]] != i
                frz_bool = jnp.where(
                    agnt_bool,
                    state.freeze[
                        state.grid[t[0], t[1]]-len(Items)
                    ].max(axis=-1) <= 0,
                    False
                )
                interact_bool = agent_pickups[
                    state.grid[t[0], t[1]]-len(Items)
                ].sum() > INTERACT_THRESHOLD
                return jnp.logical_and(
                    jnp.logical_and(
                        jnp.logical_and(agnt_bool, frz_bool),
                        interact_bool
                    ),
                    self_bool
                )

            # check 1 ahead
            clip = partial(jnp.clip, min=0, max=self.GRID_SIZE - 1)
            one_step_targets = jax.vmap(
                lambda p: p + STEP[p[2]]
            )(state.agent_locs)

            one_step_targets = jax.vmap(clip)(one_step_targets)

            one_step_interacts = jax.vmap(
                check_valid_zap
            )(t=one_step_targets, i=self._agents)

            # check 2 ahead
            two_step_targets = jax.vmap(
                lambda p: p + 2*STEP[p[2]]
            )(state.agent_locs)

            two_step_targets = jax.vmap(clip)(two_step_targets)

            two_step_interacts = jax.vmap(
                check_valid_zap
            )(t=two_step_targets, i=self._agents)

            # check forward-right & manually check out-of-bounds
            target_right = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] + 1) % 4]
            )(state.agent_locs)

            right_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    (t > self.GRID_SIZE - 1).any(),
                    (t < 0).any(),
                )
            )(target_right)

            target_right = jnp.where(
                right_oob_check[:, None],
                one_step_targets,
                target_right
            )

            right_interacts = jax.vmap(
                check_valid_zap
            )(t=target_right, i=self._agents)

            # check forward-left & manually check out-of-bounds
            target_left = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] - 1) % 4]
            )(state.agent_locs)

            left_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    (t > self.GRID_SIZE - 1).any(),
                    (t < 0).any(),
                )
            )(target_left)

            target_left = jnp.where(
                left_oob_check[:, None],
                one_step_targets,
                target_left
            )

            left_interacts = jax.vmap(
                check_valid_zap
            )(t=target_right, i=self._agents)

            # all interacts = whether there is an agent at target & whether
            # agent is not frozen already & is qualified to interact
            all_interacts = jnp.concatenate(
                [
                    jnp.expand_dims(one_step_interacts, axis=-1),
                    jnp.expand_dims(two_step_interacts, axis=-1),
                    jnp.expand_dims(right_interacts, axis=-1),
                    jnp.expand_dims(left_interacts, axis=-1)
                ],
                axis=-1
            )

            # clipping all interactions to between 0 and 3 to index the correct
            # interaction bool from all_interacts
            interact_idxs = jnp.clip(
                actions - Actions.zap_forward,
                0,
                Actions.zap_left - Actions.zap_forward
            )
            all_interacts = all_interacts[
                jnp.arange(all_interacts.shape[0]),
                interact_idxs
            ] * zaps * agent_pickups * (state.freeze.max(axis=-1) <= 0)

            # update grid with all zaps
            aux_grid = jnp.copy(state.grid)

            o_items = jnp.where(
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            t_items = jnp.where(
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            r_items = jnp.where(
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        interact_idx
                    )

            l_items = jnp.where(
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        interact_idx
                    )

            # rectified list of items at each agent's target, new freeze
            # penalties
            it_fr = jnp.concatenate(
                [jnp.zeros((1, num_agents), dtype=jnp.int16), state.freeze],
                axis=0
            )

            it_fr = jnp.where(
                all_interacts.any(),
                fix_interactions(
                    key,
                    all_interacts,
                    actions,
                    state,
                    o_items,
                    t_items,
                    r_items,
                    l_items
                ),
                it_fr
            )
            items, new_freezes = it_fr[0], it_fr[1:]

            qualified_to_zap = zaps * agent_pickups * (
                state.freeze.max(axis=-1) <= 0
            )

            # update grid
            def update_grid(a_i, t, i, grid):
                return grid.at[t[:, 0], t[:, 1]].set(
                    jax.vmap(jnp.where)(
                        a_i,
                        i,
                        aux_grid[t[:, 0], t[:, 1]]
                    )
                )

            aux_grid = update_grid(qualified_to_zap, one_step_targets, o_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, two_step_targets, t_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_right, r_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_left, l_items, aux_grid)

            state = state.replace(
                grid=jnp.where(
                    jnp.any(zaps),
                    aux_grid,
                    state.grid
                )
            )

            # rewards
            _rewards = jax.vmap(
                jax.vmap(
                    _get_reward,
                    in_axes=(None, 0, None)
                ),
                in_axes=(None, None, 0)
            )(state, self._agents-len(Items), self._agents-len(Items))

            interact = jnp.isin(items, self._agents)
            rewards = jax.vmap(
                lambda itx, a, i: jnp.where(
                    itx,
                    _rewards[a-len(Items), i-len(Items)],
                    jnp.array([0, 0], dtype=jnp.int16)
                )
            )(interact, self._agents, items)

            # allocate rewards to the "zappers"
            zapper_rewards = jax.vmap(
                lambda itx, i: jnp.where(
                    itx,
                    rewards[i-len(Items)][0],
                    0
                )
            )(interact, items)

            # allocate rewards to the "zappees", allowing for "chained" zaps
            # (agent 1 zaps agent 2, agent 2 zaps agent 3 in the same time
            # step, so agent 2 gets reward from agent 1 & agent 3)

            # allocate the rewards of those who got zapped
            zappee_rewards = jax.vmap(
                lambda itx, idx, r: jnp.where(
                    itx,
                    zapper_rewards[idx-len(Items)] + r[1],
                    0
                )
            )(interact, items, rewards)

            # replace the rewards of those who didn't get zapped with the
            # rewards of agent 0's reward
            zappee_rewards = jax.vmap(
                lambda itx, r: jnp.where(
                    itx,
                    r,
                    zappee_rewards[0]
                )
            )(interact, zappee_rewards)

            # convert all zapped agents to their idxs, replacing non-agent
            # items to agent 0's index (it acts as a pad for the array)
            item_idxs = jnp.clip(items - len(Items), 0)

            # allocate all final rewards to appropriate agents
            rewards = zapper_rewards.at[item_idxs].set(zappee_rewards)

            return rewards, state, new_freezes

        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: jnp.ndarray
        ):
            """Step the environment."""
            actions = jnp.array(actions)

            # freeze check
            actions = jnp.where(
                state.freeze.max(axis=-1) > 0,
                Actions.stay,
                actions
            )

            # moving all agents
            all_new_locs = jax.vmap(
                lambda p, a: jnp.int16(p + ROTATIONS[a])
                % jnp.array(
                    [self.GRID_SIZE + 1, self.GRID_SIZE + 1, 4],
                    dtype=jnp.int16
                )
            )(p=state.agent_locs, a=actions)

            agent_move = actions == Actions.forward
            all_new_locs = jax.vmap(
                lambda m, n, p: jnp.where(m, n + STEP[p[2]], n)
            )(
                m=agent_move,
                n=all_new_locs,
                p=state.agent_locs
            )
            all_new_locs = jax.vmap(
                jnp.clip,
                in_axes=(0, None, None)
            )(
                all_new_locs,
                jnp.array([0, 0, 0], dtype=jnp.int16),
                jnp.array(
                    [self.GRID_SIZE - 1, self.GRID_SIZE - 1, 3],
                    dtype=jnp.int16
                ),
            )

            # if you bounced back to your original space,
            # change your move to stay (for collision logic)
            agents_move = jax.vmap(
                lambda n, p: jnp.any(n[:2] != p[:2])
            )(n=all_new_locs, p=state.agent_locs)

            # generate bool mask for agents colliding
            collision_matrix = check_collision(all_new_locs)

            # sum & subtract "self-collisions"
            collisions = jnp.sum(
                collision_matrix,
                axis=-1,
                dtype=jnp.int8
            ) - 1
            collisions = jnp.minimum(collisions, 1)

            # identify which of those agents made wrong moves
            collided_moved = jnp.maximum(
                collisions - ~agents_move,
                0
            )

            # fix collisions at the correct indices
            new_locs = jax.lax.cond(
                jnp.max(collided_moved) > 0,
                lambda: fix_collisions(
                    key,
                    collided_moved,
                    collision_matrix,
                    state.agent_locs,
                    all_new_locs
                ),
                lambda: all_new_locs
            )

            # update inventories
            def coin_matcher(p: jnp.ndarray) -> jnp.ndarray:
                c_matches = jnp.array([
                    state.grid[p[0], p[1]] == Items.red_coin,
                    state.grid[p[0], p[1]] == Items.blue_coin
                    ])
                return c_matches

            coin_matches = jax.vmap(coin_matcher)(p=new_locs)
            new_invs = state.agent_invs + coin_matches

            state = state.replace(
                agent_invs=new_invs
            )

            # update grid
            old_grid = state.grid

            new_grid = old_grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )
            x, y = new_locs[:, 0], new_locs[:, 1]
            new_grid = new_grid.at[x, y].set(self._agents)
            state = state.replace(grid=new_grid)

            # update agent locations
            state = state.replace(agent_locs=new_locs)

            (rewards, state, new_freezes) = _interact(key, state, actions)

            old_freeze = state.freeze

            # if we interacted, then we set freeze
            state = state.replace(freeze=new_freezes)

            decremented_freeze = jnp.where(
                jnp.logical_and(
                    new_freezes <= old_freeze,
                    new_freezes > 0
                ),
                state.freeze - 1,
                state.freeze
            )

            # if we didn't interact, then we decrement freeze
            state = state.replace(
                freeze=decremented_freeze
            )

            # soft reset for anyone who has just been mobilised
            state_sft_re = _soft_reset_state(key, state, old_freeze)
            state = jax.tree.map(
                lambda x, y: jax.lax.select(
                    jnp.any(
                        jnp.logical_and(
                            state.freeze == 0,
                            old_freeze == 1
                        )
                    ),
                    x,
                    y
                ),
                state_sft_re,
                state,
            )

            state_nxt = State(
                agent_locs=state.agent_locs,
                agent_invs=state.agent_invs,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                red_coins=state.red_coins,
                blue_coins=state.blue_coins,
                freeze=state.freeze,
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # if inner episode is done, return start state for next game
            state_re = _reset_state(key)

            state_re = state_re.replace(outer_t=outer_t + 1)
            state = jax.tree.map(
                lambda x, y: jnp.where(reset_inner, x, y),
                state_re,
                state_nxt,
            )
            outer_t = state.outer_t
            reset_outer = outer_t == num_outer_steps
            done = {f'agent_{a}': reset_outer for a in self.agents}
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            rewards = jnp.where(
                reset_inner,
                jnp.zeros_like(rewards, dtype=jnp.int16),
                rewards
            )

            mean_inv = state.agent_invs.mean(axis=0)
            return (
                obs,
                state,
                rewards,
                done,
                {
                    "discount": jnp.zeros((), dtype=jnp.int8),
                    "coin_ratio": jnp.clip(
                        mean_inv[0] / (mean_inv.sum() + 1E-9),
                        0.0,
                        1.0
                    )
                },
            )

        def _soft_reset_state(
                key: jnp.ndarray,
                state: State,
                old_freeze: jnp.ndarray
            ) -> State:
            '''
            Soft reset the simulation - agents who have just become mobile
            again, after an interaction, are respawned randomly. Consumed
            coins are also respawned.

            Args:
                - key:jnp.ndarray, key for randomisation
                - state: State, the updated state
                - old_freeze: jnp.ndarray, the array of freeze per-agent freeze
                penalties.

            Returns:
                - The new state, with appropriate elements reset.
            '''
            key, subkey1, subkey2 = jax.random.split(key, 3)

            # determine which agents and coins to respawn
            agents_to_respawn = jnp.logical_and(state.freeze == 0, old_freeze == 1).any(axis=-1)
            red_coins_to_respawn = state.grid[state.red_coins[:, 0], state.red_coins[:, 1]] != Items.red_coin
            blue_coins_to_respawn = state.grid[state.blue_coins[:, 0], state.blue_coins[:, 1]] != Items.blue_coin

            # create mask of all occupied positions
            occupied_mask = jnp.zeros(grid_size * grid_size, dtype=bool)
            occupied_indices = state.agent_locs[:, 0] * grid_size + state.agent_locs[:, 1]
            occupied_mask = occupied_mask.at[occupied_indices].set(~agents_to_respawn)
            occupied_indices = state.red_coins[:, 0] * grid_size + state.red_coins[:, 1]
            occupied_mask = occupied_mask.at[occupied_indices].set(~red_coins_to_respawn)
            occupied_indices = state.blue_coins[:, 0] * grid_size + state.blue_coins[:, 1]
            occupied_mask = occupied_mask.at[occupied_indices].set(~blue_coins_to_respawn)

            # shuffle all positions
            shuffled_positions = jax.random.permutation(subkey1, self.SPAWNS)

            # use the shuffled positions to respawn, skipping occupied ones
            new_agent_positions = jnp.where(
                agents_to_respawn[:, None],
                jnp.array([(-1, -1, -1)]*self.num_agents, dtype=jnp.int16),
                state.agent_locs
            )
            new_red_coin_positions = jnp.where(
                red_coins_to_respawn[:, None],
                jnp.array([(-1, -1)]*self.NUM_COINS, dtype=jnp.int16),
                state.red_coins
            )
            new_blue_coin_positions = jnp.where(
                blue_coins_to_respawn[:, None],
                jnp.array([(-1, -1)]*self.NUM_COINS, dtype=jnp.int16),
                state.blue_coins
            )

            def update_positions(carry, x):
                new_a_pos, new_r_pos, new_b_pos, mask, agent_idx, red_idx, blue_idx = carry
                pos = shuffled_positions[x]
                is_occupied = mask[x]
                
                # agent update
                agent_needs_respawn = jnp.logical_and(agent_idx < self.num_agents, agents_to_respawn[agent_idx])
                agent_update = jnp.logical_and(agent_needs_respawn, ~is_occupied)
                new_agent_positions = jnp.where(
                    agent_update,
                    new_a_pos.at[agent_idx].set(pos),
                    new_a_pos
                )
                agent_idx += jnp.logical_or(agent_update, ~agent_needs_respawn)
                is_occupied = jnp.logical_or(is_occupied, agent_update)

                # red coin update
                red_needs_respawn = jnp.logical_and(red_idx < self.NUM_COINS, red_coins_to_respawn[red_idx])
                red_update = jnp.logical_and(red_needs_respawn, ~is_occupied)
                new_red_coin_positions = jnp.where(
                    red_update,
                    new_r_pos.at[red_idx].set(pos),
                    new_r_pos
                )
                red_idx += jnp.logical_or(red_update, ~red_needs_respawn)
                is_occupied = jnp.logical_or(is_occupied, red_update)

                # blue coin update
                blue_needs_respawn = jnp.logical_and(blue_idx < self.NUM_COINS, blue_coins_to_respawn[blue_idx])
                blue_update = jnp.logical_and(blue_needs_respawn, ~is_occupied)
                new_blue_coin_positions = jnp.where(
                    blue_update,
                    new_b_pos.at[blue_idx].set(pos),
                    new_b_pos
                )
                blue_idx += jnp.logical_or(blue_update, ~blue_needs_respawn)

                # update occupied mask
                new_mask = mask.at[x].set(is_occupied)

                return ((new_agent_positions, new_red_coin_positions, new_blue_coin_positions, new_mask, agent_idx, red_idx, blue_idx), None)

            init_carry = (new_agent_positions[:, :2], new_red_coin_positions, new_blue_coin_positions, occupied_mask, 0, 0, 0)
            ((new_agent_positions, new_red_coin_positions, new_blue_coin_positions, _, _, _, _), _) = jax.lax.cond(
                agents_to_respawn.any(),
                lambda: jax.lax.scan(
                    update_positions,
                    init_carry,
                    jnp.arange(grid_size * grid_size),
                    unroll=8
                ),
                lambda: (init_carry, None)
            )

            # set unfrozen agent positions to empty
            grid = jnp.where(
                agents_to_respawn.any(),
                state.grid.at[
                    state.agent_locs[:, 0],
                    state.agent_locs[:, 1]
                ].set(jnp.int16(Items.empty)),
                state.grid
            )

            grid = jnp.where(
                agents_to_respawn.any(),
                grid.at[
                    new_agent_positions[:, 0],
                    new_agent_positions[:, 1]
                ].set(jnp.int16(self._agents)),
                state.grid
            )

            # set coin positions in grid
            grid = jnp.where(
                jnp.logical_and(agents_to_respawn.any(), red_coins_to_respawn.any()),
                grid.at[
                    new_red_coin_positions[:, 0],
                    new_red_coin_positions[:, 1]
                ].set(jnp.int16(Items.red_coin)),
                grid
            )
            grid = jnp.where(
                jnp.logical_and(agents_to_respawn.any(), blue_coins_to_respawn.any()),
                grid.at[
                    new_blue_coin_positions[:, 0],
                    new_blue_coin_positions[:, 1]
                ].set(jnp.int16(Items.blue_coin)),
                grid
            )

            # set agent locs & dirs
            player_dir = jnp.where(
                agents_to_respawn.any(),
                jax.random.randint(
                    subkey2, shape=(
                        self.num_agents,
                        ), minval=0, maxval=3, dtype=jnp.int8
                ),
                False
            )
            player_dir = jnp.where(
                agents_to_respawn,
                player_dir,
                state.agent_locs[:, 3]
            )

            agent_locs = jnp.where(
                agents_to_respawn.any(),
                jnp.array(
                    [new_agent_positions[:, 0], new_agent_positions[:, 1], player_dir],
                    dtype=jnp.int16
                ).T,
                jnp.array([-1, -1, -1], jnp.int8)
            )

            # new full agent loc list
            new_agent_locs = jnp.where(
                agents_to_respawn.any(),
                agent_locs,
                state.agent_locs
            )

            # adjusted full invs list
            new_invs = jnp.where(
                agents_to_respawn[:, None],
                jnp.array(
                    [(0,0)]*self.num_agents,
                    dtype=jnp.int8
                ),
                state.agent_invs
            )

            return State(
                agent_locs=new_agent_locs,
                agent_invs=new_invs,
                inner_t=state.inner_t,
                outer_t=state.outer_t,
                grid=grid,
                red_coins=new_red_coin_positions,
                blue_coins=new_blue_coin_positions,
                freeze=state.freeze,
            )

        def _reset_state(
            key: jnp.ndarray
        ) -> State:
            key, subkey = jax.random.split(key)

            # Find the free spaces in the grid
            grid = jnp.zeros((self.GRID_SIZE, self.GRID_SIZE), jnp.int16)

            all_positions = jax.random.permutation(subkey, self.SPAWNS)
            total_items = num_agents + NUM_COIN_TYPES * self.NUM_COINS
            
            agent_pos = all_positions[:num_agents]
            coin_pos = all_positions[num_agents:total_items]

            red_coin_pos = coin_pos[:self.NUM_COINS]
            blue_coin_pos = coin_pos[self.NUM_COINS:]

            grid = grid.at[
                red_coin_pos[:, 0],
                red_coin_pos[:, 1]
            ].set(jnp.int16(Items.red_coin))
            grid = grid.at[
                blue_coin_pos[:, 0],
                blue_coin_pos[:, 1]
            ].set(jnp.int16(Items.blue_coin))

            player_dir = jax.random.randint(
                subkey, shape=(
                    num_agents,
                    ), minval=0, maxval=3, dtype=jnp.int8
            )

            agent_locs = jnp.array(
                [agent_pos[:, 0], agent_pos[:, 1], player_dir],
                dtype=jnp.int16
            ).T

            grid = grid.at[
                agent_locs[:, 0],
                agent_locs[:, 1]
            ].set(jnp.int16(self._agents))

            freeze = jnp.array(
                [[-1]*num_agents]*num_agents,
            dtype=jnp.int16
            )
            return State(
                agent_locs=agent_locs,
                agent_invs=jnp.array([(0,0)]*num_agents, dtype=jnp.int8),
                inner_t=0,
                outer_t=0,
                grid=grid,
                red_coins=red_coin_pos,
                blue_coins=blue_coin_pos,
                freeze=freeze,
            )

        def reset(
            key: jnp.ndarray
        ) -> Tuple[jnp.ndarray, State]:
            state = _reset_state(key)
            obs = _get_obs(state)
            return obs, state

        # overwrite Gymnax as it makes single-agent assumptions
        # self.step = jax.jit(_step)
        self.step_env = jax.jit(_step)
        self.reset = jax.jit(reset)
        self.get_obs_point = jax.jit(_get_obs_point)
        self.get_reward = jax.jit(_get_reward)

        # for debugging
        self.get_obs = jax.jit(_get_obs)
        self.cnn = cnn

        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

    @property
    def name(self) -> str:
        """Environment name."""
        return "MGinTheGrid"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Actions)

    def action_space(
        self, agent_id: Union[int, None] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        _shape_obs = (
            (self.OBS_SIZE, self.OBS_SIZE, (len(Items)-1) + 10)
            if self.cnn
            else (self.OBS_SIZE**2 * ((len(Items)-1) + 10),)
        )

        return spaces.Box(
                low=0, high=1E9, shape=_shape_obs, dtype=jnp.uint8
            ), _shape_obs
    
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        _shape = (
            (self.GRID_SIZE, self.GRID_SIZE, NUM_TYPES + 4)
            if self.cnn
            else (self.GRID_SIZE**2 * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)
    
    def render_tile(
        self,
        obj: int,
        agent_dir: Union[int, None] = None,
        agent_hat: bool = False,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, agent_hat, highlight, tile_size)
        if obj:
            key = (obj, 0, 0, 0) + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = onp.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3),
            dtype=onp.uint8,
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # class Items(IntEnum):

        if obj in self._agents:
            # Draw the agent
            agent_color = self.PLAYER_COLOURS[obj-len(Items)]
        elif obj == Items.red_coin:
            # Draw the red coin as GREEN COOPERATE
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (44.0, 160.0, 44.0)
            )
        elif obj == Items.blue_coin:
            # Draw the blue coin as DEFECT/ RED COIN
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
            )
        elif obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))

        elif obj == Items.interact:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))

        elif obj == 99:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))

        elif obj == 100:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))

        elif obj == 101:
            # white square
            fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))

        # Overlay the agent on top
        if agent_dir is not None:
            if agent_hat:
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                    0.3,
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(
                    tri_fn,
                    cx=0.5,
                    cy=0.5,
                    theta=0.5 * math.pi * (1 - agent_dir),
                )
                fill_coords(img, tri_fn, (255.0, 255.0, 255.0))

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
                0.0,
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (1 - agent_dir)
            )
            fill_coords(img, tri_fn, agent_color)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img
        return img

    def render(
        self,
        state: State,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros_like(onp.array(self.GRID))

        # Compute the total grid size
        width_px = self.GRID.shape[0] * tile_size
        height_px = self.GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        grid = onp.array(state.grid)
        grid = onp.pad(
            grid, ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)), constant_values=Items.wall
        )
        for a in range(self.num_agents):
            startx, starty = self.get_obs_point(
                state.agent_locs[a]
            )
            highlight_mask[
                startx : startx + self.OBS_SIZE, starty : starty + self.OBS_SIZE
            ] = True

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None
                agent_here = []
                for a in self._agents:
                    agent_here.append(cell == a)
                # if cell in [1,2]:
                #     print(f'coordinates: {i},{j}')
                #     print(cell)

                agent_dir = None
                for a in range(self.num_agents):
                    agent_dir = (
                        state.agent_locs[a,2].item()
                        if agent_here[a]
                        else agent_dir
                    )
                
                agent_hat = False
                for a in range(self.num_agents):
                    agent_hat = (
                        bool(state.agent_invs[a].sum() > INTERACT_THRESHOLD)
                        if agent_here[a]
                        else agent_hat
                    )

                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    agent_hat=agent_hat,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        img = onp.rot90(
            img[
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                :,
            ],
            2,
        )
        # Render the inventory
        # agent_inv = []
        # for a in range(self.num_agents):
        #     agent_inv.append(self.render_inventory(state.agent_inventories[a], img.shape[1]))


        time = self.render_time(state, img.shape[1])
        # img = onp.concatenate((img, *agent_inv, time), axis=0)
        img = onp.concatenate((img, time), axis=0)
        return img

    def render_inventory(self, inventory, width_px) -> onp.array:
        tile_height = 32
        height_px = NUM_COIN_TYPES * tile_height
        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // self.NUM_COINS
        for j in range(0, NUM_COIN_TYPES):
            num_coins = inventory[j]
            for i in range(int(num_coins)):
                cell = None
                if j == 0:
                    cell = 99
                elif j == 1:
                    cell = 100
                tile_img = self.render_tile(cell, tile_size=tile_height)
                ymin = j * tile_height
                ymax = (j + 1) * tile_height
                xmin = i * tile_width
                xmax = (i + 1) * tile_width
                img[ymin:ymax, xmin:xmax, :] = onp.resize(
                    tile_img, (tile_height, tile_width, 3)
                )
        return img

    def render_time(self, state, width_px) -> onp.array:
        inner_t = state.inner_t
        outer_t = state.outer_t
        tile_height = 32
        img = onp.zeros(shape=(2 * tile_height, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // (self.num_inner_steps)
        j = 0
        for i in range(0, inner_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        tile_width = width_px // (self.num_outer_steps)
        j = 1
        for i in range(0, outer_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        return img