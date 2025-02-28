import jax 
import jax.numpy as jnp
from functools import partial
import os
import pickle
from typing import Tuple, List
from .map import Map

import numpy as np
import chex
from enum import IntEnum
import matplotlib.axes._axes as axes

import jaxmarl.environments.jaxnav.jaxnav_graph_utils as _graph_utils

def rotation_matrix(theta: float) -> jnp.ndarray:
    """ Rotate about the z axis. Assume theta in radians """
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta)]
    ])

class SampleTestCaseTypes(IntEnum):
    RANDOM = 0
    GRID = 1

class GridMapCircleAgents(Map):
    
    def __init__(self,
                 num_agents: int,
                 rad,
                 map_size: Tuple[int, int],
                 fill: float=0.4,
                 cell_size: float=1.0,
                 sample_test_case_type='random',
                 **map_kwargs):
        super().__init__(num_agents, rad, map_size, **map_kwargs)
        assert self.rad < 1  # collision code only works for radius <1
        
        self.width = map_size[0]
        self.height = map_size[1]
        self.length = self.width*self.height
        self.pos_offset = jnp.full((2,), 0.5)
        
        self.cell_size = cell_size
        self.scaled_rad = self.scale_coords(rad)
        self.circle_check_window = jnp.ceil(self.scaled_rad).astype(jnp.int32)
        idxs = jnp.arange(-self.circle_check_window-1, self.circle_check_window+1)
        self.cc_x_idx, self.cc_y_idx = jnp.meshgrid(idxs, idxs)
        self.cell_half_height = self.cell_size / 2
        
        # determine max number of blocks
        if sample_test_case_type == 'random':
            self.sample_test_case_type = SampleTestCaseTypes.RANDOM
        elif sample_test_case_type == 'grid':
            self.sample_test_case_type = SampleTestCaseTypes.GRID
        else:
            raise ValueError(f"Invalid sample_test_case_type: {sample_test_case_type}")
        self.fill = fill
        self.free_grids = (self.width-2)*(self.height-2)
        self.n_clutter = jnp.floor(self.free_grids*self.fill).astype(int)
    
    @partial(jax.jit, static_argnums=[0])
    def sample_test_case(self, rng: chex.PRNGKey):
        
        return jax.lax.switch(
            self.sample_test_case_type,
            [
                super().sample_test_case,
                self.grid_sample_test_case,
            ],
            rng
        )
        
    def grid_sample_test_case(self, key):
        """ NOTE this won't throw an error if it's not possible, will just loop forever"""
        assert self.cell_size == 1.0
        
        key, _key = jax.random.split(key)
        map_data = self.sample_map(_key)
        inside_grid = map_data.at[1:-1, 1:-1].get()
        iwidth = self.width - 2
        
        def _sample_pair(key, start_masks, goal_masks):
            
            flat_occ = start_masks.flatten()
            key, _key = jax.random.split(key)
            start_idx = jax.random.choice(_key, len(flat_occ), (1,), p=(1-flat_occ))[0]
            start = jnp.array([start_idx % iwidth, start_idx // iwidth])  # [x, y]
            actual_idx = (start + 1).astype(jnp.int32)
            # connected_region = _graph_utils.component_mask_with_pos(inside_grid, start_idx)  # BUG not working on inside grid
            if self.valid_path_check:
                connected_region = _graph_utils.component_mask_with_pos(map_data, actual_idx).at[1:-1, 1:-1].get()
            else:
                connected_region = 1-inside_grid
            masked_start = connected_region.at[start[1], start[0]].set(0)
            goal_possibilities = masked_start & (1 - goal_masks)
            valid = jnp.any(goal_possibilities)  # only valid if possible goal locations
                        
            goal_idx = jax.random.choice(key, len(flat_occ), (1,), p=goal_possibilities.flatten())[0]
            goal = jnp.array([goal_idx % iwidth, goal_idx // iwidth])  # [x, y]
            
            return valid, start, goal
            
        def scan_fn(carry, rng):
            i, pos, start_mask, goal_mask = carry
            def _cond_fn(val):                
                return val[0] 
            
            def _body_fn(val):
                valid, rng, pos = val
                
                rng, _rng_pair = jax.random.split(rng)
                valid, start, goal = _sample_pair(_rng_pair, start_mask, goal_mask)
                
                positions = jnp.concatenate([start[None], goal[None]], axis=0)
                pos = pos.at[i].set(positions)
                
                return jnp.bitwise_not(valid), rng, pos
            
            (_, rng, pos) = jax.lax.while_loop(
                _cond_fn,
                _body_fn,
                (True, rng, pos)                
            )
            
            start = pos.at[i, 0].get().astype(jnp.int32)
            goal = pos.at[i, 1].get().astype(jnp.int32)
            start_mask = start_mask.at[start[1], start[0]].set(1)
            goal_mask = goal_mask.at[goal[1], goal[0]].set(1)
            return (i+1, pos, start_mask, goal_mask), None
        
                
        fill_max = jnp.max(jnp.array(self.map_size)) + self.rad*2
        pos = jnp.full((self.num_agents, 2, 2), fill_max)  # [num_agents, [start_pose, goal_pose]]
        
        key, key_scan = jax.random.split(key)
        key_scan = jax.random.split(key_scan, self.num_agents)
        (_, pos, _, _), _ = jax.lax.scan(scan_fn, (0, pos, inside_grid, inside_grid), key_scan)
        theta = jax.random.uniform(key, (self.num_agents, 2, 1), minval=-jnp.pi, maxval=jnp.pi)
        cases = jnp.concatenate([pos + 1.5, theta], axis=2)
        
        return map_data, cases
        
    @partial(jax.jit, static_argnums=[0])
    def sample_map(self, key):
        """ Sample map grid from uniform distribution """

        key_fill, key_shuff = jax.random.split(key)

        base_map = self._gen_base_grid()

        free_idx = jnp.arange(0, self.free_grids)
        num_fill = jax.random.randint(key_fill, (1,), 0, self.n_clutter)[0]
        
        map_within = jnp.where(free_idx<num_fill, 1, 0)
        #map_fill = jax.random.shuffle(key_shuff, map_within)
        map_fill = jax.random.permutation(key_shuff, map_within, independent=True)
        map_fill = jnp.reshape(map_fill, (self.height-2, self.width-2))
        return base_map.at[1:-1, 1:-1].set(map_fill)

    
    ### === Collision checking === ###
    @partial(jax.jit, static_argnums=[0])
    def check_circle_map_collision(self, pos, map_grid, rad=None):
        """ For a circle agent"""
        def _variable_grid_size_check(pos, rad, map_grid):
            
            # check inside
            @partial(jax.vmap, in_axes=(0, 0))
            def __check_inside(x_idx, y_idx):
                rc = jnp.array([x_idx, y_idx]) + 0.5
                dist_l2 = jnp.linalg.norm(scaled_pos - rc)
                c = jnp.minimum(dist_l2 - self.scaled_rad, 0.0)
                return c <= 0
            scaled_pos = self.scale_coords(pos)
            
            grid_pos = jnp.floor(scaled_pos).astype(jnp.int32)
            x_idx = self.cc_x_idx + grid_pos[0]
            y_idx = self.cc_y_idx + grid_pos[1]
            
            c_inside = __check_inside(x_idx.flatten(), y_idx.flatten())
            col_check = jnp.any(map_grid.at[y_idx.flatten(), x_idx.flatten()].get() & c_inside)
            
            # check perimeter
            theta = jnp.linspace(0, 2*jnp.pi, 100)  # NOTE hardcoded
            x_cir = rad * jnp.cos(theta) + pos[0] 
            y_cir = rad * jnp.sin(theta) + pos[1] 
            
            x_cir_idx = jnp.floor(self.scale_coords(x_cir)).astype(jnp.int32) 
            y_cir_idx = jnp.floor(self.scale_coords(y_cir)).astype(jnp.int32) 
            col_check = jnp.any(map_grid.at[x_cir_idx, y_cir_idx].get()) | col_check
            
            return col_check
        
        def _grid_size_of_1_check(pos, rad, map_grid):
            h, w = map_grid.shape 
            min_x, min_y = jnp.floor(jnp.maximum(jnp.zeros(2), pos-rad)).astype(int)
            max_x, max_y = jnp.floor(jnp.minimum(jnp.array([w, h]), pos+rad)).astype(int)
                    
            map_c_list = jnp.array([
                [min_x, min_y],
                [max_x, min_y],
                [min_x, max_y],
                [max_x, max_y],
            ]) + self.cell_half_height
                
            grid_check = self._check_grid(map_c_list, pos, rad)
            
            map_occ = jnp.array([
                map_grid[min_y, min_x],
                map_grid[min_y, max_x],
                map_grid[max_y, min_x],
                map_grid[max_y, max_x],
            ]).astype(int)
            return jnp.any((map_occ+grid_check)>1)
        if rad is None: rad = self.rad
        
        return jax.lax.switch(
            int(self.cell_size == 1.0),
            [
                _variable_grid_size_check,
                _grid_size_of_1_check,
            ],
            *(pos, rad, map_grid)
        )    
    
    
    def get_circle_map_occupancy_mask(self, pos, map_grid, rad=None):
        if rad is None: rad=self.rad
        
        wall_map = jnp.zeros(map_grid.shape, dtype=jnp.int32)
        
        theta = jnp.linspace(0, 2*jnp.pi, 100)
        pos_to_check = jnp.array([jnp.cos(theta), jnp.sin(theta)]).T * rad + pos
        print('pos to check shape', pos_to_check.shape)
        idxs = jnp.floor(pos_to_check).astype(int)
        wall_map = wall_map.at[idxs[:, 1], idxs[:, 0]].set(1)
        
        x_mesh, y_mesh = jnp.meshgrid(jnp.arange(0, self.width), jnp.arange(0, self.height))
        mesh = jnp.dstack([x_mesh, y_mesh]).reshape((-1,2))
        cc = jnp.linalg.norm(mesh - pos, axis=1) < rad
        inside_mask = cc.reshape((self.height, self.width)).astype(int)
        print('inside mask', inside_mask)
        return wall_map | inside_mask
        
    
    def check_agent_beam_intersect(self, beam, pos, theta, range_resolution, rad=None):
        """ Check for intersection between a lidar beam and an agent. """
        if rad is None: rad = self.rad
        d = beam[-1] - beam[0]
        f = beam[0] - pos
        
        a = jnp.dot(d, d)
        b = 2*jnp.dot(f, d)
        c = jnp.dot(f, f) - self.rad**2
        
        descrim = b**2 - 4*a*c
        
        t1 = (-b - jnp.sqrt(descrim))/(2*a)
        # t2 = (-b + jnp.sqrt(descrim))/(2*a)
        
        miss = (descrim < 0) | (t1 < 0) | (t1 > 1) #| (host_idx==other_idx)  # | (t2 < 0) | (t2 > 1)
        
        intersect = beam[0] + t1*d
        idx = jnp.floor(jnp.linalg.norm(intersect - beam[0])/range_resolution).astype(int)
        return jax.lax.select(miss, -1, idx)
    
    @partial(jax.jit, static_argnums=[0])
    def check_point_map_collision(self, pos, map_grid):
        """ For a point """
        pos = jnp.floor(self.scale_coords(pos)).astype(int)
        return map_grid.at[pos[1], pos[0]].get() == 1
    
    def check_all_agent_agent_collisions(self, agent_positions: chex.Array, agent_theta: chex.Array) -> chex.Array:
        
        @partial(jax.vmap, in_axes=(0, None))
        def _check_agent_collisions(agent_idx: int, agent_positions: chex.Array) -> bool:
            # TODO this function is a little clunky FIX 
            z = jnp.zeros(agent_positions.shape)
            z = z.at[agent_idx,:].set(jnp.ones(2)*self.rad*2.1)  
            x = agent_positions + z
            return jnp.any(jnp.sqrt(jnp.sum((x - agent_positions[agent_idx,:])**2, axis=1)) <= self.rad*2) 
        
        return _check_agent_collisions(jnp.arange(agent_positions.shape[0]), agent_positions)
    
    @partial(jax.jit, static_argnums=[0])
    def _gen_base_grid(self):
        """ Generate base grid map with walls around border """
        
        map = jnp.zeros((self.height, self.width), dtype=int)
        map = map.at[0,:].set(1)
        map = map.at[-1,:].set(1)
        map = map.at[:, 0].set(1)
        map = map.at[:, -1].set(1)
        
        return map
    
    @partial(jax.vmap, in_axes=[None, 0, None, None])
    def _check_grid(self, c, pos, radius):
        p = jnp.clip(pos - c, -self.cell_half_height, self.cell_half_height) 
        p = p + c 
        return jnp.linalg.norm(p - pos) <= radius
    
    @partial(jax.jit, static_argnums=[0, 4])
    def check_line_collision(self, pos1, pos2, map_data, max_length=5.0):
        """ uses same method as lidar (ray tracing) """
        resolution = 0.05
        line_length = jnp.linalg.norm(pos2-pos1)
        angle = jnp.arctan2(pos2[1]-pos1[1], pos2[0]-pos1[0])

        points = jnp.arange(0, max_length, resolution)
        points = points[:, None] * jnp.array([jnp.cos(angle), jnp.sin(angle)]) + pos1
        coords = jnp.floor(points).astype(int)
        lidar_hits = map_data[coords[:, 1], coords[:, 0]]

        num_points = jnp.floor(line_length/resolution).astype(int)
        idx_range = jnp.arange(points.shape[0])
        lidar_mask = jnp.where(idx_range<num_points, 1, 0)
        lidar_hits = lidar_hits * lidar_mask

        return jnp.any(lidar_hits)
    
    @partial(jax.jit, static_argnums=[0])
    def passable_check(self, pos1, pos2, map_data):
        """ Check if a path exists between pos1 and pos2"""
        
        def _passable(grid, posa, posb):
            grid = _graph_utils.component_mask_with_pos(grid, posa)
            return grid.at[posb[1], posb[0]].get()
        
        grid = map_data.astype(jnp.bool_)
        pos1 = jnp.floor(pos1).astype(jnp.int32)
        pos2 = jnp.floor(pos2).astype(jnp.int32)
        
        if len(pos2.shape) == 2: # batch eval 
            return jax.vmap(_passable, in_axes=(None, 0, 0))(
                grid, pos1, pos2
            )
        else:
            return _passable(grid, pos1, pos2)
        
    def scale_coords(self, x):
        return x / self.cell_size
    
    def dikstra_path(self, map_data, pos1, pos2):
        """ Computes shorted path (if possible) between `pos1` and `pos2` on the grid specified by `map_data`.
        
        Method TL;DR: Dijkstra's algorithm. JIT'd while loop through the open set of nodes, updating the distance
                      to go for each neighbour in `_body`. Terminates when the open set is empty.
        
        TODO: also terminate if the goal is reached. Add unit tests.
        
        Output:
        - `valid` (bool): True if a path exists
        - `d` (float): distance of shortest path, INF if no path exists
        """
        
        h, w = map_data.shape
        NEIGHBOURS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # left, right, up, down
        INF = 1e8  # large number to represent infinity (jnp.inf produced unwanted nan's)
        
        def _flatten_idx(i, j):
            return i * w + j
        
        def _unflatten_idx(idx):
            return jnp.array([idx // w, idx % w])
        
        start = jnp.floor(pos1).astype(jnp.int32)
        end = jnp.floor(pos2).astype(jnp.int32)
        
        h, w = map_data.shape
        start_flat_idx = start[0] * w + start[1]
        set_distances_to_go = jnp.ones((h*w,)) * INF
        
        set_distances_to_go = set_distances_to_go.at[start_flat_idx].set(0)
        set_visited = jnp.zeros((h*w,)) 
        flat_node_idx = jnp.argmin(set_distances_to_go + INF * set_visited)
        
        carry = (
            flat_node_idx,
            set_distances_to_go,
            set_visited
        )
        
        def _cond(carry):
            return carry[1][carry[0]] < INF 
        
        def _body(carry):
            flat_node_idx, set_distances_to_go, set_visited = carry
            unflat_node_idx = _unflatten_idx(flat_node_idx)
            n_idx = unflat_node_idx + NEIGHBOURS
            flat_n_idx = jax.vmap(_flatten_idx)(n_idx[:, 0], n_idx[:, 1])
            
            n_visited = set_visited.at[flat_n_idx].get()
            n_distances_to_go = set_distances_to_go.at[flat_n_idx].get()
            new_distance = set_distances_to_go[flat_node_idx] + 1
            n_valid_node = map_data[n_idx[:, 1], n_idx[:, 0]] == 0
            updated_neighbour_distances = jnp.where(
                (n_distances_to_go > new_distance) &
                (n_visited == 0) & 
                (n_valid_node), 
                new_distance, 
                n_distances_to_go)
            
            set_distances_to_go = set_distances_to_go.at[flat_n_idx].set(updated_neighbour_distances)
            set_visited = set_visited.at[flat_node_idx].set(1)
            flat_node_idx = jnp.argmin(set_distances_to_go + INF * set_visited)
            return (flat_node_idx, set_distances_to_go, set_visited)
            
        _, set_distances_to_go, _ = jax.lax.while_loop(
            _cond,
            _body,
            carry
        )
        d = set_distances_to_go[_flatten_idx(end[0], end[1])]
        valid = d < INF
        return valid, jax.lax.select(valid, d, 0.0)
        
    ### === VISUALIZATION === ###
    def plot_map(self, ax: axes.Axes, map_grid: jnp.ndarray):
        """ Plot map grid and scale xticks """
        # print('plotting map of size ', map_grid.shape, 'with cell size', self.cell_size)
        ax.imshow(map_grid, extent=(0, map_grid.shape[1], 0, map_grid.shape[0]), origin="lower", cmap='binary', zorder=0)
        ax.set_xlim(0.0, self.map_size[0])
        ax.set_ylim(0.0, self.map_size[1])
        
        # set ticks to be scaled for grid size
        step_size = 1 / self.cell_size
        x_range = jnp.arange(0, map_grid.shape[1]+1, step_size)
        y_range = jnp.arange(0, map_grid.shape[0]+1, step_size)
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_range*self.cell_size)
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range*self.cell_size)
        
    # def plot_pose(self, ax: axes.Axes, pose):
    #     ax.scatter(pose[:, 0, 0], pose[:, 0, 1], c='b', marker='+')
    #     ax.scatter(pose[:, 1, 0], pose[:, 1, 1], c='r', marker='+')
        
    def plot_agents(self,
                    ax: axes.Axes,
                    pos: jnp.ndarray,
                    theta: jnp.ndarray,
                    goal: jnp.ndarray,
                    done: jnp.ndarray,
                    plot_line_to_goal=True,
                    colour_agents_by_idx=False,
                    rad=None):
        """ Plot agents """
        from matplotlib.patches import Circle
        if rad is None: rad = self.rad
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow', 'brown', 'grey', 'cyan']
        if (done.shape[0] > len(colors)) and colour_agents_by_idx:
            print('Too many agents to colour by index')
            colour_agents_by_idx = False
            
        colours = ['black' if done else 'red' for done in done]
        if colour_agents_by_idx:
            colours = ['black' if done else colors[i] for i, done in enumerate(done)]

        for i in range(done.shape[0]):
            circle = Circle(pos[i], rad, facecolor=colours[i])
            ax.add_patch(circle)

            x = pos[i][0] + rad * np.cos(theta[i])
            y = pos[i][1] + rad * np.sin(theta[i])
            ax.plot([pos[i][0], x], [pos[i][1], y], color='black')
            
            if plot_line_to_goal:
                ax.plot([pos[i][0], goal[i][0]], [pos[i][1], goal[i][1]], color='black', alpha=0.5)
  
    def plot_agent_path(self,
                        ax: axes.Axes,
                        x_seq: jnp.ndarray,
                        y_seq: jnp.ndarray,):
        """ Plot agent path """
        x = self.scale_coords(x_seq)
        y = self.scale_coords(y_seq)
        ax.plot(x, y, c='b', linewidth=2.0, zorder=1)
  
default_coords = jnp.array([
    [-0.25, -0.25],
    [-0.25, 0.25],
    [0.25, 0.25],
    [0.25, -0.25],
])
jackal_coords = jnp.array([
    [-0.254, -0.215],
    [-0.254, 0.215],
    [0.254, 0.215],
    [0.254, -0.215],
])
middle_line = jnp.array([
    [0.0, 0.0],
    [0.254, 0.0],
])

class GridMapPolygonAgents(GridMapCircleAgents):
    """ Map for homogenous, convex polygon agents. 
    """
    
    def __init__(self,
                 num_agents: int,
                 rad,
                 map_size,
                 agent_coords=default_coords,
                 middle_line=middle_line,
                 **map_kwargs):
        super().__init__(num_agents, rad, map_size, **map_kwargs)
        
        self.agent_coords=agent_coords
        self.middle_line=middle_line
        
        # define window around agent to check for map collisions
        min_x = jnp.min(agent_coords[:, 0])
        max_x = jnp.max(agent_coords[:, 0])
        min_y = jnp.min(agent_coords[:, 1])
        max_y = jnp.max(agent_coords[:, 1])
        side_length = jnp.maximum(max_x - min_x, max_y - min_y)
                
        cell_ratio = side_length / self.cell_size
        
        self.agent_window = jnp.ceil(cell_ratio*2).astype(int)  # NOTE times 2 should be enough
        self.idx_offsets = jnp.arange(-self.agent_window, self.agent_window+1)        
        
        #  2D with one set of coords for all agents 
        assert (len(self.agent_coords.shape) == 2)
        # or \
            # ((self.agent_coords.shape[0] == self.num_agents) and \
                # (len(self.agent_coords.shape) == 3))

    @partial(jax.jit, static_argnums=[0])
    def transform_coords(self, pos, theta, coords):
        r = rotation_matrix(theta)
        return jnp.matmul(r, coords.T).T + pos

    @partial(jax.jit, static_argnums=[0])
    def check_agent_map_collision(self, pos, theta, map_grid, agent_coords=None):
        """ Check for collision between agent and map. For polygon agents. 
        For now assuming all agents have the same shape and that side lengths 
        are less than the grid size. """   
                
        if agent_coords is None: agent_coords = self.agent_coords
                
        idx_to_check = jnp.floor(pos / self.cell_size).squeeze()  # [x, y]
        idx_0 = (idx_to_check[0] + self.idx_offsets).astype(int)
        idx_1 = (idx_to_check[1] + self.idx_offsets).astype(int)

        idx_pairs = jax.vmap(
            lambda x, y: jax.vmap(lambda a, b: jnp.array([a, b]), in_axes=(None, 0))(x, y),
            in_axes=(0, None)
        )(idx_1, idx_0).reshape((-1, 2))
        
        # need to scale to take account of grid size
        transformed_coords = self.transform_coords(pos, theta.squeeze(), agent_coords)
        scaled_coords = transformed_coords / self.cell_size
        scaled_coords_appended = jnp.concatenate([scaled_coords, scaled_coords[0, :][None]], axis=0)

        cut = jnp.any(
            jax.vmap(self._checkGrid,
                     in_axes=(None, None, 0, None))(scaled_coords_appended[:-1], scaled_coords_appended[1:], idx_pairs, map_grid)
        )
        
        inside = jnp.any(
            jax.vmap(self._checkInsideGrid,
                     in_axes=(None, 0, None))(scaled_coords, idx_pairs, map_grid)
        )
        return cut | inside
        
    @partial(jax.jit, static_argnums=[0])
    def get_agent_map_occupancy_mask(self, pos, theta, map_grid, agent_coords=None):
        
        if agent_coords is None: agent_coords = self.agent_coords
        
        map_mask = jnp.ones(map_grid.shape, dtype=jnp.int32)
        
        idx_to_check = jnp.floor(pos / self.cell_size).squeeze()
        idx_0 = (idx_to_check[0] + self.idx_offsets).astype(int)
        idx_1 = (idx_to_check[1] + self.idx_offsets).astype(int)

        idx_pairs = jax.vmap(
            lambda x, y: jax.vmap(lambda a, b: jnp.array([a, b]), in_axes=(None, 0))(x, y),
            in_axes=(0, None)
        )(idx_1, idx_0).reshape((-1, 2))
        
        # need to scale to take account of grid size
        transformed_coords = self.transform_coords(pos, theta.squeeze(), agent_coords)
        scaled_coords = transformed_coords / self.cell_size
        scaled_coords_appended = jnp.concatenate([scaled_coords, scaled_coords[0, :][None]], axis=0)

        cut = jax.vmap(self._checkGrid,
                       in_axes=(None, None, 0, None))(scaled_coords_appended[:-1], scaled_coords_appended[1:], idx_pairs, map_mask)
        inside = jax.vmap(self._checkInsideGrid,
                          in_axes=(None, 0, None))(scaled_coords, idx_pairs, map_mask)
        
        collisions = cut | inside
        valid_idx = (idx_pairs[:, 0] >= 0) & (idx_pairs[:, 0] < self.height) & (idx_pairs[:, 1] >= 0) & (idx_pairs[:, 1] < self.width) & collisions
        idx_pairs = jnp.where(jnp.repeat(valid_idx[:, None], 1, 1), idx_pairs, jnp.zeros(idx_pairs.shape)).astype(int)
        map_mask = jnp.zeros(map_grid.shape, dtype=jnp.int32)
        
        return map_mask.at[idx_pairs[:, 0], idx_pairs[:, 1]].set(1)

    
    def _checkGrid(self, x1y1, x2y2, grid_idx, map_grid):
        
        def _checkLineLine(x1, y1, x2, y2, x3, y3, x4, y4):
            """ Check collision between line (x1, y1) -- (x2, y2) and line (x3, y3) -- (x4, y4) """
            uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            c = (uA >= 0) & (uA <= 1) & (uB >= 0) & (uB <= 1)
            return c.astype(jnp.bool_)      
        
        def _checkLineRect(x1, y1, x2, y2, rx, ry):
            """ Check collision between line (x1, y1) -- (x2, y2) and rectangle with bottom left corner at (rx, ry) 
            and width and height of 1."""
            vmap_checkLineLine = jax.vmap(_checkLineLine, in_axes=(None, None, None, None, 0, 0, 0, 0))
            x3 = jnp.array([0, 1, 0, 0]) + rx
            y3 = jnp.array([0, 0, 0, 1]) + ry
            x4 = jnp.array([0, 1, 1, 1]) + rx
            y4 = jnp.array([1, 1, 0, 1]) + ry
            checks = vmap_checkLineLine(x1, y1, x2, y2, x3, y3, x4, y4)
            return jnp.any(checks)
        
        @partial(jax.vmap, in_axes=(0, 0, None))
        def _checkSide(x1y1, x2y2, grid_idx):
            x1, y1 = x1y1
            x2, y2 = x2y2
            ry, rx = grid_idx[0], grid_idx[1]
            filled = map_grid[ry, rx]
            c = _checkLineRect(x1, y1, x2, y2, rx, ry)
            return c & filled
        
        valid_idx = (grid_idx[0] >= 0) & (grid_idx[0] < self.height) & (grid_idx[1] >= 0) & (grid_idx[1] < self.width)
        return jnp.any(_checkSide(x1y1, x2y2, grid_idx)) & valid_idx
    
    def _checkInsideGrid(self, sides, grid_idx, map_grid):
        """ Check if polygon is inside grid cell, NOTE assumes grid cell is of size 1x1 """
        
        def _checkPolyWithinRect(sides, rx, ry):
            """ Check if polygon is within rectangle with bottom left corner at (rx, ry) and width and height of 1."""
            
            def _checkPointRect(x, y, rx, ry):
                """ Check if point (x, y) is within rectangle with bottom left corner at (rx, ry) and width and height of 1."""
                return (x >= rx) & (x <= rx+1) & (y >= ry) & (y <= ry+1)
            
            vmap_checkPointRect = jax.vmap(_checkPointRect, in_axes=(0, 0, None, None))
            checks = vmap_checkPointRect(sides[:, 0], sides[:, 1], rx, ry)
            return jnp.all(checks)
        
        valid_idx = (grid_idx[0] >= 0) & (grid_idx[0] < self.height) & (grid_idx[1] >= 0) & (grid_idx[1] < self.width)
        inside = _checkPolyWithinRect(sides, grid_idx[1], grid_idx[0])
        return inside & map_grid[grid_idx[0], grid_idx[1]] & valid_idx
    
    @partial(jax.jit, static_argnums=[0])
    def check_all_agent_agent_collisions(self, agent_positions: chex.Array, agent_theta: chex.Array, agent_coords=None) -> chex.Array:
        """ Use Separating Axis Theorem (SAT) to check for collisions between convex polygon agents. 
        
        Separating Axis Theorem TL;DR: Searches for a line that separates two convex polygons. If no line is found, the polygons are colliding.
                                       To search for a separating line, we look at the normals of the edges of the polygons and project the vertices
                                       of all polygons onto these normals. If the projections of the vertices of one polygon do not overlap with the
                                       projections of the vertices of the other polygon, then the polygons are not colliding.
        """    
        
        def _orthogonal(v):
            return jnp.array([v[1], -v[0]])
        
        if agent_coords is None: agent_coords = self.agent_coords
        
        transformed_coords = jax.vmap(self.transform_coords, in_axes=(0, 0, None))(agent_positions, agent_theta.squeeze(), agent_coords)
        all_coords = transformed_coords.reshape((-1, 2))  # [num_agents*4, 2]
        trans_rolled = jnp.roll(transformed_coords, 1, axis=1)
        
        edges = transformed_coords - trans_rolled  # [num_agents, 4, 2]
        orthog_edges = jax.vmap(_orthogonal, in_axes=(0))(edges.reshape((-1, 2))).reshape((-1, 4, 2))  # NOTE assuming 4 sides

        all_axis = orthog_edges / jnp.linalg.norm(orthog_edges, axis=2)[:,:,None]  # [num_agents, 4, 2]
            
        def _project_axis(axis):
            return jax.vmap(jnp.dot, in_axes=(None, 0))(axis, all_coords)
            
        axis_projections = jax.vmap(_project_axis)(all_axis.reshape((-1, 2)))
        axis_projections_by_agent = axis_projections.reshape((-1, self.num_agents, 4)) 
        axis_ranges_by_agent = jnp.stack([jnp.min(axis_projections_by_agent, axis=2), jnp.max(axis_projections_by_agent, axis=2)], axis=-1)
        axis_by_agent_range_by_agent = axis_ranges_by_agent.reshape((self.num_agents, -1) + axis_ranges_by_agent.shape[1:])
        def _calc_overlaps(agent_idx, agent_axis_ranges):
            overlaps = (agent_axis_ranges[:, agent_idx, 0][:, None] <= agent_axis_ranges[:, :, 1]) & (agent_axis_ranges[:, :, 0] <= agent_axis_ranges[:, agent_idx, 1][:, None])            
            overlaps = overlaps.at[:, agent_idx].set(False)
            return jnp.all(overlaps, axis=0)
        
        overlaps_matrix = jax.vmap(_calc_overlaps)(jnp.arange(self.num_agents), axis_by_agent_range_by_agent) # all overlaps for all vertex is a collision
        # need to match matrix triangles
        def _join_matrix(rows, cols):
            return jnp.any(jnp.bitwise_and(rows, cols))
            
        c_free = jax.vmap(_join_matrix)(overlaps_matrix, overlaps_matrix.T)
        return c_free
                
    def check_agent_beam_intersect(self, beam, pos, theta, range_resolution, agent_coords=None):
        """ Check for intersection between a lidar beam and an agent. """
        if agent_coords is None: agent_coords = self.agent_coords
        
        @partial(jax.vmap, in_axes=(None, None, 0, 0))
        def _checkSide(beam_start, beam_end, side_start, side_end):
            """ Check collision between line (x1, y1) -- (x2, y2) and line (x3, y3) -- (x4, y4) """
            x1, y1 = beam_start
            x2, y2 = beam_end
            x3, y3 = side_start
            x4, y4 = side_end
            
            uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            c = (uA >= 0) & (uA <= 1) & (uB >= 0) & (uB <= 1)
            
            ix = x1 + (uA * (x2-x1))
            iy = y1 + (uA * (y2-y1))
            intersect = jnp.array([ix, iy])
            idx = jnp.floor(jnp.linalg.norm(intersect - beam[0])/range_resolution)
            
            return jax.lax.select(c, idx, jnp.inf)
        
        tc = self.transform_coords(pos, theta, agent_coords)
        tc = jnp.concatenate([tc, tc[0, :][None]], axis=0)
        
        idxs = _checkSide(beam[0], beam[-1], tc[:-1], tc[1:])
        idx = jnp.min(idxs)
        return jax.lax.select(idx==jnp.inf, -1.0, idx).astype(int)

    def plot_agents(
                self,
                ax: axes.Axes,
                pos: jnp.ndarray,
                theta: jnp.ndarray,
                goal: jnp.ndarray,
                done: jnp.ndarray,
                plot_line_to_goal=True,
                agent_coords=None,
                middle_line=None,
                colour_agents_by_idx=False,
        ):
        """ Plot agents """
        from matplotlib.patches import Polygon
        if agent_coords is None: agent_coords = self.agent_coords
        if middle_line is None: middle_line = self.middle_line
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow', 'brown', 'grey', 'cyan']
        if (done.shape[0] > len(colors)) and colour_agents_by_idx:
            print('Too many agents to colour by index')
            colour_agents_by_idx = False
            
        colours = ['black' if done else 'red' for done in done]
        if colour_agents_by_idx:
            colours = ['black' if done else colors[i] for i, done in enumerate(done)]

        for i in range(done.shape[0]):
            transformed_coords = self.transform_coords(pos[i], theta[i], agent_coords) / self.cell_size
            
            poly = Polygon(transformed_coords, facecolor=colours[i])
            ax.add_patch(poly)

            # middle line
            transformed_middle_line = self.transform_coords(pos[i], theta[i], self.middle_line) / self.cell_size
            ax.plot(transformed_middle_line[:, 0], transformed_middle_line[:, 1], color='black')
                        
            if plot_line_to_goal:
                pos_scaled = self.scale_coords(pos[i])
                goal_scaled = self.scale_coords(goal[i])
                ax.scatter(goal_scaled[0], goal_scaled[1], marker='+', c='g')
                ax.plot([pos_scaled[0], goal_scaled[0]], [pos_scaled[1], goal_scaled[1]], color='black', alpha=0.5) 

SMOOTHING_IDX_OFFSETS = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
INFLATE_IDX_OFFSETS_3 = SMOOTHING_IDX_OFFSETS
INFLATE_IDX_OFFSETS_5 = jnp.concatenate([INFLATE_IDX_OFFSETS_3, jnp.array([[2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [-1, 2], [-2, 2], [-2, 1], [-2, 0], [-2, -1], [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2], [2, -1],])], axis=0)


class GridMapBarn(GridMapPolygonAgents):
    
    def __init__(self,
                 num_agents, 
                 rad, 
                 map_size,
                 smoothing_iters=4,
                 smoothing_upper_count=3,
                 smoothing_lower_count=1,
                 agent_coords=jackal_coords,
                 cell_size=0.15,
                 **map_kwargs):
        
        super().__init__(num_agents, rad, map_size, agent_coords=agent_coords, cell_size=cell_size, **map_kwargs)

        self.smoothing_iters = smoothing_iters
        self.smoothing_upper_count = smoothing_upper_count
        self.smoothing_lower_count = smoothing_lower_count
        assert self.smoothing_upper_count > self.smoothing_lower_count, 'smoothing upper count must be greater than lower count'
        
        self.inner_idx = jnp.array([[i, j] for i in range(1, self.height-1) for j in range(1, self.width-1)])
        self.outer_idx = jnp.array([[i, j] for i in range(self.height) for j in range(self.width) if (i==0) or (i==self.height-1) or (j==0) or (j==self.width-1)])
    
    @partial(jax.jit, static_argnums=[0])
    def sample_test_case(self, rng: chex.PRNGKey):
        
        return self.sample_barn_test_case(rng)
    
    @partial(jax.jit, static_argnums=[0])
    def sample_barn_test_case(self, rng):
        
        def _cond_fun(val):
            valid, test_case, rng = val
            # jax.debug.print('valid {v}', v=valid)
            return ~valid
        
        def _body_fun(val):
            valid, test_case, rng = val
            rng, _rng = jax.random.split(rng)
            valid, test_case = self.barn_test_case(_rng)
            return (valid, test_case, rng)
        
        init_test_case = (jnp.zeros((2,)), jnp.zeros((2,)), jnp.zeros((self.height, self.width), dtype=jnp.int32))
        
        test_case = jax.lax.while_loop(
            _cond_fun, _body_fun, (False, init_test_case, rng)
        )
        valid, test_case, rng = test_case
        
        start, goal, smoothed_map = test_case
        
        theta = (jnp.pi/2) * jax.random.choice(rng, jnp.arange(4), (2,))
        test_case = jnp.vstack([start, goal])
        test_case = jnp.concatenate([test_case, theta[:,None]], axis=1)
        
        return smoothed_map, test_case[None]
        
    def barn_test_case(self, rng):
        # from matplotlib import pyplot as plt
        rng, _rng = jax.random.split(rng)
        base_map = self.sample_map(_rng)
        
        def _smooth_fn(map_data, unused):
            inner_map = jax.vmap(self._smooth, in_axes=(0, None))(self.inner_idx, map_data)
            inner_map = inner_map.reshape((self.height-2, self.width-2))
            map_data = map_data.at[1:-1, 1:-1].set(inner_map)
            return map_data, None
                
        smoothed_map, _ = jax.lax.scan(
            _smooth_fn, base_map, None, length=self.smoothing_iters
        )        
        
        def _inflate_obs(idx, map):
            value = map.at[idx[0], idx[1]].get()
            around = map.at[INFLATE_IDX_OFFSETS_5[:,0]+idx[0], INFLATE_IDX_OFFSETS_5[:,1]+idx[1]].set(value)
            return around
        
        inner_inflated_map = jax.vmap(_inflate_obs, in_axes=(0, None))(self.inner_idx, smoothed_map).any(axis=0)
        outer_inflated_map = jax.vmap(_inflate_obs, in_axes=(0, None))(self.outer_idx, smoothed_map).any(axis=0)
        inflated_map = inner_inflated_map | outer_inflated_map
        # print('inflated map\n', inflated_map)
        # print('valid cells', (1-inflated_map).sum())
        
        rng, _rng = jax.random.split(rng)
        start = jax.random.choice(rng, jnp.arange(self.height*self.width), p=~inflated_map.flatten())
        start = jnp.array([start % self.width, start // self.width])  # [x, y]
        # print('start:', start)
        # fig, ax = plt.subplots()
        # inflated_to_plot = inflated_map * 0.1 + smoothed_map * 0.9
        # ax.imshow(inflated_to_plot, cmap='binary')
        # plt.savefig('barn-inflated-map.png')
        
        
        # print('start_idx:', start)
        # with jax.disable_jit(False):
        connected_region = _graph_utils.component_mask_with_pos(inflated_map, start)


        min_dist = 19 
        # empty = jnp.zeros((self.height, self.width), dtype=jnp.int32)
        x_lim = jnp.clip(jnp.array([start[0]-min_dist, start[0]+min_dist+1]), 0, self.width)
        y_lim = jnp.clip(jnp.array([start[1]-min_dist, start[1]+min_dist+1]), 0, self.height)
        print('x_lim:', x_lim, 'y_lim:', y_lim)

        too_close_mask = jnp.ones((self.height, self.width))
        xrange = jnp.arange(self.width)
        too_close_mask_x = jnp.where((xrange >= x_lim[0]) & (xrange < x_lim[1]), 1, 0)
        yrange = jnp.arange(self.width)
        too_close_mask_y = jnp.where((yrange >= y_lim[0]) & (yrange < y_lim[1]), 1, 0)
        
        too_close_mask = jnp.meshgrid(too_close_mask_x, too_close_mask_y)
        valid_mask = 1-jnp.dstack(too_close_mask).all(axis=-1)
        # print('too_close_mask:', valid_mask, valid_mask.shape)
        
        masked_connected_region = connected_region * valid_mask
        
        # fig, ax = plt.subplots()
        # ax.imshow(masked_connected_region, cmap='binary')
        # plt.savefig('barn-masked-map.png')
        
        goal = jax.random.choice(rng, jnp.arange(self.height*self.width), p=masked_connected_region.flatten())
        goal = jnp.array([goal % self.width, goal // self.width])  # [x, y]
        
        
        # fig, ax = plt.subplots()
        # ax.imshow(inflated_to_plot, cmap='binary')
        # ax.plot([start[0], goal[0]], [start[1], goal[1]], c='blue', linestyle='--')
        # ax.scatter(start[0], start[1], c='red', marker='x')
        # # ax.scatter(goal % cols, goal // cols, c='green', marker='x')
        # ax.scatter(goal[0], goal[1], c='green', marker='x')
        # plt.savefig('barn-final-map.png')
        
        valid = ((1-inflated_map).sum() > 0) & (masked_connected_region.sum() > 0)
        # print('test case', (start, goal))
        return valid, (start * self.cell_size, goal * self.cell_size, smoothed_map)

        
    def _smooth(self, idx, map_data):
        
        idx_offsets = SMOOTHING_IDX_OFFSETS + idx
        valid = (idx_offsets[:, 0] > 0) & (idx_offsets[:, 0] < self.height-1) & (idx_offsets[:, 1] > 0) & (idx_offsets[:, 1] < self.width-1)
        n_full = map_data.at[idx_offsets[:,0], idx_offsets[:,1]].get() * valid
        n_full = jnp.sum(n_full)
    
        fill = ((n_full>self.smoothing_upper_count) | map_data.at[idx[0], idx[1]].get()) & (n_full>self.smoothing_lower_count)
        return jax.lax.select(fill, 1, 0)
    
class GridMapPolygonAgentsSingleMap(GridMapPolygonAgents):
    
    def __init__(self,
                 num_agents: int,
                 rad,
                 map_data: List,
                 agent_coords=default_coords,
                 middle_line=middle_line,
                 **map_kwargs):
        
        self._map_data = jnp.array(
            [[int(x) for x in row.split()] for row in map_data], 
            dtype=jnp.int32
        )
        height, width = self._map_data.shape
        map_size = (width, height)
        super().__init__(num_agents=num_agents,
                        rad=rad,
                        map_size=map_size,
                        agent_coords=agent_coords,
                        middle_line=middle_line,
                        **map_kwargs)

    @partial(jax.jit, static_argnums=[0])
    def sample_map(self, key):
        return self._map_data

class GridMapFromBuffer(GridMapCircleAgents):
    
    def __init__(self,
                 num_agents,
                 rad,
                 map_size,
                 map_grids=None,
                 dir_path=None,
                 file_prefix="map_buffer_"):
        """ 
        saved map buffers expected in format: (map_data, starts, theta, goals)
        """
        print('** Super old code beware **')
        super().__init__(num_agents, rad, map_size, fill=0.1)
        if map_grids is None and dir_path is None:
            raise ValueError("Must specify either map_grids or dir_path")
        if map_grids is not None and dir_path is not None:
            raise ValueError("Cannot specify both map_grids and dir_path")
        
        if dir_path is not None:
            # list files in dir_path 
            files = [filename for filename in os.listdir(dir_path) if filename.startswith(file_prefix)]
            print('files', files)
            test_cases = (
                jnp.empty((0, self.height, self.width), dtype=jnp.int32), 
                jnp.empty((0, 2), dtype=jnp.float32),
                jnp.empty((0, 1), dtype=jnp.float32),
                jnp.empty((0, 2), dtype=jnp.float32),
            )
            for filename in files:
                # load pkls
                filepath = os.path.join(dir_path, filename)
                with open(filepath, "rb") as f:
                    tc = pickle.load(f)
                    print('tc c', tc)
                    test_cases = jax.tree.map(lambda x, y: jnp.concatenate((x, y), axis=0), test_cases, tc)
            self.test_cases = test_cases
            self.num_test_cases = test_cases[0].shape[0]
            print('test cases', test_cases)
     
        if map_grids is not None:
            raise NotImplementedError("map_grids not implemented yet")
        
    @partial(jax.jit, static_argnums=[0])
    def sample_scenario(self, key):
        print('-- sampling scenarios -- ')
        idx = jax.random.randint(key, (1,), minval=0, maxval=self.num_test_cases)[0]
        tc = jax.tree.map(lambda x: x[idx], self.test_cases)
        print('tc ', tc)
        map_data = tc[0]
        print('map data', map_data.shape)
        
        theta = jnp.array([tc[2], 0])
        print('theta', theta)
        
        case = jnp.array([tc[1], tc[3]])
        print('case', case)
        test_case = jnp.concatenate([case, theta], axis=-1)
        
        return map_data, test_case
    
    '''@partial(jax.jit, static_argnums=[0])
    def sample_map(self, key):
        """ Sample map grid from pre-specified map grids list """
        if self.map_grids.shape[0]>1:
            map_idx = jax.random.randint(key, (1,), minval=0, maxval=len(self.map_grids))[0]
            map_grid = self.map_grids[map_idx]
        else:
            map_grid = self.map_grids[0]
        return map_grid'''
    
def rrt_reward(new_pos, pos, goal):
    goal_reached = jnp.linalg.norm(new_pos - goal) <= 0.3
    #if goal_reached: print('goal reached')
    weight_g = 0.2
    goal_rew = 1
    rga = weight_g * (jnp.linalg.norm(pos - goal) - jnp.linalg.norm(new_pos - goal))
    rg = jnp.where(goal_reached, goal_rew, rga)
    return rg
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    key = jax.random.PRNGKey(3) # 3, 7, 9
    
    file_path = "/home/alex/repos/jax-multirobsim/failure_maps/cosmic-waterfall-17"
    map_gen = GridMapFromBuffer(1, 0.3, (7, 7), dir_path=file_path)
    
    
    s = map_gen.sample_scenario(key)    
    raise
    key, key_rrt = jax.random.split(key)
    
    map_gen = GridMapCircleAgents(1, 0.3, (10, 10), 0.5)
    map_data, case = map_gen.sample_scenario(key)
    
    start = case[:, 0, :2].flatten()
    goal = case[:, 1, :2].flatten()
    print('case', case, 'start', start, 'goal', goal)
    
    '''gr, parent = map_gen.a_star(map_data, start, goal)
    print('parent', parent)
    
    fig, ax = plt.subplots()
    
    ax.imshow(map_data, extent=(0, map_data.shape[1], 0, map_data.shape[0]), origin="lower", cmap='binary', alpha=0.8)
    
    zero_grid = np.zeros((10, 10))
    x, y = jnp.meshgrid(jnp.arange(map_data.shape[0]), jnp.arange(map_data.shape[1]))#.reshape(-1, 2)
    coords = jnp.dstack((y.flatten(), x.flatten())).squeeze()
    for i in range(parent.shape[0]):
        if parent[i] == -1: continue
        node = coords[parent[i]]
        print('node', node)
        zero_grid[node[0], node[1]] = 1
    
    ax.imshow(zero_grid, extent=(0, 10, 0, 10), origin="lower", cmap='binary', alpha=0.2)
    
    map_gen.plot_pose(ax, case)
    plt.show()
    
    raise'''
    #print('map_data', map_data)
    tree, goalr = map_gen.rrt_star(key_rrt, map_data, start, goal)
    print('tree', tree, 'goalr', goalr)
    print('case', case)
    
    fig, ax = plt.subplots()
    
    map_gen.plot(ax, map_data)
    
    #ax.scatter(case[:, 0, 0], case[:, 0, 1], c='r')
    for n in range(tree.shape[0]):
        if tree[n, 0] == 0.0: break
        ax.scatter(tree[n, 0], tree[n, 1], c='gray')
        pi = tree[n, 2]
        if pi == -1: continue
        pi = int(pi)
        ax.plot([tree[n, 0], tree[pi, 0]], [tree[n, 1], tree[pi, 1]], c='gray', marker='+', alpha=0.75)
        
    

    if goalr:
        goal_idx = jnp.argwhere(tree[:,-1]==1)
        print('goal_idx', goal_idx)
        
        for g_idx in goal_idx:
            c_idx = g_idx[0]
            path_length = 0.0
            rew = 0.0
            while c_idx != 0:
                print('cidx', c_idx, 'tree row', tree[c_idx])
                c_pos = tree[c_idx, :2]
                p_idx = int(tree[c_idx, 2])
                p_pos = tree[p_idx, :2]
                path_length += jnp.linalg.norm(c_pos - p_pos)
                rew += rrt_reward(c_pos, p_pos, goal)
                ax.plot([c_pos[0], p_pos[0]], [c_pos[1], p_pos[1]], c='r', alpha=0.25)
                print('p_pos', p_pos, 'c_pos', c_pos, 'rew', rew)
                c_idx = p_idx
            print('path_length', path_length, 'rew', rew)
    
    map_gen.plot_pose(ax, case)
    plt.show()
    '''raise
    p_idx = parent[goal_idx]
    print('p_idx', p_idx)
    while p_idx != -1:
        print('corrds', coords[p_idx])
        ax.plot( [coords[p_idx, 1]+0.5, coords[goal_idx, 1]+0.5], [coords[p_idx, 0]+0.5, coords[goal_idx, 0]+0.5], c='r')
        goal_idx = p_idx
        p_idx = parent[goal_idx]

    
    plt.show()
    
    raise
    pos = jnp.array([1.5, 1.5])
    x = map_gen.check_agent_collision(pos, map_data)
    print(x)
    print(map_data)'''
    
    