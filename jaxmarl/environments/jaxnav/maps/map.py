import jax
from jax import numpy as jnp
import numpy as np
from functools import partial
import matplotlib.axes._axes as axes
import chex 

class Map(object):
    """ Base class for a map """
    
    def __init__(
        self,
        num_agents,
        rad,
        map_size,
        start_pad=1.5,
        valid_path_check=False,
    ):
        assert start_pad>=1.0, 'start_pad must be greater than or equal to 1.0'
        
        self.num_agents = num_agents
        self.rad = rad
        self.map_size = map_size
        self.start_pad=start_pad
        self.valid_path_check = valid_path_check
        
        # Test case sampling TODO fix this to be not hard coded
        self.dist_to_goal = 50 # 5
        self.rrt_samples = 1000
        self.rrt_step_size = 0.25
        self.goal_radius = 0.3
        
    @partial(jax.jit, static_argnums=[0])
    def sample_scenario(self, key):
        """ Sample map grid and agent start/goal positions """
        
        key_map, key_case = jax.random.split(key)
        map_data = self.sample_map(key_map)
        test_case = self.sample_test_case(key_case, map_data)
        return map_data, test_case
    
    @partial(jax.vmap, in_axes=[None, 0])
    def sample_scenarios(self, key):
        return self.sample_scenario(key)
        
    @partial(jax.jit, static_argnums=[0])
    def sample_map(self, key):
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=[0])
    def sample_test_case(self, key):
        """ Sample a test case for a given map """
        key, _key = jax.random.split(key)
        map_data = self.sample_map(_key)
        radii = jnp.array([self.rad*self.start_pad, self.goal_radius])

        def _sample_pair(key: chex.PRNGKey):
            """ Sample a start and goal pose for an agent """
            key_s, key_g, key_t = jax.random.split(key, 3)
            low_lim = 1 + self.rad 
            high_lim = self.map_size[1] - 1 - self.rad
            start = jax.random.uniform(key_s, (1, 2), minval=low_lim, maxval=high_lim)
            g_low_lim = jnp.clip(start - self.dist_to_goal, low_lim, high_lim)
            g_high_lim = jnp.clip(start + self.dist_to_goal, low_lim, high_lim)
            goal = jax.random.uniform(key_g, (1, 2), minval=g_low_lim, maxval=g_high_lim)
            theta = jax.random.uniform(key_t, (2, 1), minval=-jnp.pi, maxval=jnp.pi)
            positions = jnp.concatenate([start, goal], axis=0)
            poses = jnp.concatenate([positions, theta], axis=1)
            return poses
        
        def _agent_collision(pos, test_case, rad): 
            dists = jnp.linalg.norm(test_case-pos, axis=1) <= rad*2
            return jnp.any(dists)
        
        def _cond_idx(val):
            """ true while i is less than the number of agents """
            key, i, case = val
            return i < case.shape[0]
        
        def _body_idx(val, key):
            """ samples a start and goal pair for an agent, taking into account pairs sampled previously """
            i, case = val
            def _cond_pos(val):
                """ Check if the sampled pair is valid. Checks
                1. check start and goal do not collide with the map
                3. check the start is not within a radius with other agents' starts
                4. check the start is not within a radius with other agents' starts 
                5. check if start and goal are too close
                """
                
                key, pair, case = val
                temp_case = case.at[i].set(pair+self.rad*3)  # ensure no conflict 
                
                map_collisions = jnp.any(jax.vmap(self.check_circle_map_collision, in_axes=[0, None, 0])(pair[:, :2], map_data, radii))
                agent_collisions = jnp.any(jax.vmap(_agent_collision, in_axes=[0, 1, None])(pair[:, :2], temp_case[:, :, :2], self.rad*self.start_pad))
                
                too_close = (jnp.linalg.norm(pair[0, :2] - pair[1, :2]) <= 2*self.rad).astype(jnp.bool_)
                
                check = map_collisions | agent_collisions | too_close
                
                if self.valid_path_check:
                    valid_path = self.passable_check(pair[0, :2], pair[1, :2], map_data)  # WARNING can make code too slow
                    check = check | ~valid_path
                
                return check
                                
                # return map_collisions | agent_collisions | too_close | ~valid_path
                
                #print('p map', jnp.any(pmap_collision(pair, map_grid, rad)))
                #jax.debug.print('p {pair} map {p}, s ag {s}, g ag {g}, dist f{d} dist {c}', pair=pair, p=jnp.any(pmap_collision(pair, map_grid, rad)), s=agent_collision(pair[0], temp_case[:, 0, :], rad), g=agent_collision(pair[1], temp_case[:, 1, :], rad), d=jnp.linalg.norm(pair[0] - pair[1]) >= dist_to_goal, c=(jnp.linalg.norm(pair[0] - pair[1]) <= 2*rad).astype(jnp.bool_))
                """ true while pos is not valid """
                # 1. check if start collides with map
                # 2. check if goal collides with map
                # 3. check if start collides with other agents
                # 4. check if goal collides with other agents's goals 
                # 5. check if start and goal are too close
                
                return self.check_agent_map_collision(pair[0], map_data, self.rad*self.start_pad) \
                    | self.check_agent_map_collision(pair[1], map_data) \
                    | _agent_collision(pair[0], temp_case[:, 0, :], self.rad) \
                    | _agent_collision(pair[1], temp_case[:, 1, :], self.rad) \
                    | (jnp.linalg.norm(pair[0] - pair[1]) >= self.dist_to_goal).astype(jnp.bool_) \
                    | (jnp.linalg.norm(pair[0] - pair[1]) <= 2*self.rad).astype(jnp.bool_) #\ | ~(self.rrt(key_rrt, map_data, pair[0], pair[1]))
        
            def _body_pos(val):
                """ Sample a start and goal pair """
                key, pair, case = val
                key, key_point = jax.random.split(key)
                pair = _sample_pair(key_point)
                case = case.at[i].set(pair)
                #jax.debug.print('case {c}', c=case)
                return key, pair, case
        
            key, key_point = jax.random.split(key)
            pair = _sample_pair(key_point)
            case = case.at[i].set(pair)
                        
            key, pair, case = jax.lax.while_loop(
                _cond_pos,
                _body_pos,
                (key, pair, case),
            )
            
            i += 1
            return (i, case), None

        fill_max = jnp.max(jnp.array(self.map_size)) + self.rad*2
        case = jnp.full((self.num_agents, 2, 3), fill_max)  # [num_agents, [start_pose, goal_pose]]

        i = 0
        
        key_scan = jax.random.split(key, self.num_agents)
        (_, case), _ = jax.lax.scan(_body_idx, (i, case), key_scan)
        
        # Add intial orientation
        #theta = jax.random.uniform(key, (self.num_agents, 2, 1), minval=-jnp.pi, maxval=jnp.pi)
        #case = jnp.concatenate([case, theta], axis=-1)        
        return map_data, case
    
    @partial(jax.jit, static_argnums=[0])
    def check_circle_map_collision(self, pos, map_data, rad=None):
        """ Check collision between a circle at position `pos` of radius `rad` and the map.
        If rad is None, use the class rad. """
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=[0])
    def check_agent_map_collision(self, pos, theta, map_data, **agent_kwargs):
        """ Check collision between an agent at position `pos` and the map"""
        # NOTE should we switch these functions to use pose, i.e. [pos_x, pos_y, theta]?
        raise NotImplementedError
    
    def check_all_agent_agent_collisions(self, pos, theta):
        """ Check collision between all agents """
        raise NotImplementedError
        
    def check_agent_beam_intersect(self, beam, pos, theta, range_resolution, **agent_kwargs):
        """ Check for intersection between a lidar beam and an agent. """
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=[0])
    def check_point_map_collision(self, pos, map_data):
        """ Check collision between `pos` and the map"""
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=[0])
    def check_line_collision(self, pos1, pos2, map_data):
        """ Check collision between line (pos1) -- (pos2) the map"""
        raise NotImplementedError
    
    def passable_check(self, pos1, pos2, map_data):
        """ Check whether a path exists between pos1 and pos2.
        Note, this does not return the path, only whether a path exists. """
        raise NotImplementedError
    
    @partial(jax.jit, static_argnums=[0])
    def check_agent_translation(self, start, end, map_data):
        """ True for valid translation, False for invalid translation """
        
        l_slope = (end[1] - start[1]) / (end[0] - start[0])
        perpendicular_slope = -1 / l_slope
        delta_x = self.rad / (1 + perpendicular_slope**2)**0.5
        delta_y = perpendicular_slope * delta_x
        #jax.debug.print('per slope {p}', p=perpendicular_slope)
        # Calculate the coordinates of the two points
        delta = jnp.array([delta_x, delta_y])
        start_lower = start + delta
        start_upper = start - delta    
        end_lower = end + delta
        end_upper = end - delta
    
        lower = self.check_line_collision(start_lower, end_lower, map_data)
        upper = self.check_line_collision(start_upper, end_upper, map_data)
        #jax.debug.print('lower {l}, upper {u}', l=lower, u=upper)
        return ~lower & ~upper
    

    @partial(jax.jit, static_argnums=[0])
    def rrt(self, key, map_data, start, goal):
        """ Run RRT algorithm to find a path between start and goal """

        INF = 10000

        print('key')
        low_lim = 1 + self.rad 
        high_lim = self.map_size[1] - 1 - self.rad
        goal_square = jnp.floor(goal).astype(jnp.int32).squeeze()
        #print('goal square', goal_square)
        
        tree = jnp.empty((self.rrt_samples, 3))  # [sample, [x, y, parent]]
        tree = tree.at[0].set(jnp.append(start, -1))
        rrt_idx = 1
        gr = False
            
        def _cond_fun(val):
            i, gr = val[0], val[1]
            return (i < self.rrt_samples) & ~gr
        
        def _body_fun(val):
            i, gr, rrt_idx, tree, key = val
            key, key_s = jax.random.split(key)
            # Sample position, find closest idx and increment towards sampled pos
            sampled_pos = jax.random.uniform(key_s, (2,), minval=low_lim, maxval=high_lim)
            #closest_idx = jnp.argmin(jnp.linalg.norm(tree[:, :2] - sampled_pos, axis=1))
            distances = jnp.linalg.norm(tree[:, :2] - sampled_pos, axis=1)
            distance_mask = jnp.where(jnp.arange(self.rrt_samples) < rrt_idx, 0, 1) * INF
            closest_idx = jnp.argmin(distances + distance_mask)

            tree_pos = tree[closest_idx, :2]
            step_size = jnp.minimum(self.rrt_step_size, jnp.linalg.norm(sampled_pos - tree_pos))
            
            angle = jnp.arctan2(sampled_pos[1] - tree_pos[1], sampled_pos[0] - tree_pos[0])
            test_pos = tree_pos + jnp.array([jnp.cos(angle), jnp.sin(angle)]) * step_size
            
            # Check free space, line collision
            free_space = ~self.check_agent_map_collision(test_pos, map_data)
            line_collision = self.check_agent_translation(tree_pos, test_pos, map_data)
            valid = free_space & line_collision
            
            #goal_square_reached = jnp.array_equal(jnp.floor(test_pos), goal_square)
            goal_reached = jnp.linalg.norm(test_pos - goal) < self.goal_radius
            new_node = jax.lax.select(valid, jnp.concatenate([test_pos, jnp.array([closest_idx])]), jnp.zeros((3,)))
            tree = tree.at[rrt_idx].set(new_node)
            
            rrt_idx += 1*valid
            gr = gr | (goal_reached & valid)
            return (i+1, gr, rrt_idx, tree, key)
            
        val = jax.lax.while_loop(_cond_fun, _body_fun, (0, gr, rrt_idx, tree, key))
        gr, tree = val[1], val[3]    
        
        return tree, gr
    
    @partial(jax.jit, static_argnums=[0])
    def rrt_star(self, key, map_data, start, goal):
        """ Run RRT* algorithm to find an optimal path between start and goal """

        INF = 10000

        low_lim = 1 + self.rad 
        high_lim = self.map_size[1] - 1 - self.rad
        goal_square = jnp.floor(goal).astype(jnp.int32).squeeze()
        
        check_point_connections = jax.vmap(self.check_agent_translation, in_axes=(None, 0, None))
        
        tree = jnp.empty((self.rrt_samples, 5))  # [sample, [x, y, parent, cost, goal]]
        #print('start',start)
        tree = tree.at[0].set(jnp.append(start, jnp.array([-1, 0.0, 0.0])))
        rrt_idx = 1
        goal_reached = False
        goal_idx = jnp.full((30,), -1, dtype=jnp.int32)
            
        rrt_star_neighbours = 20
            
        def _cond_fun(val):
            i, goal_reached = val[0], val[1]
            return (i < self.rrt_samples) # & ~goal_reached
        
        def _body_fun(val):
            #jax.debug.print('body start')
            i, goal_reached, rrt_idx, tree, key = val
            key, key_s = jax.random.split(key)
            #jax.debug.print('rrt idx {r}', r=rrt_idx)
            # Sample position, find closest idx and increment towards sampled pos
            sampled_pos = jax.random.uniform(key_s, (2,), minval=low_lim, maxval=high_lim)
            distances = jnp.linalg.norm(tree[:, :2] - sampled_pos, axis=1)
            distance_mask = jnp.where(jnp.arange(self.rrt_samples) < rrt_idx, 0, 1) * INF

            closest_idx = jnp.argmin(distances + distance_mask)
            #jax.debug.print('closest idx {c}, d {d}', c=closest_idx, d=distances + distance_mask)
            tree_pos = tree[closest_idx, :2]
            step_size = jnp.minimum(self.rrt_step_size, jnp.linalg.norm(sampled_pos - tree_pos))
            
            angle = jnp.arctan2(sampled_pos[1] - tree_pos[1], sampled_pos[0] - tree_pos[0])
            test_pos = tree_pos + jnp.array([jnp.cos(angle), jnp.sin(angle)]) * step_size
            
            # Check free space, line collision
            free_space = ~self.check_agent_map_collision(test_pos, map_data)
            circle_trans = self.check_agent_translation(test_pos, tree_pos, map_data) # NOTE do we need this for our valid check?
            valid = free_space & circle_trans
            #goal_reached = jnp.array_equal(jnp.floor(test_pos), goal_square)
            goal_just_reached = jnp.linalg.norm(test_pos - goal) < self.goal_radius
            #jax.debug.print('valid {v}, free space {f}, circle trans {c}', v=valid, f=free_space, c=circle_trans)

            # Find parent
            tree_dist = jnp.linalg.norm(tree[:, :2] - test_pos, axis=1)  # todo correct for zeros
            distance_mask = jnp.where(jnp.arange(self.rrt_samples) < rrt_idx, 0, 1) * INF
            #jax.debug.print('tree dist {t}', t=tree_dist.shape)
            parent_poss = jnp.argsort(tree_dist+distance_mask)[:rrt_star_neighbours]

            #jax.debug.print('parent poss {p}', p=parent_poss)
            in_range = jnp.where(parent_poss < rrt_idx, True, False)
            invalid_parent = ~check_point_connections(test_pos, tree[parent_poss, :2], map_data) | ~in_range
            #print('invalid parent', invalid_parent)
            #jax.debug.print('invalid parent {i}, not in range {r}', i=invalid_parent, r=~in_range)
            invalid_parent = invalid_parent | ~in_range
            parent_cost = tree[parent_poss, 3] + tree_dist[parent_poss] + INF*invalid_parent
            parent_rel_idx = jnp.argmin(parent_cost)
            parent_idx = parent_poss[parent_rel_idx]
            cost = jnp.min(parent_cost)
            valid = valid & ~invalid_parent[parent_rel_idx]
            #jax.debug.print('valid {v}, parent valid {p}', v=valid, p=~invalid_parent[parent_rel_idx])
            new_node = jax.lax.select(valid, jnp.concatenate([test_pos, jnp.array([parent_idx, cost, goal_just_reached.astype(jnp.int32)])]), jnp.zeros(tree.shape[1]))
            #jax.debug.print('new node {n}, goal reached {g}', n=new_node, g=goal_just_reached)
            tree = tree.at[rrt_idx].set(new_node)
            
            # rewire
            invalid_child = invalid_parent.at[parent_rel_idx].set(True)
            
            new_cost = cost + tree_dist[parent_poss] + INF*invalid_child
            rewire = (new_cost < tree[parent_poss, 3]) & valid 
            
            new_nodes = jnp.where(rewire[:, None], jnp.concatenate([tree[parent_poss, :2], jnp.full((rrt_star_neighbours, 1), rrt_idx), new_cost[:, None], tree[parent_poss, 4][:, None]], axis=1), tree[parent_poss])
            #jax.debug.print('new nodes {n}', n=new_nodes)
            tree = tree.at[parent_poss].set(new_nodes)
            
            new_goal_node = goal_just_reached & valid

            rrt_idx += 1*valid
            goal_reached = goal_reached | new_goal_node
            return (i+1, goal_reached, rrt_idx, tree, key)
            
        val = jax.lax.while_loop(_cond_fun, _body_fun, (0, goal_reached, rrt_idx, tree, key))
        goal_reached, tree = val[1], val[3]    
        
        return tree, goal_reached
            
    def plot_rrt_tree(self, ax, tree, goal_reached=False, goal=None, rrt_reward=None, name="env"):
        
        for n in range(tree.shape[0]):
            if tree[n, 0] == 0.0: break
            ax.scatter(tree[n, 0], tree[n, 1], c='gray')
            parent_idx = tree[n, 2]
            if parent_idx == -1: continue
            parent_idx = int(parent_idx)
            ax.plot([tree[n, 0], tree[parent_idx, 0]], [tree[n, 1], tree[parent_idx, 1]], c='gray', marker='+', alpha=0.75)
    
                
        if goal_reached:
            if rrt_reward is not None: 
                assert goal is not None            
                rewards = []
            
            path_lengths = []
            goal_idx = jnp.argwhere(tree[:,-1]==1)
            
            for g_idx in goal_idx:
                c_idx = g_idx[0]
                path_length = 0.0
                if rrt_reward is not None: rew = 0.0
                while c_idx != 0:
                    #print('cidx', c_idx, 'tree row', tree[c_idx])
                    c_pos = tree[c_idx, :2]
                    p_idx = int(tree[c_idx, 2])
                    p_pos = tree[p_idx, :2]
                    path_length += jnp.linalg.norm(c_pos - p_pos)
                    if rrt_reward is not None: rew += rrt_reward(c_pos, p_pos, goal)
                    ax.plot([c_pos[0], p_pos[0]], [c_pos[1], p_pos[1]], c='r', alpha=0.25)
                    c_idx = p_idx
                path_lengths.append(path_length)
                #print('path_length:', path_length)
                if rrt_reward is not None: 
                    #print('reward:', rew)
                    rewards.append(rew)
        
            if rrt_reward is not None:
                max_rew_idx = jnp.argmax(jnp.array(rewards))
                print(name, ' max reward:', rewards[max_rew_idx], 'path length:', path_lengths[max_rew_idx])  
            else:
                print(name, ' min path length:', jnp.min(jnp.array(path_lengths)))
    
    def plot_map(self,
                 ax: axes.Axes,
                 map_data: jnp.ndarray,) -> None:
        raise NotImplementedError
            
    def plot_agents(
                self,
                ax: axes.Axes,
                pos: jnp.ndarray,
                theta: jnp.ndarray,
                goal: jnp.ndarray,
                done: jnp.ndarray,
                plot_line_to_goal=True,
                colour_agents_by_idx=False,
        ) -> None:
        raise NotImplementedError
    
    def plot_agent_path(
        self,
        ax: axes.Axes,
        x_seq: jnp.ndarray,
        y_seq: jnp.ndarray,
    ) -> None:
        raise NotImplementedError
     
            
    
        
    
