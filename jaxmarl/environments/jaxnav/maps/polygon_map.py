""" 
NOT USED IN A LONG TIME

collision code drawn from: https://www.jeffreythompson.org/collision-detection/poly-circle.php
"""

import jax
import jax.numpy as jnp
from functools import partial

from .map import Map
from jaxmarl.environments.jaxnav.jaxnav_utils import rot_mat

class PolygonMap(Map):
    
    def __init__(self, 
                 num_agents,
                 rad,
                 map_size,
                 num_sides=4,
                 max_num_shapes=40,
                 min_edge_length=0.2,
                 max_edge_length=4.0): 
        super().__init__(num_agents, rad, map_size)
        print('Super old code beware')
        self.num_sides = num_sides
        self.max_num_shapes = max_num_shapes
        self.min_edge_length = min_edge_length
        self.max_edge_length = max_edge_length
        
        if self.num_sides != 4:
            raise NotImplementedError('Only 4-sided polygons are currently supported')
        self.shape_fn = self._sample_parrallelogram
       
    ### === MAP GENERATION === ### 
    @partial(jax.jit, static_argnums=[0])
    def sample_map(self, key):
        """ Sample polygon map, returns coordinates of vertices """
        key_num, key_coords = jax.random.split(key)
    
        # rectangle for map bounds
        bounds = jnp.array([
            [[0.0, 0.15], [self.map_size[0], 0.15], [self.map_size[0], 0.0], [0.0, 0.0]],
            [[0.15, 0.0], [0.15, self.map_size[1]], [0.0, self.map_size[1]], [0.0, 0.0]],
            [[0.0, self.map_size[1]], [self.map_size[0], self.map_size[1]], [self.map_size[0], self.map_size[1]-0.15], [0.0, self.map_size[1]-0.15]],
            [[self.map_size[0]-0.15, 0.0], [self.map_size[0]-0.15, self.map_size[1]], [self.map_size[0],self.map_size[1]], [self.map_size[0], 0.0]],
        ])
        
        
        # out of bounds constant
        oobounds = jnp.full((self.num_sides, 2), 1.0, dtype=jnp.float32) + jnp.array([self.map_size[0], self.map_size[1]])
        
        num_shapes = jax.random.randint(key_num, (1,), minval=0, maxval=self.max_num_shapes)
        p_idx = jnp.arange(0, self.max_num_shapes)
        p_mask = jnp.where(p_idx<=num_shapes, 0, 1)
            
        key_coords = jax.random.split(key_coords, self.max_num_shapes)
        coords = self.shape_fn(key_coords)
        
        mask = p_mask[:, None, None] * jnp.repeat(oobounds[None], self.max_num_shapes, axis=0)
        coords = coords + mask
        return jnp.append(coords, bounds, axis=0)
    
    ### === COLLISION DETECTION === ###
    @partial(jax.jit, static_argnums=[0])
    def check_agent_map_collision(self, pos, coords):
        """ Check collision between a line and a circle subject to a map boundary """
        #print('agent check coords shape', coords.shape)

        @partial(jax.vmap, in_axes=(0, 0, None))
        def _check_ac(p1, p2, pos):
        
            def _in_map(p1, p2, pos, r):
                return (p1[0] >= 0) & (p1[0] <= self.map_size[0]) & (p1[1] >= 0) & (p1[1] <= self.map_size[1]) & \
                    (p2[0] >= 0) & (p2[0] <= self.map_size[0]) & (p2[1] >= 0) & (p2[1] <= self.map_size[1])
                
            return jax.lax.cond(_in_map(p1, p2, pos, self.rad), lambda _: line_circle_collision(p1, p2, pos, self.rad), lambda _: False, None)
    
        edges = self.gen_edges(coords)
        return jnp.any(_check_ac(edges[:, 0], edges[:, 1], pos))
    
    @partial(jax.jit, static_argnums=[0])
    def check_map_collision(self, pos, coords, radius):
        """ For a circle agent """
        l = self.check_agent_map_collision(pos, coords)
        i = jnp.any(self.check_point_map_collision(pos, coords))
        return l | i
    
    # NOTE need to add ray tracing for lidar - constrains size of usable polygon
    
    @partial(jax.jit, static_argnums=[0])  
    def check_point_map_collision(self, pos, coords):
        """ Check if a point is inside a polygon within a map, 
        NOTE: all map coords should be passed"""
        
        @partial(jax.vmap, in_axes=[None, 0, 0])
        def _vc(pos, vc, vn):
            c = (((vc[1] > pos[1]) & (vn[1] < pos[1])) |\
                ((vc[1] < pos[1]) & (vn[1] > pos[1]))) & \
                (pos[0] < (vn[0] - vc[0]) * (pos[1] - vc[1]) / (vn[1] - vc[1]) + vc[0])

            return c 
        
        coords = jnp.concatenate((coords, coords[:, 0][:, None]), axis=1)
        return jnp.sum(_vc(pos, coords[:, :-1].reshape(-1, 2), coords[:, 1:].reshape(-1, 2))) % 2 != 0
         
    ### === UTILS === ### 
    def gen_edges(self, coords):
        """ Generate edges from coordinates and flatten array"""
        coordsp = jnp.append(coords, coords[:, 0].reshape(-1, 1, 2), axis=1)
        return jnp.column_stack((coordsp[:, :-1], coordsp[:, 1:])).reshape((-1, 2, 2))
    
    
    @partial(jax.vmap, in_axes=[None, 0])
    def _sample_parrallelogram(self, key):
        """ Sample a parralelogram from a uniform distribution """
        
        # Sample parralelogram parameters
        key_c, key_o, key_s, key_a = jax.random.split(key, 4)
        centre = jax.random.uniform(key_c, (1, 2), minval=0.0, maxval=self.map_size[1])
        orientation = jax.random.uniform(key_o, (1,), minval=-jnp.pi, maxval=jnp.pi)
        a, b = jax.random.uniform(key_s, (2,), minval=self.min_edge_length, maxval=self.max_edge_length)
        alpha = jax.random.uniform(key_a, (1,), minval=0.1, maxval=jnp.pi)
        
        # Calculate diagonals & angles
        q = jnp.sqrt(a**2 + b**2 + 2*a*b*jnp.cos(alpha))
        p = jnp.sqrt(a**2 + b**2 - 2*a*b*jnp.cos(alpha))
        beta = jnp.arcsin(jnp.sin(alpha)*a/p)
        gamma = jnp.arcsin(jnp.sin(beta)/(q/2)*(p/2))
        
        # Calculate coordinates
        d = - p/2 * jnp.array([jnp.cos(beta), jnp.sin(-beta)]).reshape((1,2))
        b = + p/2 * jnp.array([jnp.cos(-beta), jnp.sin(-beta)]).reshape((1,2))
        c = + q/2 * jnp.array([jnp.cos(gamma), jnp.sin(gamma)]).reshape((1,2))
        a = - q/2 * jnp.array([jnp.cos(gamma), jnp.sin(gamma)]).reshape((1,2))

        return jnp.dot(jnp.array([a, b, c, d]), rot_mat(orientation)).squeeze() + centre

    ### === VISUALISATION === ###
    def plot_map(self, ax, coord):
        coord = jnp.append(coord, coord[:, 0].reshape(-1, 1, 2), axis=1)
        for c in coord:
            xs, ys = zip(*c)    
            ax.plot(xs, ys, color='black')


### === COLLISION DETECTION UTILS === ### NOTE likely a better file for these
@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def line_circle_map_collision(p1, p2, c, r, map_size):
    """ Check collision between a line and a circle subject to a map boundary """
    
    def _in_map(p1, p2, c, r):
        return (p1[0] >= 0) & (p1[0] < map_size[0]) & (p1[1] >= 0) & (p1[1] < map_size[1]) & \
               (p2[0] >= 0) & (p2[0] < map_size[0]) & (p2[1] >= 0) & (p2[1] < map_size[1])
               
    return jax.lax.cond(_in_map(p1, p2, c, r), lambda _: line_circle_collision(p1, p2, c, r), lambda _: False, None)
    

def line_circle_collision(p1, p2, c, r):
    """ Check collision between a line and a circle """
    
    cc = point_circle_collision(p1, c, r) | point_circle_collision(p2, c, r)
    
    def _on_segement(p1, p2, c, r):
        d1 = jnp.linalg.norm(p1 - c)
        d2 = jnp.linalg.norm(p2 - c)
        dl = jnp.linalg.norm(p1 - p2)
        
        return (d1+d2 >= dl-r) & (d1+d2 <= dl+r)
    
    def _line_circle(p2, p1, c, r):
    
        dx, dy = p2 - p1
        len = jnp.sqrt(dx**2 + dy**2)
        dot = ((c[0] - p1[0]) * dx + (c[1] - p1[1]) * dy) / len**2
        closest = jnp.array([p1[0] + dot * dx, p1[1] + dot * dy])
        
        # check if closest on line segment
        on_seg = _on_segement(p1, p2, c, r)
        
        dist = closest - c
        return (jnp.sqrt(dist[0]**2 + dist[1]**2) <= r) & (on_seg)
    
    return jax.lax.cond(cc, lambda _: True, lambda _: _line_circle(p1, p2, c, r), None)

def point_circle_collision(p, c, r):
    """ Check collision between a point and a circle """
    dist = p - c
    return jnp.sqrt(dist[0]**2 + dist[1]**2) <= r    


if __name__=="__main__":
    key = jax.random.PRNGKey(10)
    
    map_size = (10, 10)
    rad = 0.3
    
    map_gen = PolygonMap(2, rad, map_size)
    map_coords = map_gen.sample_map(key)
    
    print('map coords', map_coords[0])
    
    pos = jnp.array([5.3, 6.5])
    #c = map_gen.check_point_collision(pos, map_coords[0])
    c = map_gen.check_agent_map_collision(pos, map_coords)
    print('c', c)
    c = map_gen.check_map_collision(pos, map_coords, None)
    print('c', c)
    
    import matplotlib.pyplot as plt
    # from jax_multirobsim.env.sample_cases.create_sample_cases import jax_sample_case
    
    # case = jax_sample_case(key, 2, 0.3, map_size, map_coords, map_fn=map_gen.check_map_collision)
    
    # fig, ax = plt.subplots()
    
    # map_gen.plot_map(ax, map_coords)
    # #plot_sample_case()
    # ax.scatter(pos[0], pos[1], color='red')
    
    # plt.show()
    
    
    