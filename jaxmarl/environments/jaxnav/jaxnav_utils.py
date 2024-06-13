'''
Utility functions for simulators
'''

import jax
import jax.numpy as jnp
import chex
import os, pathlib
import numpy as np
from functools import partial


# map names cannot include an '_'
MAP_PATHS = {
    "blank": "blank_map.npy",
    "blank-small" : "blank-small_map.npy",
    "blank-15": "blank-15.npy",
    "central-square" : "central-square_map.npy",
    "central-square-easy" : "central-square_map.npy",  # sample cases have a lower treshold
    "cross-20": "cross-20_map.npy",
    "circle-20": "circle-20_map.npy",
    "circle-sym-20": "circle-20_map.npy",
    "corridor-10": "corridor-10_map.npy",
    "corridor-15": "corridor-15_map.npy",
    "barn-test" : "barn/barn-test_map.npy",
    "barn-20": "barn/barn-20_map.npy",
    "barn-25": "barn/barn-25_map.npy",
    "barn-30": "barn/barn-30_map.npy",
    "1-wide-c": "corridor/1-wide-c_map.npy",
    "2-wide-c": "corridor/2-wide-c_map.npy",
    "1-wide-b": "corridor/1-wide-b_map.npy",
    "5-wide-chicane": "corridor/5-wide-chicane_map.npy",
}

GRID_HALF_HEIGHT = 0.5

### --- MATHS UTILS --- 
def pol2cart(rho: float, phi: float) -> chex.Array:
    ''' Convert polar coordinates into cartesian '''
    x = rho * jnp.cos(phi)
    y = rho * jnp.sin(phi)
    return jnp.array([x,y])

def cart2pol(x, y) -> chex.Array:
    rho = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    return jnp.array([rho, phi])
    
def unitvec(theta) -> chex.Array:
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])

def wrap(angle):
    ''' Ensure angle lies in the range [-pi, pi] '''
    large = lambda x: x - 2*jnp.pi
    small = lambda x: x + 2*jnp.pi
    noChange = lambda x: x
    wrapped_angle = jax.lax.cond(angle >= jnp.pi,
        large, noChange, angle)
    wrapped_angle = jax.lax.cond(angle < -jnp.pi,
        small, noChange, wrapped_angle)
    
    return wrapped_angle

def euclid_dist(x, y):
    return jnp.norm(x-y)

def rot_mat(theta):
    """ 2x2 rotation matrix for 2D about the origin """
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]).squeeze()


### --- ENV UTILS
@jax.jit
def map_collision(pos: chex.Array, map_grid: chex.Array, radius: float) -> bool:
    """ For a circle agent, ASSUMES radius<1 and grids of size 1x1 """
    # Calculates which grid square robot overlaps in
    min_x, min_y = jnp.floor(jnp.maximum(jnp.zeros(2), pos-radius)).astype(int)
    max_x, max_y = jnp.floor(jnp.minimum(jnp.array(map_grid.shape), pos+radius)).astype(int)
    
    map_c_list = jnp.array([
        [min_x, min_y],
        [max_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ]) + GRID_HALF_HEIGHT
        
    grid_check = check_grid(map_c_list, pos, radius)
    
    map_occ = jnp.array([
        map_grid[min_y, min_x],
        map_grid[min_y, max_x],
        map_grid[max_y, min_x],
        map_grid[max_y, max_x],
    ]).astype(int)
    return jnp.any((map_occ+grid_check)>1)

@partial(jax.vmap, in_axes=[0, None, None])
def check_grid(c, pos, radius):
    p = jnp.clip(pos - c, -GRID_HALF_HEIGHT, GRID_HALF_HEIGHT) 
    p = p + c 
    return jnp.linalg.norm(p - pos) <= radius
    
'''#@partial(jax.jit)
def check_square_plot(pos, c, radius):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots()
    hh = 0.5

    p = jnp.clip(pos - c, -hh, hh) 
    
    print('p', p)
    p = p + c
    print('p', p)
    d = p - pos
    print('d', jnp.linalg.norm(d))
    
    
    square = plt.Rectangle((c[0] - hh, c[1] - hh), hh*2, hh*2, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(square)
    
    circle = Circle(pos, radius)
    ax.add_patch(circle)
        
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    plt.show()
'''

@partial(jax.jit)
def map_collision_square(pos: chex.Array, map_grid: chex.Array, radius: float) -> bool:
    """ This is for a square agent that cannot rotate """
    # calculate which grid cells robot overlaps in, assumes radius<1 
    min_x, min_y = jnp.floor(jnp.maximum(jnp.zeros(2), pos-radius)).astype(int)
    max_x, max_y = jnp.floor(jnp.minimum(jnp.array(map_grid.shape), pos+radius)).astype(int)
    rg = jnp.zeros(map_grid.shape, dtype=int)
    
    rg = rg.at[min_y, min_x].set(1)
    rg = rg.at[min_y, max_x].set(1)
    rg = rg.at[max_y, min_x].set(1)
    rg = rg.at[max_y, max_x].set(1)

    return jnp.any((rg+map_grid)>1)


### --- LOADING FILES ---
def load_map(name: str) -> chex.Array:
    ''' Load map using jnp
    Possible map names
     - blank: 30x30 blank map
    '''
    print('load map: ', name)
    return load_map_array(MAP_PATHS[name])


def load_max_cases(map_name: str, num_agents: int):
    prefix = f"{map_name}_{num_agents}_agents_"
    parent_dir_path = pathlib.Path(__file__).parent.resolve()
    dir_path = os.path.join(parent_dir_path, pathlib.Path(f"sample_cases/{map_name}/"))
    prefixed = [filename for filename in os.listdir(dir_path) if filename.startswith(prefix)]

    num_cases = max([p.split("_")[3] for p in prefixed])

    return load_cases(map_name, num_agents, num_cases)


def load_cases(map_name: str, num_agents: int, num_cases: int):
    
    filename = f"sample_cases/{map_name}/{map_name}_{num_agents}_agents_{num_cases}_cases.npy"
    parent_dir_path = pathlib.Path(__file__).parent.resolve()
    filepath = os.path.join(parent_dir_path, filename)
    return jnp.load(filepath)
    
    
def load_map_array(filename: str) -> chex.Array:
    parent_dir_path = pathlib.Path(__file__).parent.resolve()
    return jnp.load(os.path.join(parent_dir_path, pathlib.Path("maps/" + filename)))




if __name__ == "__main__":
    
    pos = jnp.array([0.9, 0.9])
    c = jnp.array([1.5, 1.5])
    rad = 0.3

    map_grid = jnp.zeros((5, 5))
    map_grid = map_grid.at[1:,1:].set(1)
    
    map_collision(pos, map_grid, rad)    
    
    