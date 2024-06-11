import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from jaxmarl.environments.jaxnav.maps.grid_map import GridMapPolygonAgents

@pytest.mark.parametrize(
    ("num_agents", "pos", "theta", "map_data", "cell_size", "disable_jit", "outcome"),
    [
        (
            1, 
            jnp.array([[1.5, 3.1]]),
            jnp.array([jnp.pi/4]),
            jnp.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]),
            1.0, 
            False,
            True,
        ),
        (
            1, 
            jnp.array([[1.5, 3.1]]),
            jnp.array([jnp.pi/4]),
            jnp.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]), 
            1.0,
            True,
            True,
        ),
        (
            1, 
            jnp.array([[3.1, 1.5]]),
            jnp.array([jnp.pi/4]),
            jnp.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]), 
            1.0,
            False,
            False,
        ),
        (
            1, 
            jnp.array([[3.1, 1.5]]),
            jnp.array([jnp.pi/4]),
            jnp.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]), 
            1.0,
            True,
            False,
        ),
        (
            1, 
            jnp.array([[1.5, 2.5]]),
            jnp.array([0.0]),
            jnp.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]), 
            1.0,
            True,
            True,
        ),
        (
            1, 
            jnp.array([[1.5, 2.5]]),
            jnp.array([0.0]),
            jnp.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]), 
            1.0,
            False,
            True,
        ),
        (
            2, 
            jnp.array([[3.1, 1.5],
                       [1.5, 3.1]]),
            jnp.array([jnp.pi/4, 0]),
            jnp.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]), 
            1.0,
            True,
            jnp.array([False, True]),
        ),
        (
            2, 
            jnp.array([[3.1, 1.5],
                       [1.5, 3.1]]),
            jnp.array([jnp.pi/4, 0]),
            jnp.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]), 
            1.0,
            False,
            jnp.array([False, True]),
        ),
    ]
)
def test_square_agent_grid_map_collisions(
    num_agents,
    pos, 
    theta,
    map_data,
    cell_size,
    disable_jit: bool,
    outcome: bool,
):
    with jax.disable_jit(disable_jit):
        map_obj = GridMapPolygonAgents(
            num_agents=num_agents,
            rad=0.3,
            map_size=map_data.shape,
            cell_size=cell_size,
        )
        
        c = jax.vmap(
            map_obj.check_agent_map_collision,
            in_axes=(0, 0, None))(
            pos,
            theta,
            map_data,
        )
        assert jnp.all(c == outcome)
    
    
@pytest.mark.parametrize(
    ("num_agents", "agent_coords", "pos", "theta", "map_data", "cell_size", "disable_jit", "outcome"),
    [
        (
            1, 
            jnp.array([
                [-0.25, -0.25],
                [-0.25, 0.25],
                [0.25, 0.25],
                [0.25, -0.25],
            ]),
            jnp.array([[1.5, 3.1]]),
            jnp.array([jnp.pi/4]),
            jnp.array([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]),
            1.0, 
            False,
            jnp.array([
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]),
        ),
    ]
) 
def test_square_agent_grid_map_occupancy_mask(
    num_agents,
    agent_coords,
    pos,
    theta,
    map_data,
    cell_size,
    disable_jit: bool,
    outcome: jnp.ndarray,
):
    with jax.disable_jit(disable_jit):
        map_obj = GridMapPolygonAgents(
            num_agents=num_agents,
            rad=0.3,
            map_size=map_data.shape,
            cell_size=cell_size,
            agent_coords=agent_coords,
        )
        
        c = jax.vmap(
            map_obj.get_agent_map_occupancy_mask,
            in_axes=(0, 0, None))(
            pos,
            theta,
            map_data,
        )
        assert jnp.all(c == outcome)

if __name__=="__main__":
    
    rng = jax.random.PRNGKey(0)
    
    num_agents = 1
    rad = 0.3
    map_params = {
        "map_size": (10, 10),
        "fill": 0.4
    }
    pos = jnp.array([[1.5, 3.1]])
    theta = jnp.array([-jnp.pi/4]) 
    goal = jnp.array([[9.5, 9.5]])
    done = jnp.array([False])
    
    map_obj = GridMapPolygonAgents(
        num_agents=num_agents,
        rad=rad,
        grid_size=1.0,
        **map_params
    )
    
    map_data = map_obj.sample_map(rng)
    print('map_data: ', map_data)
    
    c = map_obj.check_agent_map_collision(
        pos,
        theta,
        map_data, 
    )
    print('c', c)
    
    with jax.disable_jit(False):
        c = map_obj.get_agent_map_occupancy_mask(
            pos, 
            theta,
            map_data
        )
        print('c', c)
    
    plt, ax = plt.subplots()
    
    map_obj.plot_map(ax, map_data)
    map_obj.plot_agents(ax,
                        pos, 
                        theta,
                        goal,
                        done=done,
                        plot_line_to_goal=False)
    
    plt.savefig('test_map.png')