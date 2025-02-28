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

@pytest.mark.parametrize(
    ("num_agents", "pos", "theta", "map_size", "disable_jit", "outcome"),
    [
        (
            2, 
            jnp.array([[3.5, 3.1],
                       [1.5, 3.1]]),
            jnp.array([jnp.pi/4, 0.0]),
            (5, 5),
            False,
            jnp.array([False, False]),
        ),
        (
            2, 
            jnp.array([[3.5, 3.0],
                       [3.5, 3.2]]),
            jnp.array([0.0, 0.0]),
            (5, 5),
            False,
            jnp.array([True, True]),
        ),
        (
            3, 
            jnp.array([[3.5, 3.0],
                       [3.5, 3.2],
                       [4.5, 3.2]]),
            jnp.array([0.0, 0.0, 0.0]),
            (5, 5),
            False,
            jnp.array([True, True, False]),
        ),
        (
            3, 
            jnp.array([[3.5, 3.1],
                       [4.12, 3.1],
                       [0., 0.25]]),
            jnp.array([jnp.pi/4, 0.0, 0.0]),
            (5, 5),
            False,
            jnp.array([False, False, False]),
        ),
        (
            3, 
            jnp.array([[3.5, 3.1],
                       [4.1, 3.1],
                       [0., 0.25]]),
            jnp.array([jnp.pi/4, 0.0, 0.0]),
            (5, 5),
            False,
            jnp.array([True, True, False]),
        ),
        (
            6, 
            jnp.array([[3.5, 3.1],
                       [4.1, 3.1],
                       [2, 0.25],
                       [6.5, 3.1],
                       [6.1, 3.1],
                       [1.0, 6.0]]),
            jnp.array([jnp.pi/4, 0.0, 0.0, 0.0, -jnp.pi/4, 0.0]),
            (10, 10),
            False,
            jnp.array([True, True, False, True, True, False]),
        ),
    ],
)
def test_square_agent_agent_collisions(
    num_agents,
    pos, 
    theta,
    map_size,
    disable_jit: bool,
    outcome: bool,
):
    with jax.disable_jit(disable_jit):
        map_obj = GridMapPolygonAgents(
            num_agents=num_agents,
            rad=0.3,
            map_size=map_size,
        )
        c = map_obj.check_all_agent_agent_collisions(pos, theta)
        assert jnp.all(c == outcome)