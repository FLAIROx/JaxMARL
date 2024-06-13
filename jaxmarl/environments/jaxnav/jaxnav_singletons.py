import jax 
import jax.numpy as jnp
import chex
from typing import NamedTuple, List, Tuple
import matplotlib.pyplot as plt

from .jaxnav_env import JaxNav, State
from .maps import make_map

class TestCase(NamedTuple):
    map_data: list
    start_pose: tuple
    goal_pose: tuple

class JaxNavSingleton(JaxNav):
    def __init__(self,
                num_agents: int, # Number of agents
                test_case=None,
                fixed_lambda=False,
                rew_lambda=0.0,
                map_id="Grid-Rand",
                **env_kwargs
    ):
        assert len(test_case.start_pose) == num_agents, f"len start_pose: {len(test_case.start_pose)} != num_agents: {num_agents}"
        assert len(test_case.goal_pose) == num_agents
        assert map_id.startswith("Grid"), f"map_id: {map_id} does not start with Grid"

        super().__init__(num_agents,
                         map_id=map_id,
                         **env_kwargs)
        
        if fixed_lambda is True:
            self.rew_lambda = rew_lambda
        else:   
            self.rew_lambda = 0.0
        
        if test_case is None:
            raise NotImplementedError
        else:
            if map_id == "Grid-Rand-Poly-Single":
                map_id = "Grid-Rand-Poly"
            self.map_data = jnp.array(
                [[int(x) for x in row.split()] for row in test_case.map_data], 
                dtype=jnp.int32
            )
            height, width = self.map_data.shape
            self.goal_pose = jnp.array(test_case.goal_pose, dtype=jnp.float32)
            self.start_pose = jnp.array(test_case.start_pose, dtype=jnp.float32)
            
            map_kwargs = {
                "map_size": (width, height),
                "fill": 0.5,
            }
            self._map_obj = make_map(map_id, num_agents, self.rad, **map_kwargs)
                
    def reset(
        self, 
        key: chex.PRNGKey=None,
    ):
        
        state = State(
            pos=self.start_pose[:, :2],
            theta=self.start_pose[:, 2],
            vel=jnp.zeros((self.num_agents, 2)),
            done=jnp.full((self.num_agents,), False),
            term=jnp.full((self.num_agents,), False),
            goal_reached=jnp.full((self.num_agents,), False),
            move_term=jnp.full((self.num_agents,), False),
            goal=self.goal_pose[:, :2],
            ep_done=False,
            step=0,
            map_data=self.map_data,
            rew_lambda=self.rew_lambda,
        )
        obs_batch = self._get_obs(state)
        obs = {a: obs_batch[i] for i, a in enumerate(self.agents)}
        return obs, state
    
    def get_monitored_metrics(self):
        return ["NumC", "GoalR", "AgentC", "MapC", "TimeO", "Return"]
        
    def viz_testcase(self, save=True, show=False, plot_lidar=True):
        
        obs, state = self.reset()
        fig, ax = plt.subplots(figsize=(5,5))
        
        ax.set_aspect('equal', 'box')
        self._map_obj.plot_map(ax, state.map_data)
        ax.scatter(state.goal[:, 0], state.goal[:, 1], marker='+')
        self._map_obj.plot_agents(ax, state.pos, state.theta, state.goal, state.done)
        if plot_lidar:
            self.plot_lidar(ax, obs, state, 100)
        
        # plot a line from start to goal for each agent
        for i in range(self.num_agents):
            ax.plot(jnp.concatenate([state.pos[i, 0][None], state.goal[i, 0][None]]), 
                    jnp.concatenate([state.pos[i, 1][None], state.goal[i, 1][None]]), 
                    color='gray', alpha=0.2)
                                
        if save:
            plt.savefig(f'{self.name}.png')
        if show:
            plt.show()
            
    @property
    def name(self) -> str:
        return self.__class__.__name__
            

## SINGLE AGENT
# blank map
class BlankTest(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=1, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 1 1 1 1 1 1",
            ],
            start_pose=[(1.5, 1.5, 0.0)],
            goal_pose=[(5.5, 5.5, 0.0)],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)

class MiddleTest(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=1, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1",
                "1 0 0 0 0 0 1",
                "1 0 1 1 1 0 1",
                "1 0 1 1 1 0 1",
                "1 0 1 1 1 0 1",
                "1 0 0 0 0 0 1",
                "1 1 1 1 1 1 1",
            ],
            start_pose=[(1.5, 1.5, 0.0)],
            goal_pose=[(5.5, 5.5, 0.0)],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)

## MULTI-AGENT
class BlankCross2(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=2, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 1 1 1 1 1 1",
            ],
            start_pose=[(1.5, 1.5, 0.78),
                        (5.5, 5.5, 3.92)],
            goal_pose=[(5.5, 5.5, 0.0),
                       (1.5, 1.5, 0.0)],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)
        
class BlankCross4(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=4, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 1 1 1 1 1 1",
            ],
            start_pose=[(1.5, 1.5, 0.78),
                        (5.5, 5.5, 3.92),
                        (1.5, 5.5, -0.78),
                        (5.5, 1.5, -3.92)],
            goal_pose=[(5.5, 5.5, 0.0),
                       (1.5, 1.5, 0.0),
                       (5.5, 1.5, 0.0),
                       (1.5, 5.5, 0.0)],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)
        
class CircleCross(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=10,
        circle_rad=6,
        **env_kwargs
    ):
        
        width, height = circle_rad*3, circle_rad*3
        centre_x = width/2
        centre_y = height/2
        top = "1 " * int(width)
        row = "1 " + "0 " * int(width-2) + "1"
        rows = [row for _ in range(int(height)-2)]
        map_data = [top] + rows + [top]
        
        start_pose = []
        goal_pose = []
        for i in range(num_agents):
            theta = 2*jnp.pi * i / num_agents
            to_center_theta = jnp.pi + theta
            start_pose.append((circle_rad*jnp.cos(theta)+centre_y, circle_rad*jnp.sin(theta)+centre_x, to_center_theta))
            goal_theta = theta + jnp.pi
            goal_pose.append((circle_rad*jnp.cos(goal_theta)+centre_y, circle_rad*jnp.sin(goal_theta)+centre_x, goal_theta))
        
        test_case = TestCase(
            map_data=map_data,
            start_pose=start_pose,
            goal_pose=goal_pose,
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)

class BlankCrossUneven2(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=2, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 0 0 0 0 0 1",
                "1 1 1 1 1 1 1",
            ],
            start_pose=[(2.0, 2.5, 0.78),
                        (5.5, 5.5, 3.92)],
            goal_pose=[(5.5, 5.5, 0.0),
                       (1.5, 1.5, 0.0)],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)

class SingleNav1(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=1, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1 1",
                "1 1 0 0 0 0 0 0 0 0 1",
                "1 0 0 1 1 0 1 1 1 0 1",
                "1 1 0 0 0 0 0 0 0 0 1",
                "1 1 0 1 1 0 1 1 1 0 1",
                "1 0 0 0 0 0 0 1 0 0 1",
                "1 1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(9.5, 1.5, 0.78)],
            goal_pose=[(1.5, 5.5, 0.0)],
        )
                
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)
        
class SingleNav2(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=1, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1 1",
                "1 0 0 1 1 1 1 1 0 0 1",
                "1 0 0 0 1 0 0 1 1 0 1",
                "1 0 1 0 1 0 0 0 0 0 1",
                "1 0 1 0 0 0 0 1 0 0 1",
                "1 0 1 0 1 0 0 0 0 0 1",
                "1 1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(8.5, 1.5, 3.14)],
            goal_pose=[(1.5, 5.5, 0.0)],
        )
                
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)
        
class SingleNav3(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=1, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1 1",
                "1 0 0 1 1 1 1 1 0 0 1",
                "1 0 0 0 1 0 0 1 1 0 1",
                "1 0 1 0 1 0 0 0 0 0 1",
                "1 0 1 0 0 0 0 1 0 0 1",
                "1 0 1 0 1 0 0 0 0 0 1",
                "1 1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(9.1, 1.5, 3.14)],
            goal_pose=[(1.5, 5.5, 0.0)],
        )
                
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)
        
class LongCorridor2(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=2, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1",
                "1 1 1 1 1 1 1 0 0 1",
                "1 0 0 1 1 1 1 0 0 1",
                "1 0 0 0 0 0 0 0 0 1",
                "1 0 0 1 1 1 1 0 0 1",
                "1 1 1 1 1 1 1 0 0 1",
                "1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(1.5, 3.5, 0.0),
                        (8.0, 3.5, 3.14),],
            goal_pose=[(6.0, 3.5, 0.0),
                       (1.5, 3.5, 0.0),],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)        
        
class Corridor4(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=4, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 0 0 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(2.0, 2.5, 0.0),
                        (2.0, 4.5, 0.0),
                        (8.0, 2.5, 3.14),
                        (8.0, 4.5, 3.14),],
            goal_pose=[(8.0, 2.5, 3.14),
                       (8.0, 4.5, 3.14),
                       (2.0, 2.5, 0.0),
                       (2.0, 4.5, 0.0),],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)
        
class Corridor8(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=8, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 0 0 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 1",
                "1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(2.0, 2.0, 0.0),
                        (2.0, 3.5, 0.0),
                        (2.0, 5.5, 0.0),
                        (2.0, 7.0, 0.0),
                        (8.0, 2.0, 3.14),
                        (8.0, 3.5, 3.14),
                        (8.0, 5.5, 3.14),
                        (8.0, 7.0, 3.14),],
            goal_pose=[(8.0, 2.0, 3.14),
                       (8.0, 3.5, 3.14),
                       (8.0, 5.5, 3.14),
                       (8.0, 7.0, 3.14),
                       (2.0, 2.0, 0.0),
                       (2.0, 3.5, 0.0),
                       (2.0, 5.5, 0.0),
                       (2.0, 7.0, 0.0),],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)

class Layby4(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=4, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1",
                "1 0 0 1 1 1 1 0 0 1",
                "1 0 0 1 0 1 1 0 0 1",
                "1 0 0 0 0 0 0 0 0 1",
                "1 0 0 1 1 0 1 0 0 1",
                "1 0 0 1 1 1 1 0 0 1",
                "1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(2.0, 2.5, 0.0),
                        (2.0, 4.5, 0.0),
                        (8.0, 2.5, 3.14),
                        (8.0, 4.5, 3.14),],
            goal_pose=[(8.0, 2.5, 3.14),
                       (8.0, 4.5, 3.14),
                       (2.0, 2.5, 0.0),
                       (2.0, 4.5, 0.0),],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)

class Corner2(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=2, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1",
                "1 0 0 0 0 0 1",
                "1 1 1 1 0 0 1",
                "1 1 1 1 1 0 1",
                "1 1 1 1 1 0 1",
                "1 1 1 1 1 0 1",
                "1 1 1 1 1 1 1",
            ],
            start_pose=[(1.5, 1.5, 0.0),
                        (5.5, 5.5, -1.57),],
            goal_pose=[(5.5, 5.5, 3.14),
                       (1.5, 1.5, 3.14),],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)

class Chicane2(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=2, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1 1 1 1 1",
                "1 0 0 0 0 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 0 0 0 0 1",
                "1 1 1 1 1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(2.5, 2.5, 0.0),
                        (11.5, 2.5, 3.14),],
            goal_pose=[(9.5, 3.5, 3.14),
                       (2.5, 2.5, 3.14),],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)
        
class NarrowChicane2a(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=2, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1 1 1 1 1",
                "1 0 0 0 0 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 1 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 1 0 0 0 0 0 0 1",
                "1 1 1 1 1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(2.5, 2.5, 0.0),
                        (11.5, 2.5, 3.14),],
            goal_pose=[(9.5, 3.5, 3.14),
                       (2.5, 2.5, 3.14),],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)
        
class NarrowChicane2b(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=2, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1 1 1 1 1",
                "1 0 0 0 0 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 1 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 1 0 0 0 0 0 0 1",
                "1 1 1 1 1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(2.5, 1.5, 0.0),
                        (11.5, 2.5, 3.14),],
            goal_pose=[(7.5, 1.5, 3.14),
                       (2.5, 2.5, 3.14),],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)

class Chicane4(JaxNavSingleton):
    def __init__(
        self, 
        num_agents=4, 
        **env_kwargs
    ):
        test_case = TestCase(
            map_data = [
                "1 1 1 1 1 1 1 1 1 1 1 1 1 1",
                "1 0 0 0 0 0 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 1 1 0 0 0 1",
                "1 0 0 0 1 1 0 0 0 0 0 0 0 1",
                "1 1 1 1 1 1 1 1 1 1 1 1 1 1",
            ],
            start_pose=[(2.5, 1.75, 0.0),
                        (2.5, 3.25, 0.0),
                        (11.5, 1.75, 3.14),
                        (11.5, 3.25, 3.14),],
            goal_pose=[(9.5, 3.5, 3.14),
                       (11.5, 1.75, 3.14),
                       (2.5, 3.25, 3.14),
                       (4.5, 1.5, 3.14),],
        )
        
        super().__init__(num_agents,
                         test_case=test_case,
                         **env_kwargs)

# REGISTRATION
def make_jaxnav_singleton(env_id: str, **env_kwargs) -> JaxNavSingleton:  
    if env_id not in registered_singletons:
        raise ValueError(f"Singleton env_id: {env_id} not registered!")
    if env_id == "BlankTest":
        return BlankTest(**env_kwargs)
    if env_id == "MiddleTest":
        return MiddleTest(**env_kwargs)
    
    if env_id == "BlankCross2":
        return BlankCross2(**env_kwargs)
    if env_id == "BlankCross4":
        return BlankCross4(**env_kwargs)
    if env_id == "BlankCrossUneven2":
        return BlankCrossUneven2(**env_kwargs)
    if env_id == "CircleCross":
        return CircleCross(**env_kwargs)
    if env_id == "Corridor4":
        return Corridor4(**env_kwargs)
    if env_id == "Corridor8":
        return Corridor8(**env_kwargs)
    if env_id == "LongCorridor2":
        return LongCorridor2(**env_kwargs)
    if env_id == "Layby4":
        return Layby4(**env_kwargs)
    if env_id == "Corner2":
        return Corner2(**env_kwargs)
    if env_id == "Chicane2":
        return Chicane2(**env_kwargs)
    if env_id == "Chicane4":
        return Chicane4(**env_kwargs)
    if env_id == "NarrowChicane2a":
        return NarrowChicane2a(**env_kwargs)
    if env_id == "NarrowChicane2b":
        return NarrowChicane2b(**env_kwargs)
    
    if env_id == "SingleNav1":
        return SingleNav1(**env_kwargs)
    if env_id == "SingleNav2":
        return SingleNav2(**env_kwargs)
    if env_id == "SingleNav3":
        return SingleNav3(**env_kwargs)
    
    raise ValueError(f"Map: {env_id} not registered correctly!")

registered_singletons = [
    "BlankTest",
    "MiddleTest",
    "BlankCross2",
    "BlankCross4",
    "BlankCrossUneven2",
    "CircleCross",
    "SingleNav1",
    "SingleNav2",
    "SingleNav3",
    "Corridor4",
    "LongCorridor2",
    "Layby4",
    "Corner2",
    "Corridor8",
    "Chicane2",
    "Chicane4",
    "NarrowChicane2a",
    "NarrowChicane2b",
]

def make_jaxnav_singleton_collection(collection_id: str, **env_kwargs) -> Tuple[List[JaxNavSingleton], List[str]]:
    
    env_ids = registered_singleton_collections[collection_id]
    envs = []
    for env_id in env_ids:
        envs.append(make_jaxnav_singleton(env_id, **env_kwargs))
        
    return envs, env_ids

registered_singleton_collections = {
    "test": [
        "BlankTest",
    ],
    "multi":  [
        "CircleCross",
        "BlankCross4",
        "BlankCrossUneven2",
        "Corridor4",
        "LongCorridor2",
        "Layby4",
        "Corner2",
        "SingleNav1",
        "SingleNav2",
        "SingleNav3",
    ],
    "single": [
        "BlankTest",
        "MiddleTest",
        "SingleNav1",
        "SingleNav2",
        "SingleNav3"
    ],
    "hard": [
        "SingleNav2",
        "Layby4",
        "Corner2",
        "Corridor4",
        "Corridor8",
        "CircleCross",
        "NarrowChicane2a",
        "NarrowChicane2b",
        "Chicane4",
    ],
    "new": [
        "NarrowChicane2a",
        "NarrowChicane2b",
        "Chicane4",
    ],
    "corridor": [
        "BlankCross4",
        "LongCorridor2",
        "Corridor4",
        "Corner2",
    ],
    "just-long-corridor": [
        "LongCorridor2",
    ],
    "just-single2": [
        "SingleNav2",
    ],
}
    