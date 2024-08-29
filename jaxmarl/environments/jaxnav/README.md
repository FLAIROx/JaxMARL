# 🧭 JaxNav 

2D geometric navigation for differential drive robots. Using distances readings to nearby obstacles (mimicing LiDAR readings), the direction to their goal and their current velocity, robots must navigate to their goal without colliding with obstacles. 

## Environment Details

JaxNav was first introduced in ["No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery"](https://www.arxiv.org/abs/2408.15099) with an in-detail specification given in the Appendix.

### Map Types
The default map is square robots of width 0.5m moving within a world with grid based obstacled, with cells of size 1m x 1m. Map cell size can be varied to produce obstacles of higher fidelty or robot strucutre can be changed into any polygon or a circle.

We also include a map which uses polygon obstacles, but note we have not used this code in a while so there may well be issues with it.

### Observation space
By default, each robot receives 200 range readings from a 360-degree arc centered on their forward axis. These range readings have a max range of 6m but no minimum range and are discretised with a resultion of 0.05 m. Alongside these range readings, each robot receives their current linear and angular velocities along with the direction to their goal. Their goal direction is given by a vector in polar form where the distance is either the max lidar range if the goal is beyond their "line of sight" or the actual distance if the goal is within their lidar range. There is no communication between agents.

### Action Space
The environments default action space is a 2D continuous action, where the first dimension is the desired linear velocity and the second the desired angular velocity. Discrete actions are also supported, where the possible combination of linear and angular velocities are discretised into 15 options.

### Reward function
By default, the reward function contains a sparse outcome based reward alongside a dense shaping term.

## Visulisation
Visualiser contained within `jaxnav_viz.py`, with an example below:

```python
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from jaxmarl.environments.jaxnav.jaxnav_viz import JaxNavVisualizer
import jax 

env = JaxNav(num_agents=4)

rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)

obs, env_state = env.reset(_rng)

obs_list = [obs]
env_state_list = [env_state]

for _ in range(10):
    rng, act_rng, step_rng = jax.random.split(rng, 3)
    act_rngs = jax.random.split(act_rng, env.num_agents)
    actions = {a: env.action_space(a).sample(act_rngs[i]) for i, a in enumerate(env.action_spaces.keys())}
    obs, env_state, _, _, _ = env.step(step_rng, env_state, actions)
    obs_list.append(obs)
    env_state_list.append(env_state)
    
viz = JaxNavVisualizer(env, obs_list, env_state_list)
viz.animate("test.gif")
```

## TODOs:
- remove `self.rad` dependence for non circular agents
- more unit tests
- add tests for non-square agents

## Citation
JaxNav was introduced by the following paper, if you use JaxNav in your work please cite it as:

```bibtex
@misc{rutherford2024noregrets,
      title={No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery}, 
      author={Alexander Rutherford and Michael Beukman and Timon Willi and Bruno Lacerda and Nick Hawes and Jakob Foerster},
      year={2024},
      eprint={2408.15099},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.15099}, 
}
```