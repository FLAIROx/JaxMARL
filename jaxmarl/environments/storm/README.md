# Spatial-Temporal Representations of Matrix Games (STORM)

Inspired by the "in the Matrix" games in [Melting Pot 2.0](https://arxiv.org/abs/2211.13746), the [STORM](https://openreview.net/forum?id=54F8woU8vhq) environment expands on matrix games by representing them as grid-world scenarios. Agents collect resources which define their strategy during interactions and are rewarded based on a pre-specified payoff matrix. This allows for the embedding of fully cooperative, competitive or general-sum games, such as the prisoner's dilemma. 

Thus, STORM can be used for studying paradigms such as *opponent shaping*, where agents act with the intent to change other agents' learning dynamics. Compared to the Coin Game or matrix games, the grid-world setting presents a variety of new challenges such as partial observability, multi-step agent interactions, temporally-extended actions, and longer time horizons. Unlike the "in the Matrix" games from Melting Pot, STORM features stochasticity, increasing the difficulty


## Visualisation

We render each timestep and then create a gif from the collection of images. Further examples are provided [here](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/tutorials).

```python
import jax
import jax.numpy as jnp
from PIL import Image
from jaxmarl import make

# load environment
num_agents = 2
rng = jax.random.PRNGKey(18)
env = make('storm', 
        num_inner_steps=512, 
        num_outer_steps=1, 
        num_agents=num_agents, 
        fixed_coin_location=True,
        payoff_matrix=jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]]),
        freeze_penalty=5,)
rng, _rng = jax.random.split(rng)
obs, old_state = env.reset(_rng)


# render each timestep
pics = []
for t in range(512):
    rng, *rngs = jax.random.split(rng, num_agents+1)
    actions = [jax.random.choice(
        rngs[a], a=env.action_space(0).n, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.2])
    ) for a in range(num_agents)]

    obs, state, reward, done, info = env.step_env(
        rng, old_state, [a for a in actions]
    )

    img = env.render(state)
    pics.append(img)

    old_state = state

# create and save gif
pics = [Image.fromarray(img) for img in pics]        
pics[0].save(
    "state.gif",
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=1000,
    loop=0,
)
```