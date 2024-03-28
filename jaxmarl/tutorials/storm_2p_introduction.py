from PIL import Image
import os
import jax
import jax.numpy as jnp

from jaxmarl import make
from jaxmarl.environments.storm.storm_2p import Items

action = 1
render_agent_view = True
num_outer_steps = 3
num_inner_steps = 152

rng = jax.random.PRNGKey(0)

env = make('storm_2p', 
        num_inner_steps=num_inner_steps, 
        num_outer_steps=num_outer_steps, 
        fixed_coin_location=True,
        num_agents=2,
        payoff_matrix=jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]]),
        freeze_penalty=5,)

num_actions = env.action_space().n

obs, old_state = env.reset(rng)
pics = []
pics1 = []
pics2 = []

img = env.render(old_state)
img1 = env.render_agent_view(old_state, agent=0)
img2 = env.render_agent_view(old_state, agent=1)
pics.append(img)

int_action = {
    0: "left",
    1: "right",
    2: "forward",
    3: "interact",
    4: "stay",
}

key_int = {"w": 2, "a": 0, "s": 4, "d": 1, " ": 4}
env.step_env = jax.jit(env.step_env)

for t in range(num_outer_steps * num_inner_steps):
    rng, rng1, rng2 = jax.random.split(rng, 3)
    # a1 = jnp.array(2)
    # a2 = jnp.array(4)
    a1 = jax.random.choice(
        rng1, a=num_actions, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.4])
    )
    a2 = jax.random.choice(
        rng2, a=num_actions, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.4])
    )
    obs, state, reward, done, info = env.step_env(
        rng, old_state, (a1 * action, a2 * action)
    )

    print('outer t', state.outer_t)
    print('inner t', state.inner_t)
    print('done', done)
    if (state.red_pos[:2] == state.blue_pos[:2]).all():
        import pdb

        # pdb.set_trace()
        print("collision")
        print(
            f"timestep: {t}, A1: {int_action[a1.item()]} A2:{int_action[a2.item()]}"
        )
        print(state.red_pos, state.blue_pos)

    img = env.render(state)
    pics.append(img)

    if render_agent_view:
        img1 = env.render_agent_view(state, agent=0)
        img2 = env.render_agent_view(state, agent=1)
        pics1.append(img1)
        pics2.append(img2)
    old_state = state
print("Saving GIF")
pics = [Image.fromarray(img) for img in pics]
pics[0].save(
    "state.gif",
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=100,
    loop=0,
)

if render_agent_view:
    pics1 = [Image.fromarray(img) for img in pics1]
    pics2 = [Image.fromarray(img) for img in pics2]
    pics1[0].save(
        "agent1.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics1[1:],
        duration=100,
        loop=0,
    )
    pics2[0].save(
        "agent2.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics2[1:],
        duration=100,
        loop=0,
    )