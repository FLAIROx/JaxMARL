from PIL import Image
import os
import jax
import jax.numpy as jnp

from jaxmarl import make
from jaxmarl.environments.storm.storm_env import Items

action=1
render_agent_view = False
num_outer_steps = 3
# num_inner_steps = 68
#num_agents=8
num_agents=2
num_inner_steps=152

rng = jax.random.PRNGKey(18)
env = make('storm', 
        num_inner_steps=num_inner_steps, 
        num_outer_steps=num_outer_steps, 
        num_agents=num_agents, 
        fixed_coin_location=True,
        payoff_matrix=jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]]),
        freeze_penalty=5,)
num_actions = env.action_space(0).n
print('NUM ACTIONS: ', num_actions)
obs, old_state = env.reset(rng)


def pos_remove(i, val):
        grid, old_pos = val
        grid = grid.at[
            (old_pos[i, 0], old_pos[i,1])
        ].set(jnp.int8(Items.empty))
        return (grid, old_pos)

def pos_add(i, val):
        grid, new_pos = val
        grid = grid.at[(new_pos[i,0], new_pos[i,1])].set(
            jnp.int8(i+1)
        )
        return (grid, new_pos)  


# grid, _ = jax.lax.fori_loop(0, num_agents, pos_remove, (old_state.grid, old_state.agent_positions))
# new_agent_pos = jnp.array([[7, 0, 0],
#     [7, 1, 2],
#     [7, 2, 2]], dtype=jnp.int8)
# grid, _ = jax.lax.fori_loop(0, num_agents, pos_add, (grid, new_agent_pos))
# old_state= old_state.replace(agent_positions=new_agent_pos)
# old_state= old_state.replace(agent_inventories=jnp.array([[1., 1.],
#     [1., 1.],
#     [1., 1.]]))
# old_state= old_state.replace(grid=grid)

pics = []
pics1 = []
pics2 = []

# new_pos = []

img = env.render(old_state)
if not os.path.exists('state_pics'):
    os.makedirs('state_pics')
Image.fromarray(img).save("state_pics/init_state.png")
# img1 = env.render_agent_view(old_state, agent=0)
# img2 = env.render_agent_view(old_state, agent=1)
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
    rng, *rngs = jax.random.split(rng, num_agents+1)
    # a1 = jnp.array(2)
    # a2 = jnp.array(4)
    actions = [jax.random.choice(
        rngs[a], a=num_actions, p=jnp.array([0.1, 0.1,0.5, 0.1, 0.2])
    ) for a in range(num_agents)]
    # actions = [jax.random.choice(
    #     rngs[a], a=num_actions, p=jnp.array([0.0, 0.0,1.0, 0.0, 0.0])
    # ) for a in range(num_agents)]
    print('###################')
    print(f'timestep: {t} to {t+1}')
    print([action.item() for action in actions], 'actions')
    print("###################")

    obs, state, reward, done, info = env.step_env(
        rng, old_state, [a*action for a in actions]
    )
    print('outer t', state.outer_t)
    print('inner t', state.inner_t)
    print('done', done)

    img = env.render(state)
    Image.fromarray(img).save(f"state_pics/state_{t+1}.png")
    pics.append(img)

    if render_agent_view:
        img1 = env.render_agent_view(state, agent=0)
        img2 = env.render_agent_view(state, agent=1)
        img3 = env.render_agent_view(state, agent=2)
        Image.fromarray(img1).save(f"state_pics/view_ag1_{t+1}.png")
        Image.fromarray(img2).save(f"state_pics/view_ag2_{t+1}.png")
        Image.fromarray(img3).save(f"state_pics/view_ag3_{t+1}.png")
        pics1.append(img1)
        pics2.append(img2)
    old_state = state
    # if True:
    # if  t==74 or t==75 or t==76:
        # jax.debug.breakpoint()
    # if t>76:
    #     raise
print("Saving GIF")
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

if render_agent_view:
    pics1 = [Image.fromarray(img) for img in pics1]
    pics2 = [Image.fromarray(img) for img in pics2]
    pics1[0].save(
        "agent1.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics1[1:],
        duration=1000,
        loop=0,
    )
    pics2[0].save(
        "agent2.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics2[1:],
        duration=1000,
        loop=0,
    )