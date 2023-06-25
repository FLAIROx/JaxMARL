""" Built off gymnax vizualizer.py"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional

from smax.environments.multi_agent_env import MultiAgentEnv, EnvParams


class Visualizer(object):
    def __init__(
        self,
        env: MultiAgentEnv,
        state_seq,
        env_params: Optional[EnvParams] = None,
        reward_seq=None,
    ):
        self.env = env
        self.env_params = self.env.default_params if env_params is None else env_params

        self.interval = 50
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 5))

    def animate(
        self,
        save_fname: Optional[str] = None,
        view: bool = True,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            init_func=self.init,
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname)
        # Simply view it 3 times
        if view:
            plt.show(block=False)
            plt.pause(30)
            plt.close()

    def init(self):
        self.im = self.env.init_render(self.ax, self.state_seq[0], self.env_params)

    def update(self, frame):
        self.im = self.env.update_render(
            self.im, self.state_seq[frame], self.env_params
        )


class MiniSMACVisualizer(Visualizer):
    def expand_state_seq(self):
        """Because the minismac environment ticks faster than the states received
        we need to expand the states to visualise them"""
        expanded_state_seq = []
        for state, actions in self.state_seq:
            for _ in range(self.env.world_steps_per_env_step):
                expanded_state_seq.append((state, actions))
                world_actions = jnp.array([actions[i] for i in self.env.agents])
                state = self.env._world_step(state, world_actions, self.env_params)
                state = self.env._update_dead_agents(state)
        self.state_seq = expanded_state_seq

    def animate(self, save_fname: Optional[str] = None, view: bool = True):
        self.expand_state_seq()
        return super().animate(save_fname, view)

    def init(self):
        self.im = self.env.init_render(self.ax, self.state_seq[0], 0, self.env_params)

    def update(self, frame):
        self.im = self.env.update_render(
            self.im,
            self.state_seq[frame],
            frame % self.env.world_steps_per_env_step,
            self.env_params,
        )
