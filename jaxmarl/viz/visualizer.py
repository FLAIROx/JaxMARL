""" Built off gymnax vizualizer.py"""
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.smax.heuristic_enemy_smax_env import (
    EnemySMAX,
)


class Visualizer(object):
    def __init__(
        self,
        env: MultiAgentEnv,
        state_seq,
        reward_seq=None,
    ):
        self.env = env

        self.interval = 64
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
            plt.show(block=True)
            # plt.pause(30)
            # plt.close()

    def init(self):
        self.im = self.env.init_render(self.ax, self.state_seq[0])

    def update(self, frame):
        self.im = self.env.update_render(
            self.im, self.state_seq[frame]
        )


class SMAXVisualizer(Visualizer):
    """Visualiser especially for the SMAX environments. Needed because they have an internal model that ticks much faster
    than the learner's 'step' calls. This  means that we need to expand the state_sequence
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        state_seq,
        reward_seq=None,
    ):
        super().__init__(env, state_seq, reward_seq)
        self.heuristic_enemy = isinstance(env, EnemySMAX)
        self.have_expanded = False

    def expand_state_seq(self):
        """Because the smax environment ticks faster than the states received
        we need to expand the states to visualise them"""
        self.state_seq = self.env.expand_state_seq(self.state_seq)
        self.have_expanded = True

    def animate(self, save_fname: Optional[str] = None, view: bool = True):
        if not self.have_expanded:
            self.expand_state_seq()
        return super().animate(save_fname, view)

    def init(self):
        self.im = self.env.init_render(
            self.ax, self.state_seq[0], 0, 0
        )

    def update(self, frame):
        self.im = self.env.update_render(
            self.im,
            self.state_seq[frame],
            frame % self.env.world_steps_per_env_step,
            frame // self.env.world_steps_per_env_step,
        )
