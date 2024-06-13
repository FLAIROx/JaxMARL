""" Built off gymnax vizualizer.py"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, List

from .jaxnav_env import JaxNav
import jax.numpy as jnp

class JaxNavVisualizer(object):
    def __init__(self, 
                 env: JaxNav, 
                 obs_seq: List, 
                 state_seq: List,
                 reward_seq: List=None,
                 done_frames=None,
                 title_text: str=None,
                 plot_lidar=True,
                 plot_path=True,
                 plot_agent=True,
                 plot_reward=True,
                 plot_line_to_goal=True,):
        self.env = env

        self.interval = 15
        self.obs_seq = obs_seq
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.done_frames = done_frames
        self.reward = 0.0
        self.plot_lidar = plot_lidar
        self.plot_agent = plot_agent
        self.plot_path = plot_path
        self.plot_line_to_goal = plot_line_to_goal
        self.title_text = title_text
        if (plot_reward) and (reward_seq is not None):
            self.plot_reward=True
        else:
            self.plot_reward=False
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        
        if self.plot_path:
            self.path_seq = jnp.empty((len(self.state_seq), env.num_agents, 2))
            for i in range(len(self.state_seq)):
                self.path_seq = self.path_seq.at[i].set(self.state_seq[i].pos)
            

    def animate(
        self,
        save_fname: Optional[str] = None,
        view: bool = False,
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
            plt.pause(3)
            plt.close()

    def init(self):
        self.env.init_render(self.ax, self.state_seq[0], self.obs_seq[0], lidar=self.plot_lidar, agent=self.plot_agent, goal=False)

    def update(self, frame):
        self.ax.cla()
        if self.plot_path:
            for a in range(self.env.num_agents):
                plot_frame = frame
                if self.done_frames[a] < frame:
                    plot_frame = self.done_frames[a]
                self.env.map_obj.plot_agent_path(self.ax, self.path_seq[:plot_frame, a, 0], self.path_seq[:plot_frame, a, 1])
            # self.ax.plot(self.path_seq[:frame, 0], self.path_seq[:frame, 1], color='b', linewidth=2.0, zorder=1)
        self.env.init_render(self.ax, self.state_seq[frame], self.obs_seq[frame], lidar=self.plot_lidar, agent=self.plot_agent)
        txt_to_plot = []
        txt_to_plot.append(f"Time: {frame*self.env.dt:.2f} s")
            # self.ax.text(0.05, 0.95, f"Time: {frame*self.env.dt:.2f} s", transform=self.ax.transAxes, fontsize=12, verticalalignment='top', c='w')
        if self.plot_reward: 
            self.reward += self.reward_seq[frame]
            txt_to_plot.append(f"R: {self.reward:.2f}")
        if self.title_text is not None:
            title_text = self.title_text + ' ' + ' '.join(txt_to_plot)
        else:
            title_text = ' '.join(txt_to_plot)
        self.ax.set_title(title_text)
            # self.ax.text(0.05, 0.9, f"R: {self.reward:.2f}", transform=self.ax.transAxes, fontsize=12, verticalalignment='top', c='w')
        # if len(txt_to_plot) > 0:
        #     self.ax.text(0.05, 0.95, ' '.join(txt_to_plot), transform=self.ax.transAxes, fontsize=12, verticalalignment='top', c='w')
            
        # if self.plot_line_to_goal:
        #     for a in range(self.env.num_agents):
        #         plot_frame = frame
        #         if self.done_frames[a] < frame:
        #             plot_frame = self.done_frames[a] 
        #         x = jnp.concatenate([self.state_seq[plot_frame].pos[a, 0][None], self.state_seq[plot_frame].goal[a, 0][None]])
        #         y = jnp.concatenate([self.state_seq[plot_frame].pos[a, 1][None], self.state_seq[plot_frame].goal[a, 1][None]])                self.ax.plot(, 
        #                      jnp.concatenate([self.state_seq[plot_frame].pos[a, 1][None], self.state_seq[plot_frame].goal[a, 1][None]]), 
        #                      color='gray', alpha=0.4)
