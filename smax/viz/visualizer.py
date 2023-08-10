""" Built off gymnax vizualizer.py"""
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional

from smax.environments.multi_agent_env import MultiAgentEnv
from smax.environments.mini_smac.heuristic_enemy import create_heuristic_policy
from smax.environments.mini_smac.heuristic_enemy_mini_smac_env import (
    HeuristicEnemyMiniSMAC,
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


class MiniSMACVisualizer(Visualizer):
    """Visualiser especially for the MiniSMAC environments. Needed because they have an internal model that ticks much faster
    than the learner's 'step' calls. This  means that we need to expand the state_sequence
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        state_seq,
        reward_seq=None,
    ):
        super().__init__(env, state_seq, reward_seq)
        self.heuristic_enemy = isinstance(env, HeuristicEnemyMiniSMAC)
        self.have_expanded = False
        if self.heuristic_enemy:
            self.heuristic_policy = create_heuristic_policy(env, 1, shoot=env.enemy_shoots)

    def expand_state_seq(self):
        """Because the minismac environment ticks faster than the states received
        we need to expand the states to visualise them"""
        expanded_state_seq = []
        for i, (key, state, actions) in enumerate(self.state_seq):
            if self.heuristic_enemy:
                agents = self.env.all_agents
                key, key_action = jax.random.split(key)
                key_action = jax.random.split(
                    key_action, num=self.env.num_agents_per_team
                )
                obs = self.env.get_all_unit_obs(state)
                obs = jnp.array([obs[agent] for agent in self.env.enemy_agents])
                enemy_actions = jax.vmap(self.heuristic_policy)(key_action, obs)
                enemy_actions = {
                    agent: enemy_actions[
                        self.env.agent_ids[agent] % self.env.num_agents_per_team
                    ]
                    for agent in self.env.enemy_agents
                }
                actions = {k: v.squeeze() for k, v in actions.items()}
                actions = {**enemy_actions, **actions}
            else:
                agents = self.env.agents
            for _ in range(self.env.world_steps_per_env_step):
                expanded_state_seq.append((key, state, actions))
                world_actions = jnp.array([actions[i] for i in agents])
                state = self.env._world_step(state, world_actions)
                state = self.env._update_dead_agents(state)
                state = self.env._push_units_away(state)
            state = state.replace(terminal=self.env.is_terminal(state))
        self.state_seq = expanded_state_seq
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
