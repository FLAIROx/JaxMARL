"""Temporary visualizer helper that works only with parameters sharing' qlearning agents"""

from functools import partial
import jax
from jax import numpy as jnp
import numpy as np

import numpy as np
import logging
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import contextlib

class UTrackingQLearningViz:

    def __init__(self, env, agent, agent_params, hidden_dim=64, max_steps=200):
        self.env = env
        self.agent = agent
        self.agent_params = agent_params
        self.hidden_dim = hidden_dim
        self.max_steps=max_steps

    @partial(jax.jit, static_argnums=0)
    def get_rollout(self, rng):
        key, key_r, key_a = jax.random.split(rng, 3)
        
        init_x = (
            jnp.zeros((1, 1, self.env.obs_size)), # (time_step, batch_size, obs_size)
            jnp.zeros((1, 1)) # (time_step, batch size)
        )
        init_hstate = jnp.zeros((1, self.hidden_dim)) 
        _ = self.agent.init(key_a, init_hstate, init_x)
        
        init_dones = {agent:jnp.zeros(1, dtype=bool) for agent in self.env.agents+['__all__']}
        
        hstate = jnp.zeros((1*self.env.env.num_agents, self.hidden_dim))
        init_obs, env_state = self.env.batch_reset(key_r)

        def homogeneous_pass(params, hidden_state, obs, dones):
            # concatenate agents and parallel envs to process them in one batch
            agents, flatten_agents_obs = zip(*obs.items())
            original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
            batched_input = (
                jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
                jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
            )
            hidden_state, q_vals = self.agent.apply(params, hidden_state, batched_input)
            q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1) # (time_steps, n_agents, n_envs, action_dim)
            q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
            return hidden_state, q_vals
    
        def _env_step(step_state, unused):
            params, env_state, last_obs, last_dones, hstate, rng = step_state
    
            rng, key_a, key_s = jax.random.split(rng, 3)
            
            obs_ = {a:last_obs[a] for a in self.env.agents}
            obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
            dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
    
            hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
            valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, self.env.valid_actions)
            # greedy actions
            actions = jax.tree_util.tree_map(lambda q: jnp.argmax(q, axis=-1), valid_q_vals)
    
            # step
            obs, env_state, rewards, dones, info = self.env.batch_step(key_s, env_state, actions)
            info['pos']  = env_state.pos
            info['done'] = dones['__all__']
    
            step_state = (params, env_state, obs, dones, hstate, rng)
            return step_state, info
    
            
        step_state = (
            self.agent_params,
            env_state,
            init_obs,
            init_dones,
            hstate, 
            key,
        )
    
        step_state, infos = jax.lax.scan(
            _env_step, step_state, None, self.max_steps
        )
    
        return infos

    def get_animation(self, rng, save_path='./tmp_animation'):
        
        infos = self.get_rollout(rng)

        #preprocess
        x = jax.tree_map(lambda x: x[:,0], infos)
        dones = x['done']
        first_done = jax.lax.select((jnp.argmax(dones)==0)&(dones[0]!=True), dones.size, jnp.argmax(dones))
        first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
        x = jax.tree_map(lambda x: x[first_episode_mask], x)

        viz = UTrackingAnimator(
            agent_positions = jnp.swapaxes(x['pos'], 0, 1)[:self.env.env.num_agents, :, :2],
            landmark_positions = jnp.swapaxes(x['pos'], 0, 1)[self.env.env.num_agents:, :, :2],
            landmark_predictions = jnp.swapaxes(x['tracking_pred'], 0, 1),
            episode_rewards = x['rew'],
            episode_errors = jnp.swapaxes(x['tracking_error'], 0, 1),
        )
        viz.save_animation(save_path)


class UTrackingAnimator(animation.TimedAnimation):
    
    def __init__(self,
                 agent_positions,
                 landmark_positions,
                 landmark_predictions,
                 episode_rewards,
                 episode_errors,
                 lags=None):
        
        
        # general parameters
        self.frames = (agent_positions.shape[1])
        self.n_agents = len(agent_positions)
        self.n_landmarks = len(landmark_positions)
        
        self.agent_positions = agent_positions
        self.landmark_positions = landmark_positions
        self.landmark_predictions = landmark_predictions
        self.episode_rewards = episode_rewards
        self.episode_errors = episode_errors
        if lags is None:
            self.lags = self.frames
        else:
            self.lags = self.lags
        
        # create the subplots
        self.fig = plt.figure(figsize=(20, 10), dpi=120)
        self.ax_episode = self.fig.add_subplot(1, 2, 1)
        self.ax_reward = self.fig.add_subplot(2, 2, 2)
        self.ax_error = self.fig.add_subplot(2, 2, 4)
        
        self.ax_episode.set_title('Episode')
        self.ax_reward.set_title('Reward')
        self.ax_error.set_title('Prediction Error')
        
        # colors
        self.agent_colors = cm.Dark2.colors
        self.landmark_colors = [cm.summer(l*10) for l in range(self.n_landmarks)] # pastl greens
        self.prediction_colors = [cm.PiYG(l*10) for l in range(self.n_landmarks)] # pinks
        
        # init the lines
        self.lines_episode = self._init_episode_animation(self.ax_episode)
        self.lines_reward  = self._init_reward_animation(self.ax_reward)
        self.lines_error   = self._init_error_animation(self.ax_error)
        

        animation.TimedAnimation.__init__(self, self.fig, interval=100, blit=True)

    def save_animation(self, savepath='episode'):
        with contextlib.redirect_stdout(None):
            self.save(savepath+'.gif')
            self.fig.savefig(savepath+'.png')
        
        
    def _episode_update(self, data, line, frame, lags, name=None):
        line.set_data(data[max(0,frame-lags):frame, 0], data[max(0,frame-lags):frame, 1])
        if name is not None:
            line.set_label(name)
        
    def _frameline_update(self, data, line, frame, name=None):
        line.set_data(np.arange(1,frame+1), data[:frame])
        if name is not None:
            line.set_label(name)

    def _draw_frame(self, frame):
            
        # Update the episode subplot
        line_episode = 0
        # update agents heads
        for n in range(self.n_agents):
            self._episode_update(self.agent_positions[n], self.lines_episode[line_episode], frame, 1, f'Agent_{n+1}')
            line_episode += 1
            
        # update agents trajectories
        for n in range(self.n_agents):
            self._episode_update(self.agent_positions[n], self.lines_episode[line_episode], max(0,frame-1), self.lags)
            line_episode += 1

        # landmark real positions
        for n in range(self.n_landmarks):
            self._episode_update(self.landmark_positions[n], self.lines_episode[line_episode], frame, self.lags, f'Landmark_{n+1}_real')
            line_episode += 1

        # landmark predictions
        for n in range(self.n_landmarks):
            self._episode_update(self.landmark_predictions[n], self.lines_episode[line_episode], frame, self.lags, f'Landmark_{n+1}_predictions')
            line_episode += 1

        self.ax_episode.legend()
        
        # Update the reward subplot
        self._frameline_update(self.episode_rewards, self.lines_reward[0], frame)
        
        # Update the error subplot
        for n in range(self.n_landmarks):
            self._frameline_update(self.episode_errors[n], self.lines_error[n], frame, f'Landmark_{n+1}_error')
        self.ax_error.legend()

        self._drawn_artists = self.lines_episode + self.lines_reward + self.lines_error
        
        
    def _init_episode_animation(self, ax):
        # retrieve the episode dimensions
        x_max = max(self.agent_positions[:,:,0].max(),
                    self.landmark_positions[:,:,0].max())

        x_min = min(self.agent_positions[:,:,0].min(),
                    self.landmark_positions[:,:,0].min())

        y_max = max(self.agent_positions[:,:,1].max(),
                    self.landmark_positions[:,:,1].max())

        y_min = min(self.agent_positions[:,:,1].min(),
                    self.landmark_positions[:,:,1].min())

        abs_min = min(x_min, y_min)
        abs_max = max(x_max, y_max)

        ax.set_xlim(abs_min-1, abs_max+1)
        ax.set_ylim(abs_min-1,abs_max+1)
        ax.set_ylabel('Y Position')
        
        # remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # lines:
        # 1. agent head
        # 2. agent trajectory
        # 3. landmark real
        # 4. landmark prediction
        lines = [ax.plot([],[],'o',color=self.agent_colors[a], alpha=0.8,markersize=8)[0] for a in range(self.n_agents)] + \
                [ax.plot([],[],'o',color=self.agent_colors[a], alpha=0.2,markersize=4)[0] for a in range(self.n_agents)] + \
                [ax.plot([],[],'s',color=self.landmark_colors[l], alpha=0.8,markersize=8)[0] for l in range(self.n_landmarks)] + \
                [ax.plot([],[],'s',color=self.prediction_colors[l], alpha=0.2,markersize=4)[0] for l in range(self.n_landmarks)]
                    
        return lines
    
    def _init_reward_animation(self, ax):
        ax.set_xlim(0, self.frames)
        ax.set_ylim(self.episode_rewards.min(), self.episode_rewards.max()+1)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Reward')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)) # force integer ticks
        
        # remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        lines = [ax.plot([],[], color='green')[0]]
        return lines
        
    def _init_error_animation(self, ax):
        ax.set_xlim(0, self.frames)
        ax.set_ylim(self.episode_errors.min(), self.episode_errors.max())
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Prediction error')
        
        # remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        lines = [ax.plot([],[], color=self.prediction_colors[l])[0] for l in range(self.n_landmarks)]
        return lines
      

    def new_frame_seq(self):
        return iter(range(self.frames))

    def _init_draw(self):
        lines = self.lines_episode + self.lines_reward + self.lines_error
        for l in lines:
            l.set_data([], [])