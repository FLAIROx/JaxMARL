import jax
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import contextlib
import os
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

class UTrackingAnimator(animation.FuncAnimation):

    def __init__(
        self,
        agent_positions,
        landmark_positions,
        landmark_predictions,
        episode_rewards,
        episode_errors,
        lags=None,
        interval=100,
        legend=True,
        fps=30
    ):
        # Convert JAX arrays to NumPy for Matplotlib compatibility
        self.agent_positions = np.asarray(agent_positions)
        self.landmark_positions = np.asarray(landmark_positions)
        self.landmark_predictions = np.asarray(landmark_predictions)
        self.episode_rewards = np.asarray(episode_rewards)
        self.episode_errors = np.asarray(episode_errors)
        
        # General parameters
        self.frames = self.agent_positions.shape[1]
        self.n_agents = len(agent_positions)
        self.n_landmarks = len(landmark_positions)
        self.lags = lags if lags is not None else self.frames
        self.fps = fps

        # Create the figure and subplots
        self.fig = plt.figure(figsize=(20, 10), dpi=120)
        # Adjust subplot widths: 60% for trajectory, 40% for metrics
        self.ax_episode = self.fig.add_subplot(1, 2, 1)
        self.ax_reward = self.fig.add_subplot(2, 2, 2)
        self.ax_error = self.fig.add_subplot(2, 2, 4)

        # Set titles
        self.ax_episode.set_title("Agent and Landmark Trajectories")
        self.ax_reward.set_title("Episode Reward")
        self.ax_error.set_title("Landmark Prediction Error")

        # legend
        self.legend = legend

        # Initialize artists
        self._init_episode_animation()
        self._init_reward_animation()
        self._init_error_animation()

        # Call parent constructor
        super().__init__(self.fig, self._update, frames=self.frames, 
                         interval=interval, blit=True)

    def _init_episode_animation(self):
        # Set up color schemes
        self.agent_colors = plt.cm.tab10(np.linspace(0, 1, self.n_agents))
        self.landmark_colors = plt.cm.Set2(np.linspace(0, 1, self.n_landmarks))
        self.prediction_colors = plt.cm.Set1(np.linspace(0, 1, self.n_landmarks))

        # Calculate boundaries with padding
        all_positions = np.concatenate([
            self.agent_positions.reshape(-1, 2),
            self.landmark_positions.reshape(-1, 2),
            self.landmark_predictions.reshape(-1, 2)
        ])
        
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        
        # Calculate range and center for square aspect
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        padding = max_range * 0.1
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Set equal range for both axes to ensure square aspect
        half_range = max_range / 2 + padding
        self.ax_episode.set_xlim(center_x - half_range, center_x + half_range)
        self.ax_episode.set_ylim(center_y - half_range, center_y + half_range)
        
        # Set equal aspect ratio (makes plot square)
        self.ax_episode.set_aspect('equal')
        
        # Styling
        self.ax_episode.grid(True, ls='--', alpha=0.5)
        self.ax_episode.set_xlabel('X Position')
        self.ax_episode.set_ylabel('Y Position')
        self.ax_episode.set_facecolor('#f0f8ff')  # Light blue background

        # Create artists for agents
        self.agent_trajectories = []
        self.agent_start_markers = []
        self.agent_end_markers = []
        
        for i, color in enumerate(self.agent_colors):
            # Trajectory line
            traj, = self.ax_episode.plot(
                [], [], color=color, lw=1.8, alpha=0.9,
                label=f"Agent {i+1} Trajectory"
            )
            self.agent_trajectories.append(traj)
            
            # Start marker (static)
            start_marker = self.ax_episode.scatter(
                self.agent_positions[i, 0, 0],
                self.agent_positions[i, 0, 1],
                color=color, marker='*', s=120, 
                edgecolor='k', zorder=10,
                label=f"Agent {i+1} Start"
            )
            self.agent_start_markers.append(start_marker)
            
            # End marker (current position)
            end_marker, = self.ax_episode.plot(
                [], [], 'o', color=color, markersize=10,
                markeredgecolor='k', markeredgewidth=1,
                label=f"Agent {i+1} Current"
            )
            self.agent_end_markers.append(end_marker)

        # Create artists for landmarks
        self.landmark_true_lines = []
        self.landmark_pred_lines = []
        self.landmark_true_points = []
        self.landmark_pred_points = []
        
        for i, (true_color, pred_color) in enumerate(zip(self.landmark_colors, self.prediction_colors)):
            # True positions
            true_line, = self.ax_episode.plot(
                [], [], 'o-', color=true_color, 
                markersize=5, alpha=0.7, lw=1.5,
                label=f"Landmark {i+1} True Path"
            )
            self.landmark_true_lines.append(true_line)
            
            # Predicted positions
            pred_line, = self.ax_episode.plot(
                [], [], '--', color=pred_color, 
                markersize=0, alpha=0.9, lw=2,
                label=f"Landmark {i+1} Prediction"
            )
            self.landmark_pred_lines.append(pred_line)
            
            # Current true position
            true_point, = self.ax_episode.plot(
                [], [], 's', color=true_color, 
                markersize=8, alpha=1.0, markeredgecolor='k'
            )
            self.landmark_true_points.append(true_point)
            
            # Current prediction
            pred_point, = self.ax_episode.plot(
                [], [], 'D', color=pred_color, 
                markersize=6, alpha=1.0, markeredgecolor='k'
            )
            self.landmark_pred_points.append(pred_point)

        # Create a comprehensive legend
        if self.legend:
            legend_handles = []
            legend_labels = []
            
            # Agent elements
            for i in range(self.n_agents):
                legend_handles.append(Line2D([0], [0], color=self.agent_colors[i], lw=2))
                legend_labels.append(f"Agent {i+1} Path")
                legend_handles.append(Line2D([0], [0], marker='*', color=self.agent_colors[i], 
                                        markersize=10, linestyle=''))
                legend_labels.append(f"Agent {i+1} Start")
                legend_handles.append(Line2D([0], [0], marker='o', color=self.agent_colors[i], 
                                        markersize=8, linestyle='', markeredgecolor='k'))
                legend_labels.append(f"Agent {i+1} Current")
            
            # Landmark elements
            for i in range(self.n_landmarks):
                legend_handles.append(Line2D([0], [0], color=self.landmark_colors[i], 
                                            marker='s', markersize=8, linestyle='-'))
                legend_labels.append(f"Landmark {i+1} True Path")
                legend_handles.append(Line2D([0], [0], color=self.prediction_colors[i], 
                                            linestyle='--', lw=2))
                legend_labels.append(f"Landmark {i+1} Prediction")
            
            # Position legend outside the plot area
            self.ax_episode.legend(
                legend_handles, legend_labels, 
                # loc='upper left', 
                # bbox_to_anchor=(1.02, 1),
                # borderaxespad=0.,
                frameon=True,
                framealpha=0.8,
                ncol=1
            )
            
            # Adjust layout to accommodate legend
            # plt.tight_layout()
            # self.fig.subplots_adjust(right=0.8)

    def _init_reward_animation(self):
        self.ax_reward.grid(True, ls='--', alpha=0.5)
        self.ax_reward.set_xlim(0, self.frames)
        self.ax_reward.set_ylim(
            self.episode_rewards.min() - 0.1, 
            self.episode_rewards.max() + 0.1
        )
        self.ax_reward.set_xlabel("Timestep")
        self.ax_reward.set_ylabel("Reward")
        
        # Reward line
        self.reward_line, = self.ax_reward.plot([], [], color='#2ca02c', lw=2)

    def _init_error_animation(self):
        self.ax_error.grid(True, ls='--', alpha=0.5)
        self.ax_error.set_xlim(0, self.frames)
        self.ax_error.set_ylim(
            self.episode_errors.min() - 0.1, 
            self.episode_errors.max() + 0.1
        )
        self.ax_error.set_xlabel("Timestep")
        self.ax_error.set_ylabel("Prediction Error")
        
        # Error lines
        self.error_lines = []
        for i, color in enumerate(self.prediction_colors):
            line, = self.ax_error.plot(
                [], [], color=color, lw=1.5,
                label=f"Landmark {i+1}"
            )
            self.error_lines.append(line)
        self.ax_error.legend()

    def _update(self, frame):
        # Update agents
        for i in range(self.n_agents):
            # Update trajectory
            x = self.agent_positions[i, :frame+1, 0]
            y = self.agent_positions[i, :frame+1, 1]
            self.agent_trajectories[i].set_data(x, y)
            
            # Update end marker
            if frame < self.frames:
                self.agent_end_markers[i].set_data(
                    [self.agent_positions[i, frame, 0]],
                    [self.agent_positions[i, frame, 1]]
                )

        # Update landmarks
        for i in range(self.n_landmarks):
            # True positions
            true_x = self.landmark_positions[i, :frame+1, 0]
            true_y = self.landmark_positions[i, :frame+1, 1]
            self.landmark_true_lines[i].set_data(true_x, true_y)
            
            # Current true position
            if frame < self.frames:
                self.landmark_true_points[i].set_data(
                    [self.landmark_positions[i, frame, 0]],
                    [self.landmark_positions[i, frame, 1]]
                )
            
            # Predicted positions
            pred_x = self.landmark_predictions[i, :frame+1, 0]
            pred_y = self.landmark_predictions[i, :frame+1, 1]
            self.landmark_pred_lines[i].set_data(pred_x, pred_y)
            
            # Current prediction
            if frame < self.frames:
                self.landmark_pred_points[i].set_data(
                    [self.landmark_predictions[i, frame, 0]],
                    [self.landmark_predictions[i, frame, 1]]
                )

        # Update reward plot
        self.reward_line.set_data(np.arange(frame+1), self.episode_rewards[:frame+1])

        # Update error plot
        for i in range(self.n_landmarks):
            self.error_lines[i].set_data(
                np.arange(frame+1), 
                self.episode_errors[i, :frame+1]
            )

        # Return all artists that need redrawing
        return (
            self.agent_trajectories + 
            self.agent_end_markers +
            self.landmark_true_lines +
            self.landmark_true_points +
            self.landmark_pred_lines +
            self.landmark_pred_points +
            [self.reward_line] +
            self.error_lines
        )

    def save_animation(self, savepath="episode", save_gif=True):
        
        if save_gif:
            full_path = savepath + ".gif"
            writer = 'pillow'  # Use pillow for GIF
            self.save(full_path, writer=writer, fps=self.fps)
            print(f"Animation saved to {full_path}")
        else:
            full_path = savepath + ".mp4"
            writer = 'ffmpeg'
            self.save(full_path, writer=writer, fps=self.fps)
            print(f"Animation saved to {full_path}")
        
        # Also save static image
        self.save_fig(savepath + ".png")

    def save_fig(self, savepath="episode.png"):
        # Update to last frame
        self._update(self.frames - 1)
        self.fig.savefig(savepath, bbox_inches='tight', dpi=150)
        plt.close(self.fig)
        print(f"Static image saved to {savepath}")


def animate_from_infos(infos, num_agents=3, save_path="./outputs", save_gif=True, fps=10):
    """
    Animate an episode from the infos dictionary. Remember to set 'infos_for_render' to True in the environment.
    """
    assert "render" in infos, "Set infos_for_render=True in environment"

    # Preprocess data
    x = infos['render']
    dones = x["done"]
    first_done = np.argmax(dones) if np.any(dones) else len(dones)
    
    # Create mask for first complete episode
    first_episode_mask = np.arange(len(dones)) <= first_done
    
    # Filter data for first episode
    filtered = {}
    for key, val in x.items():
        if isinstance(val, (jnp.ndarray, np.ndarray)):
            filtered[key] = np.asarray(val)[first_episode_mask]
    
    viz = UTrackingAnimator(
        agent_positions=np.swapaxes(filtered["pos"], 0, 1)[:num_agents, :, :2],
        landmark_positions=np.swapaxes(filtered["pos"], 0, 1)[num_agents:, :, :2],
        landmark_predictions=np.swapaxes(filtered["tracking_pred"], 0, 1),
        episode_rewards=filtered["reward"],
        episode_errors=np.swapaxes(filtered["tracking_error"], 0, 1),
        fps=10,
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    viz.save_animation(save_path, save_gif=save_gif)
