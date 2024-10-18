import jax
from jax import numpy as jnp
import numpy as np

# for plot
import logging

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import contextlib


class UTrackingAnimator(animation.TimedAnimation):

    def __init__(
        self,
        agent_positions,
        landmark_positions,
        landmark_predictions,
        episode_rewards,
        episode_errors,
        lags=None,
    ):
        """
        agent_positions: np.array (n_agents, frames, 2)
        landmark_positions: np.array (n_landmarks, frames, 2)
        landmark_predictions: np.array (n_landmarks, frames, 2)
        episode_rewards: np.array (frames,)
        episode_errors: np.array (n_landmarks, frames)
        lags: int, number of frames to show in the past, default is the number of frames
        """

        # general parameters
        self.frames = agent_positions.shape[1]
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

        self.ax_episode.set_title("Episode")
        self.ax_reward.set_title("Reward")
        self.ax_error.set_title("Prediction Error")

        # colors
        self.agent_colors = cm.Dark2.colors
        self.landmark_colors = [
            cm.summer(l * 10) for l in range(self.n_landmarks)
        ]  # pastl greens
        self.prediction_colors = [
            cm.PiYG(l * 10) for l in range(self.n_landmarks)
        ]  # pinks

        # init the lines
        self.lines_episode = self._init_episode_animation(self.ax_episode)
        self.lines_reward = self._init_reward_animation(self.ax_reward)
        self.lines_error = self._init_error_animation(self.ax_error)

        animation.TimedAnimation.__init__(self, self.fig, interval=100, blit=True)

    def save_animation(self, savepath="episode"):
        with contextlib.redirect_stdout(None):
            self.save(savepath + ".gif")
            self.fig.savefig(savepath + ".png")

    def _episode_update(self, data, line, frame, lags, name=None):
        line.set_data(
            data[max(0, frame - lags) : frame, 0], data[max(0, frame - lags) : frame, 1]
        )
        if name is not None:
            line.set_label(name)

    def _frameline_update(self, data, line, frame, name=None):
        line.set_data(np.arange(1, frame + 1), data[:frame])
        if name is not None:
            line.set_label(name)

    def _draw_frame(self, frame):

        # Update the episode subplot
        line_episode = 0
        # update agents heads
        for n in range(self.n_agents):
            self._episode_update(
                self.agent_positions[n],
                self.lines_episode[line_episode],
                frame,
                1,
                f"Agent_{n+1}",
            )
            line_episode += 1

        # update agents trajectories
        for n in range(self.n_agents):
            self._episode_update(
                self.agent_positions[n],
                self.lines_episode[line_episode],
                max(0, frame - 1),
                self.lags,
            )
            line_episode += 1

        # landmark real positions
        for n in range(self.n_landmarks):
            self._episode_update(
                self.landmark_positions[n],
                self.lines_episode[line_episode],
                frame,
                self.lags,
                f"Landmark_{n+1}_real",
            )
            line_episode += 1

        # landmark predictions
        for n in range(self.n_landmarks):
            self._episode_update(
                self.landmark_predictions[n],
                self.lines_episode[line_episode],
                frame,
                self.lags,
                f"Landmark_{n+1}_predictions",
            )
            line_episode += 1

        self.ax_episode.legend()

        # Update the reward subplot
        self._frameline_update(self.episode_rewards, self.lines_reward[0], frame)

        # Update the error subplot
        for n in range(self.n_landmarks):
            self._frameline_update(
                self.episode_errors[n],
                self.lines_error[n],
                frame,
                f"Landmark_{n+1}_error",
            )
        self.ax_error.legend()

        self._drawn_artists = self.lines_episode + self.lines_reward + self.lines_error

    def _init_episode_animation(self, ax):
        # retrieve the episode dimensions
        x_max = max(
            self.agent_positions[:, :, 0].max(), self.landmark_positions[:, :, 0].max()
        )

        x_min = min(
            self.agent_positions[:, :, 0].min(), self.landmark_positions[:, :, 0].min()
        )

        y_max = max(
            self.agent_positions[:, :, 1].max(), self.landmark_positions[:, :, 1].max()
        )

        y_min = min(
            self.agent_positions[:, :, 1].min(), self.landmark_positions[:, :, 1].min()
        )

        abs_min = min(x_min, y_min)
        abs_max = max(x_max, y_max)

        ax.set_xlim(abs_min - 1, abs_max + 1)
        ax.set_ylim(abs_min - 1, abs_max + 1)
        ax.set_ylabel("Y Position")

        # remove frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # lines:
        # 1. agent head
        # 2. agent trajectory
        # 3. landmark real
        # 4. landmark prediction
        lines = (
            [
                ax.plot(
                    [], [], "o", color=self.agent_colors[a], alpha=0.8, markersize=8
                )[0]
                for a in range(self.n_agents)
            ]
            + [
                ax.plot(
                    [], [], "o", color=self.agent_colors[a], alpha=0.2, markersize=4
                )[0]
                for a in range(self.n_agents)
            ]
            + [
                ax.plot(
                    [], [], "s", color=self.landmark_colors[l], alpha=0.8, markersize=8
                )[0]
                for l in range(self.n_landmarks)
            ]
            + [
                ax.plot(
                    [],
                    [],
                    "s",
                    color=self.prediction_colors[l],
                    alpha=0.2,
                    markersize=4,
                )[0]
                for l in range(self.n_landmarks)
            ]
        )

        return lines

    def _init_reward_animation(self, ax):
        ax.set_xlim(0, self.frames)
        ax.set_ylim(self.episode_rewards.min(), self.episode_rewards.max() + 1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # force integer ticks

        # remove frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        lines = [ax.plot([], [], color="green")[0]]
        return lines

    def _init_error_animation(self, ax):
        ax.set_xlim(0, self.frames)
        ax.set_ylim(self.episode_errors.min(), self.episode_errors.max())
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Prediction error")

        # remove frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        lines = [
            ax.plot([], [], color=self.prediction_colors[l])[0]
            for l in range(self.n_landmarks)
        ]
        return lines

    def new_frame_seq(self):
        return iter(range(self.frames))

    def _init_draw(self):
        lines = self.lines_episode + self.lines_reward + self.lines_error
        for l in lines:
            l.set_data([], [])


def animate_from_infos(infos, num_agents=3, save_path="./outputs"):
    """
    Animate an episode from the infos dictionary. Remember to set 'infos_for_render' to True in the environment.
    """

    assert (
        "render" in infos
    ), "Missing information in infos. Make sure to set infos_for_render=True in the environment."

    # preprocess
    x = infos['render']
    dones = x["done"]
    first_done = jax.lax.select(
        (jnp.argmax(dones) == 0) & (dones[0] != True), dones.size, jnp.argmax(dones)
    )
    first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
    x = jax.tree_map(lambda x: x[first_episode_mask], x)

    viz = UTrackingAnimator(
        agent_positions=jnp.swapaxes(x["pos"], 0, 1)[:num_agents, :, :2], # (num_agents, frames, 2)
        landmark_positions=jnp.swapaxes(x["pos"], 0, 1)[num_agents:, :, :2], # (num_landmarks, frames, 2)
        landmark_predictions=jnp.swapaxes(x["tracking_pred"], 0, 1), # (num_landmarks, frames, 2)
        episode_rewards=x["reward"], # (frames,)
        episode_errors=jnp.swapaxes(x["tracking_error"], 0, 1), # (num_landmarks, frames)
    )
    viz.save_animation(save_path)
