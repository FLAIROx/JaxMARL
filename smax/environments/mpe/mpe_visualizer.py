import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional


class MPEVisualizer(object):
    def __init__(
        self,
        env,
        state_seq: list,
        reward_seq=None,
    ):
        self.env = env

        self.interval = 64
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))

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

        if view:
            plt.show(block=True)
            # plt.pause(30)
            # plt.close()

    def init(self):
        self.im = self.env.init_render(self.ax, self.state_seq[0])

    def update(self, frame):
        self.im = self.env.update_render(self.im, self.state_seq[frame])
