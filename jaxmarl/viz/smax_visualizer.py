import einops as ein
import math
import jax
from matplotlib.artist import Artist
from matplotlib import text as mtext
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.backend_bases import RendererBase
from typing import Sequence, cast

from jaxmarl.environments.smax.smax_env import State as SMAXState
from jaxmarl.environments.smax import SMAX
from matplotlib.patches import Circle, Rectangle


class SMAXVisualizer:
    """Visualiser especially for the SMAX environments. Needed because they have an internal model that ticks much faster
    than the learner's 'step' calls. This  means that we need to expand the state_sequence
    """

    def __init__(self, env: SMAX, state_seq: list):
        self.env = env
        self.state_seq = state_seq

        height = 6
        aspect_ratio = self.env.map_height / self.env.map_width
        figsize = (aspect_ratio * height, height / aspect_ratio)

        self.fig = Figure(figsize=figsize)
        self.ax = self.fig.subplots()

        # render circles

        self.ax.set_xlim((0.0, self.env.map_width))
        self.ax.set_ylim((0.0, self.env.map_height))
        self.ax.set_aspect("equal")

        self.title = self.ax.set_title("")

        self.ally_units: list[DrawnUnit] = []
        for i in range(env.num_allies):
            color = "blue"
            c = DrawnUnit((0.0, 0.0), 0.0, "?", color=color)
            self.ax.add_patch(c)
            self.ax.add_artist(c.label)
            self.ally_units.append(c)

        self.enemy_units: list[DrawnUnit] = []
        for i in range(env.num_enemies):
            color = "green"
            c = DrawnUnit((0.0, 0.0), 0.0, "?", color=color)
            self.ax.add_patch(c)
            self.ax.add_artist(c.label)
            self.enemy_units.append(c)

        # render bullets
        self.bullet_rects: list[Rectangle] = []
        for _ in env.agents:
            r = Rectangle((0.0, 0.0), 0.5, 0.5, color="gray")
            self.ax.add_patch(r)
            self.bullet_rects.append(r)

        self.agent_being_shot = jax.jit(self.env.agent_being_shot)
        self.agent_can_shoot = jax.jit(self.env.agent_can_shoot)

    def get_artists(self) -> Sequence[Artist]:
        return [
            *self.bullet_rects,
            *self.ally_units,
            *self.enemy_units,
            self.title,
        ]

    def update_artists(
        self, state: SMAXState, jact: dict[str, int], step_idx: int
    ) -> Sequence[Artist]:
        outer_step_i = step_idx // self.env.world_steps_per_env_step
        self.title.set_text(f"Step: {outer_step_i}")

        attacked_agents = set(
            int(self.agent_being_shot(i, jact[agent]))
            for i, agent in enumerate(self.env.agents)
            if jact[agent] > self.env.num_movement_actions - 1
            and self.agent_can_shoot(state, i, jact[agent])
        )

        # render circles
        for i in range(self.env.num_allies):
            du = self.ally_units[i]
            if state.unit_alive[i]:  # type: ignore
                color = "blue" if i not in attacked_agents else "cornflowerblue"
                du.set_data(
                    tuple(state.unit_positions[i]),  # type: ignore
                    color=color,
                    rad=self.env.unit_type_radiuses[state.unit_types[i]].item(),  # type: ignore
                    text=self.env.unit_type_shorthands[state.unit_types[i]],  # type: ignore
                )
                du.set_visible(True)
            else:
                du.set_visible(False)

        for i in range(self.env.num_enemies):
            du = self.enemy_units[i]
            idx = i + self.env.num_allies
            if state.unit_alive[idx]:  # type: ignore
                color = "green" if idx not in attacked_agents else "limegreen"
                du.set_data(
                    tuple(state.unit_positions[idx]),  # type: ignore
                    color=color,
                    rad=self.env.unit_type_radiuses[state.unit_types[idx]].item(),  # type: ignore
                    text=self.env.unit_type_shorthands[state.unit_types[idx]],  # type: ignore
                )
                du.set_visible(True)
            else:
                du.set_visible(False)

        # render bullets
        for agent in self.env.agents:
            i = self.env.agent_ids[agent]
            br = self.bullet_rects[i]
            attacked_idx = self.agent_being_shot(i, jact[agent])
            if jact[agent] < self.env.num_movement_actions or not self.agent_can_shoot(
                state, i, jact[agent]
            ):
                br.set_visible(False)
            else:
                substep = step_idx % self.env.world_steps_per_env_step
                frac = substep / (self.env.world_steps_per_env_step - 1)
                shooter_pos: jax.Array = state.unit_positions[i]  # type: ignore
                target_pos: jax.Array = state.unit_positions[attacked_idx]  # type: ignore
                bullet_pos = (1 - frac) * shooter_pos + frac * target_pos
                br.set_xy(tuple(bullet_pos))
                br.set_visible(True)

        return self.get_artists()

    def animate(
        self,
        trace: dict[str, SMAXState | dict[str, int]],
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""

        n_substeps = self.env.world_steps_per_env_step

        # Flatten the substep and env step dimension
        states: SMAXState = jax.tree.map(
            lambda x: ein.rearrange(x, "i o ... -> (i o) ..."), trace["substates"]
        )
        # And for the actions account for action repetition
        breakpoint()
        acts: dict[str, int] = jax.tree.map(
            lambda x: ein.repeat(x, "o ... -> (i o) ...", i=n_substeps),
            trace["joint_action"],
        )

        n_frames, *_ = cast("jax.Array", states.terminal).shape
        frames: list[tuple[SMAXState, dict[str, int], int]] = []

        for i in range(n_frames):
            state, act = jax.tree.map(lambda x: x[i], (states, acts))
            frames.append((state, act, i))

        init_state, init_jact, _ = frames[0]

        return animation.FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            init_func=lambda: self.update_artists(init_state, init_jact, 0),
            blit=False,
            interval=50,
        )

    def update(self, frame: tuple[SMAXState, dict[str, int], int]) -> Sequence[Artist]:
        state, jact, frame_idx = frame
        return self.update_artists(state, jact, frame_idx)

class DrawnUnit(Circle):
    def __init__(self, xy: tuple[float, float], rad: float, text: str, color: str):
        super().__init__(xy, rad, color=color)
        x, y = xy
        tx, ty = x - (1.0 / math.sqrt(2)) * rad, y - (1.0 / math.sqrt(2)) * rad
        self.label = mtext.Text(tx, ty, text, fontsize="xx-small", color="white")

    def set_data(self, xy: tuple[float, float], color: str, rad: float, text: str):
        self.set_center(xy)
        self.set_radius(rad)
        self.set_color(color)
        x, y = xy
        tx, ty = (
            x - (1.0 / math.sqrt(2)) * self.radius,
            y - (1.0 / math.sqrt(2)) * self.radius,
        )
        self.label.set_position((tx, ty))
        self.label.set_text(text)

    def set_figure(self, fig):
        super().set_figure(fig)
        self.label.set_figure(fig)

    @Circle.axes.setter
    def axes(self, new_axes):
        Circle.axes.fset(self, new_axes)  # type: ignore
        self.label.axes = new_axes

    def draw(self, renderer: RendererBase):
        super().draw(renderer)
        self.label.draw(renderer)

    def set_visible(self, b: bool) -> None:
        self.label.set_visible(b)
        return super().set_visible(b)
