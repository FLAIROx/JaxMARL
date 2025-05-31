import argparse
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.common import Actions
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts as layouts
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer


class InteractiveOvercookedV2:

    def __init__(self, layout, agent_view_size=None, no_jit=False, debug=False):
        self.debug = debug
        self.no_jit = no_jit

        self.env = OvercookedV2(layout=layout, agent_view_size=agent_view_size)
        self.viz = OvercookedV2Visualizer()

    def run(self, key):
        self.key = key
        with jax.disable_jit(self.no_jit):
            self._run()

    def _run(self):
        self._reset()

        self.viz.window.reg_key_handler(self._handle_input)
        self.viz.show(block=True)

    def _handle_input(self, event):
        if self.debug:
            print("Pressed", event.key)

        ACTION_MAPPING = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.up,
            "down": Actions.down,
            " ": Actions.interact,
            "tab": Actions.stay,
        }

        match event.key:
            case "escape":
                self.viz.window.close()
                return
            case "backspace":
                self._reset()
                return
            case key if key in ACTION_MAPPING:
                action = ACTION_MAPPING[key]
            case key:
                print(f"Key {key} not recognized")
                return

        self._step(action)

    def _redraw(self):
        self.viz.render(self.state, agent_view_size=self.env.agent_view_size)

    def _reset(self):
        self.key, key = jax.random.split(self.key)
        _, state = jax.jit(self.env.reset)(key)
        self.state = state

        self._redraw()

    def _step(self, action):
        self.key, subkey = jax.random.split(self.key)

        actions = {f"agent_{i}": jnp.array(action) for i in range(self.env.num_agents)}
        if self.debug:
            print("Actions: ", actions)

        obs, state, reward, done, info = jax.jit(self.env.step_env)(
            subkey, self.state, actions
        )
        self.state = state
        print(f"t={state.time}: reward={reward['agent_0']}, done = {done['__all__']}")

        if self.debug:
            a0_obs = obs["agent_0"]
            a0_obs = jnp.transpose(a0_obs, (2, 0, 1))
            print("Agent 0 observation: ", a0_obs)
            print("Reward: ", reward)
            print("Shaped reward: ", info["shaped_reward"])

        if done["__all__"]:
            self._reset()
        else:
            self._redraw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout", type=str, help="Overcooked layout", default="cramped_room"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=0,
    )
    parser.add_argument(
        "--agent_view_size",
        type=int,
        help="Number of cells in the agent view. If not provided, the agent will see the whole grid.",
    )
    parser.add_argument(
        "--no_jit",
        default=False,
        help="Disable JIT compilation",
        action="store_true",
    )
    parser.add_argument(
        "--debug", default=False, help="Debug mode", action="store_true"
    )
    args = parser.parse_args()

    if len(args.layout) == 0:
        raise ValueError("You must provide a layout.")
    layout = layouts[args.layout]

    interactive = InteractiveOvercookedV2(
        layout=layout,
        agent_view_size=args.agent_view_size,
        no_jit=args.no_jit,
        debug=args.debug,
    )

    key = jax.random.PRNGKey(args.seed)
    interactive.run(key)
