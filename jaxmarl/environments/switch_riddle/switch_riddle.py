import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete, MultiDiscrete
import chex
from flax import struct
from typing import Tuple, Dict
from functools import partial


@struct.dataclass
class State:
    bulb_state: bool  # light on or off
    agent_in_room: int  # index of the agent in room
    has_been: chex.Array  # [bool]*n_agents
    done: bool  # done is the same for all the agents
    step: int


class SwitchRiddle(MultiAgentEnv):
    def __init__(
        self,
        num_agents=3,
        initial_bulb_state=False,
        reward_all_die=-1.0,
        reward_all_live=1.0,
    ):
        assert num_agents >= 3, "The minimum number of agents for this environment is 3"
        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

        # action space
        self.game_actions = {
            "NOTHING": 0,
            "SWITCH_LIGHT": 1,
            "TELL": 2,
        }
        self.game_actions_idx = {v: k for k, v in self.game_actions.items()}
        self.action_spaces = {i: Discrete(len(self.game_actions)) for i in self.agents}

        # observation space is: IN_ROOM (1/0) and LIGHT STATE (1/0)
        self.observation_spaces = {i: MultiDiscrete([2, 2]) for i in self.agents}

        # Parameters
        self.max_steps = 4 * self.num_agents - 6
        self.initial_bulb_state = initial_bulb_state
        self.reward_all_die = reward_all_die
        self.reward_all_live = reward_all_live

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        state = State(
            bulb_state=jnp.array(self.initial_bulb_state),
            agent_in_room=jax.random.randint(
                key, shape=(), minval=0, maxval=self.num_agents
            ),
            has_been=jnp.full(self.num_agents, 0),
            done=jnp.array(False),
            step=0,
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        # get the actions as array
        actions = jnp.array([actions[i] for i in self.agents])

        # the relevant action is only the one of the current agent in the room
        bulb_state = state.bulb_state
        agent_in_room = state.agent_in_room
        agent_action = actions.at[
            agent_in_room
        ].get()  # the relevant action is the one of the agent in the room
        has_been = state.has_been.at[agent_in_room].set(1)  # current agent is in room
        done = state.done
        step = state.step + 1

        # check if to change the state of the light
        bulb_state = jnp.where(
            agent_action == self.game_actions["SWITCH_LIGHT"],
            ~state.bulb_state,  # change the state of the light
            state.bulb_state,  # do nothing
        ).squeeze()

        # get the reward if agent spoke
        reward = jnp.where(
            agent_action == self.game_actions["TELL"],
            jnp.where(
                jnp.all(has_been),
                self.reward_all_live,  # positive reward if agent spoke and all agents have been in room
                self.reward_all_die,  # negative reward if agent spoke and not all agents have been in room
            ),
            0.0,  # no reward if agent didn't speak
        ).squeeze()

        # done if an agent spoke or maximum step reached
        done = jnp.logical_or(
            agent_action == self.game_actions["TELL"], state.step + 1 >= self.max_steps
        ).squeeze()

        # update the environment internal state
        state = State(
            bulb_state=bulb_state,
            agent_in_room=jax.random.randint(
                key, shape=(), minval=0, maxval=self.num_agents
            ),
            has_been=has_been,
            done=done,
            step=step,
        )

        # prepare outputs
        obs = self.get_obs(state)
        rewards = {a: reward for a in self.agents}
        dones = {a: done for a in self.agents + ["__all__"]}
        info = {}

        return obs, state, rewards, dones, info

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> dict:
        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> jnp.ndarray:
            """Return observation for agent i."""
            #print('state agent in room', state.agent_in_room)
            agent_in_room = (state.agent_in_room == aidx)
            #print('agent in room', agent_in_room)
            bulb_state = jnp.logical_and(
                agent_in_room, # only the current agent in the room can see the bulb state
                state.bulb_state
            ).squeeze()
            #print('agent shape', agent_in_room.shape, 'bulb state', bulb_state.shape)
            return jnp.array([agent_in_room, bulb_state]).astype(int)

        obs = _observation(self.agent_range, state)
        #print('obs', obs.shape)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    def render(self, state: State):
        print(f"\nCurrent step: {state.step}")
        print(f"Bulb state: {'On' if state.bulb_state else 'Off'}")
        print(f"Agent in room: {state.agent_in_room}")
        print(f"Agents been in room: {state.has_been}")
        print(f"Done: {state.done}")

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__


def example():
    num_agents = 5
    key = jax.random.PRNGKey(0)

    from jaxmarl import make
    env = make('switch_riddle', num_agents=num_agents)

    obs, state = env.reset(key)
    env.render(state)

    for _ in range(20):
        key, key_reset, key_act, key_step = jax.random.split(key, 4)

        env.render(state)
        print("obs:", obs)

        # Sample random actions.
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        print(
            "action:",
            env.game_actions_idx[actions[env.agents[state.agent_in_room]].item()],
        )

        # Perform the step transition.
        obs, state, reward, done, infos = env.step(key_step, state, actions)

        print("reward:", reward["agent_0"])


if __name__ == "__main__":
    with jax.disable_jit():
        example()
