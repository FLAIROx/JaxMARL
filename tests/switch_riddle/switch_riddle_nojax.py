# inspired from: https://colab.research.google.com/gist/MJ10/2c0d1972f3dd1edcc3cd17c636aac8d2/dial.ipynb
import numpy as np


class SwitchRiddleNoJax:
    def __init__(self, n_agents: int = 3, parallel_envs: int = 1):
        """
        Initializes the Switch Game with given parameters.
        """
        self.game_actions = {
            "NOTHING": 0,
            "SWITCH_LIGHT": 1,
            "TELL": 2,
        }

        self.game_states = {
            "OUTSIDE": 0,
            "INSIDE": 1,
        }

        self.n_agents = n_agents
        self.agents_ids = [f"agent_{i}" for i in range(n_agents)]
        self.bs = parallel_envs
        self.nsteps = 4 * n_agents - 6

        self.game_action_space = 2
        self.game_comm_limited = True

        self.initial_bulb_state = False

        self.reward_all_live = 1
        self.reward_all_die = -1

    def reset(self):
        """
        Resets the environment for the next episode and sets up the agent sequence for the next episode.
        """
        # Step count
        self.step_count = 0

        # Rewards
        self.reward = [
            dict(zip(["__all__"] + self.agents_ids, [0] * self.n_agents))
            for _ in range(self.bs)
        ]

        # Who has been in the room?
        self.has_been = np.zeros((self.bs, self.nsteps, self.n_agents))

        # Terminal state
        self.terminal = np.zeros(self.bs, dtype=np.int_)

        # Bulb state
        self.bulb_state = np.full(self.bs, self.initial_bulb_state)

        # Active agent
        self.active_agent = np.zeros((self.bs, self.nsteps), dtype=np.int_)
        for b in range(self.bs):
            for step in range(self.nsteps):
                agent_id = np.random.randint(self.n_agents)
                self.active_agent[b][step] = agent_id
                self.has_been[b][step][agent_id] = 1

        return self.get_obs(), self.get_state()

    def step(self, action):
        obs, state, reward, done = self.step_env(action)
        if np.all(done):
            obs, state = self.reset()
        return obs, state, reward, self.terminal

    def step_env(self, actions):
        """
        Takes a step in the environment based on the actions provided for each agent.
        """
        for b, batch_actions in enumerate(actions):
            if self.step_count < self.nsteps and not self.terminal[b]:
                agent_id = self.active_agent[b][self.step_count]
                agent_action = batch_actions[f"agent_{agent_id}"]

                if agent_action == self.game_actions["SWITCH_LIGHT"]:
                    self.bulb_state[b] = not self.bulb_state[b]
                elif agent_action == self.game_actions["TELL"]:
                    if np.all(self.has_been[b].sum(axis=0) > 0):
                        reward = self.reward_all_live
                    else:
                        reward = self.reward_all_die
                    self.reward[b] = {k: reward for k in self.reward[b]}
                    self.terminal[b] = 1

        self.step_count += 1
        if self.step_count >= self.nsteps:
            self.terminal = np.ones(self.bs, dtype=np.int_)

        return self.get_obs(), self.get_state(), self.reward, self.terminal

    def get_obs(self):
        """
        Returns the observation for each agent.
        """
        obs = []
        for b in range(self.bs):
            batch_obs = {}
            for i, agent_id in enumerate(self.agents_ids):
                in_room = (
                    int(self.active_agent[b][self.step_count] == i)
                    if self.step_count < self.nsteps
                    else 0
                )
                batch_obs[agent_id] = np.array([in_room, int(self.bulb_state[b])])
            obs.append(batch_obs)
        return obs

    def get_state(self):
        """
        Returns the state for each batch.
        """
        states = []
        for b in range(self.bs):
            state = np.zeros(self.n_agents + 1)
            if self.step_count < self.nsteps:
                state[self.active_agent[b][self.step_count]] = 1
            state[-1] = int(self.bulb_state[b])
            states.append(state)
        return states

    def render(self):
        """
        Displays the current state and active agent for each batch.
        """
        for b in range(self.bs):
            print(f"Batch {b}:")
            print(f"Step count: {self.step_count}")
            print(
                f"Agent in room: {self.active_agent[b][self.step_count] if self.step_count < self.nsteps else None}"
            )
            print(f"Bulb state: {int(self.bulb_state[b])}")
            print(f"Terminal state: {self.terminal[b]}")
            print(f"Reward: {self.reward[b]}")
            print()

    def sample_actions(self, action_name: str = None):
        """
        Generates random actions for each agent in a batch.
        """
        actions = []
        for b in range(self.bs):
            batch_actions = {}
            for agent_id in self.agents_ids:
                if action_name is None:
                    action = np.random.randint(len(self.game_actions))
                else:
                    action = self.game_actions[action_name]
                batch_actions[agent_id] = action
            actions.append(batch_actions)
        return actions

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__
