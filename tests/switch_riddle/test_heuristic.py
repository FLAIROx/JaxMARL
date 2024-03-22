import jax
import jax.numpy as jnp
from jaxmarl import make
import pytest

def get_nothing_actions(env):
    actions = {agent: jnp.array(env.game_actions["NOTHING"]) for agent in env.agents}
    return actions

def action_for_agent_in_room(env, state, action_name):
    actions = get_nothing_actions(env)
    actions[f"agent_{state.agent_in_room}"] = jnp.array(env.game_actions[action_name])
    return actions

@pytest.mark.parametrize("seed", [0, 999])
@pytest.mark.parametrize("num_agents", [3, 5, 10])
class TestHeuristic:

    def test_light_switch(self, seed, num_agents):
        """Check that the light is correctly switched"""
        env = make("switch_riddle", num_agents=num_agents)
        key = jax.random.PRNGKey(seed)

        for _ in range(100):
            key, key_reset, key_step = jax.random.split(key, 3)
            obs, state = env.reset(key_reset)

            prev_bulb_state = state.bulb_state
            actions = action_for_agent_in_room(env, state, "SWITCH_LIGHT")

            _, state, _, done, _ = env.step(key_step, state, actions)

            # Check if the bulb state has changed (only if the env didn't just restarted)
            if not state.step == 0 and state.bulb_state != ~prev_bulb_state:
                raise ValueError(
                    "Light state should have changed after switching, but it didn't."
                )
        print(
            'Test "light switch" passed with {} agents and seed {}.'.format(
                num_agents, seed
            )
        )


    def test_has_been(self, seed, num_agents):
        """Check that the has been parameter is correctly updated"""
        env = make("switch_riddle", num_agents=num_agents)
        key = jax.random.PRNGKey(seed)

        for _ in range(100):
            key, key_reset, key_step = jax.random.split(key, 3)
            obs, state = env.reset(key_reset)

            # make all the agents visit the room
            for i in range(num_agents):
                key, key_step = jax.random.split(key)
                state = state.replace(agent_in_room=jnp.array(i))

                actions = get_nothing_actions(env)
                _, state, _, _, _ = env.step(key_step, state, actions)

                if state.has_been.at[i].get() != 1:
                    raise ValueError(
                        f"Agent {i} visited the room but has_been parameter {state.has_been} doesn't count for that."
                    )

            if state.has_been.sum() != num_agents:
                raise ValueError(
                    "ALl agents visited the room but the has_been parameter doesn't say so."
                )
        print('Test "has been" passed with {} agents and seed {}.'.format(num_agents, seed))


    def test_positive_reward(self, seed, num_agents):
        """Check positive reward is correctly assigned"""
        env = make("switch_riddle", num_agents=num_agents)
        key = jax.random.PRNGKey(seed)

        for _ in range(100):
            key, key_reset, key_step = jax.random.split(key, 3)
            obs, state = env.reset(key_reset)

            # make all the agents visit the room
            for i in range(num_agents):
                key, key_step = jax.random.split(key)
                state = state.replace(agent_in_room=jnp.array(i))

                actions = get_nothing_actions(env)
                _, state, _, _, _ = env.step(key_step, state, actions)

            actions = action_for_agent_in_room(env, state, "TELL")
            _, state, rewards, _, _ = env.step(key_step, state, actions)

            if not all(rewards[a] == env.reward_all_live for a in env.agents):
                raise ValueError(
                    f"All agents have been in room and an agent spoke but the reward is not {env.reward_all_live}"
                )

        print(
            'Test "positive reward" passed with {} agents and seed {}.'.format(
                num_agents, seed
            )
        )


    def test_negative_reward(self, seed, num_agents):
        """Check negagtive reward is correctly assigned"""
        env = make("switch_riddle", num_agents=num_agents)
        key = jax.random.PRNGKey(seed)

        for _ in range(100):
            key, key_reset, key_step = jax.random.split(key, 3)
            obs, state = env.reset(key_reset)

            actions = action_for_agent_in_room(env, state, "TELL")
            _, state, rewards, _, _ = env.step(key_step, state, actions)

            if not all(rewards[a] == env.reward_all_die for a in env.agents):
                raise ValueError(
                    f"Not all agents have been in room and an agent spoke but the reward is not {env.reward_all_die}"
                )

        print(
            'Test "negative reward" passed with {} agents and seed {}.'.format(
                num_agents, seed
            )
        )


    def test_neutral_reward(self, seed, num_agents):
        """Check neutral reward is correctly assigned"""
        env = make("switch_riddle", num_agents=num_agents)
        key = jax.random.PRNGKey(seed)

        for _ in range(100):
            key, key_reset, key_step = jax.random.split(key, 3)
            obs, state = env.reset(key_reset)

            actions = get_nothing_actions(env)
            _, state, rewards, _, _ = env.step(key_step, state, actions)

            if not all(rewards[a] == 0 for a in env.agents):
                raise ValueError(f"Agents did nothing but the reward is not 0")

        print(
            'Test "neutral reward" passed with {} agents and seed {}.'.format(
                num_agents, seed
            )
        )


    def test_environment_termination(self, seed, num_agents):
        """Check environment is terminated correctly"""
        key = jax.random.PRNGKey(seed)
        env = make("switch_riddle", num_agents=num_agents)

        # Case where an agent speaks
        for _ in range(100):
            key, key_reset, key_step = jax.random.split(key, 3)
            obs, state = env.reset(key_reset)

            actions = action_for_agent_in_room(env, state, "TELL")
            obs, state, reward, done, infos = env.step_env(key_step, state, actions)

            assert done[
                "__all__"
            ], "The environment did not terminate correctly when an agent speaks."

        # Case where maximum time steps is reached
        for _ in range(100):
            key, key_reset, key_step = jax.random.split(key, 3)
            obs, state = env.reset(key_reset)

            for i in range(env.max_steps):
                key, key_step = jax.random.split(key)
                actions = get_nothing_actions(env)
                obs, state, reward, done, infos = env.step_env(key_step, state, actions)

            assert done[
                "__all__"
            ], "The environment did not terminate correctly when the maximum time steps is reached."

        print(
            'Test "environment termination" passed with {} agents and seed {}.'.format(
                num_agents, seed
            )
        )


    def test_consistency(self, seed, num_agents):
        """Check randomness of the environment is deterministic."""
        key = jax.random.PRNGKey(seed)
        env1 = make("switch_riddle", num_agents=num_agents)
        env2 = make("switch_riddle", num_agents=num_agents)

        def check_equal(d1, d2):
            assert d1.keys() == d2.keys(), "The dictionaries to compare have different keys"
            return all(jnp.all(d1[a] == d2[a]) for a in d1.keys())

        # Reset both environments with the same key
        obs1, state1 = env1.reset(key)
        obs2, state2 = env2.reset(key)

        # Ensure that the initial state and observations are the same
        assert check_equal(
            state1.__dict__, state2.__dict__
        ), "The initial states of the two environments do not match."
        assert check_equal(
            obs1, obs2
        ), "The initial observations of the two environments do not match."

        # Play out the sequence of actions in both environments
        for i in range(100):
            key, key_reset, key_act, key_step = jax.random.split(key, 4)

            key_act = jax.random.split(key_act, num_agents)
            actions1 = {
                agent: env1.action_space(agent).sample(key_act[i])
                for i, agent in enumerate(env1.agents)
            }
            actions2 = {
                agent: env2.action_space(agent).sample(key_act[i])
                for i, agent in enumerate(env2.agents)
            }

            assert check_equal(
                actions1, actions2
            ), f"The random actions of the two environments do not match at step {i}."

            obs1, state1, reward1, done1, _ = env1.step(key_step, state1, actions1)
            obs2, state2, reward2, done2, _ = env2.step(key_step, state2, actions2)

            # Ensure that the state and observations are the same at each step
            assert check_equal(
                state1.__dict__, state2.__dict__
            ), f"The states of the two environments do not match at step {i}."
            assert check_equal(
                obs1, obs2
            ), f"The observations of the two environments do not match at step {i}."
            assert check_equal(
                reward1, reward2
            ), f"The rewards of the two environments do not match at step {i}."
            assert check_equal(
                done1, done1
            ), f"The dones of the two environments do not match at step {i}."

        print(
            'Test "consistency" passed with {} agents and seed {}.'.format(num_agents, seed)
        )


'''def main():
    for num_agents in [3, 5, 10, 50, 100]:
        for key in [0, 999]:
            test_light_switch(key, num_agents)
            test_has_been(key, num_agents)
            test_positive_reward(key, num_agents)
            test_negative_reward(key, num_agents)
            test_neutral_reward(key, num_agents)
            test_environment_termination(key, num_agents)
            test_consistency(key, num_agents)
    print("All tests passed")'''

'''
if __name__ == "__main__":
    main()'''
