import os
import json
import numpy as np
import jax
from jax import numpy as jnp
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper

env = make("hanabi")
dir_path = os.path.dirname(os.path.realpath(__file__))

def pad_array(arr, target_length):
    pad_size = target_length - len(arr)
    if pad_size > 0:
        return np.pad(arr, (0, pad_size), "constant")
    else:
        return arr


def get_action_sequences():
    with open(os.path.join(dir_path, "actions.json"), "r") as f:
        actions_traj = json.load(f)

    max_len = max(len(a) for a in actions_traj)
    actions_traj = np.array([pad_array(np.array(a), max_len) for a in actions_traj])
    return jnp.array(actions_traj)


def get_decks():

    color_map = dict(zip(["R", "Y", "G", "W", "B"], range(5)))

    def gen_cards(color, rank):
        card = np.zeros((5, 5))
        card[color, rank] = 1
        return card

    with open(os.path.join(dir_path, "decks.json"), "r") as f:
        decks = json.load(f)

    # encode into card-matrices
    decks = np.array(
        [
            [gen_cards(color_map[card[0]], int(card[1]) - 1) for card in deck["deck"]]
            for deck in decks
        ]
    )

    return jnp.array(decks)


def get_scores():
    with open(os.path.join(dir_path, "scores.txt"), "r") as f:
        scores = [int(line.split(",")[1].split("\n")[0]) for line in f.readlines()]
    return jnp.array(scores)


def get_injected_score(deck, actions):

    def _env_step(env_state, action):

        curr_player = jnp.where(env_state.cur_player_idx == 1, size=1)[0][0]
        actions = jnp.array([20, 20]).at[curr_player].set(action)
        actions = {agent: action for agent, action in zip(env.agents, actions)}

        new_obs, new_env_state, reward, done, info = env.step(
            jax.random.PRNGKey(0), env_state, actions
        )
        return new_env_state, (reward, done)

    obs, env_state = env.reset_from_deck(jax.random.PRNGKey(0), deck)
    _, (rewards, dones) = jax.lax.scan(_env_step, env_state, actions)

    def first_episode_returns(rewards, dones):
        first_done = jax.lax.select(
            jnp.argmax(dones) == 0.0, dones.size, jnp.argmax(dones)
        )
        first_episode_mask = jnp.where(
            jnp.arange(dones.size) <= first_done, True, False
        )
        return jnp.where(first_episode_mask, rewards, 0.0).sum()

    cum_rewards = first_episode_returns(rewards["__all__"], dones["__all__"])
    return cum_rewards


def test_injected_decks():
    """
    This tests consists in injecting in the Hanabi environment a set of decks and actions that are known to produce a certain score.
    The test checks if the scores produced by the environment are the same as the expected ones.
    """
    print('Hanabi Test: test_injected_decks')
    actions_seq = get_action_sequences()
    decks = get_decks()
    true_scores = get_scores()
    scores = jax.jit(jax.vmap(get_injected_score))(decks, actions_seq)
    assert (
        true_scores == scores
    ).all(), "The injected decks-actions didn't produce the expeceted scores"
    print("Test passed")


def main():
    test_injected_decks()

if __name__ == "__main__":
    main()