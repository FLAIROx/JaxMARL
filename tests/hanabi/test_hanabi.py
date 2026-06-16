import os
import json
import numpy as np
import jax
from jax import numpy as jnp
from jaxmarl import make
from jaxmarl.environments.hanabi.hanabi import HanabiEnv

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

    obs, env_state = env.reset_from_deck(deck)
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


def make_single_life_env():
    return HanabiEnv(
        num_agents=2,
        num_colors=5,
        num_ranks=5,
        hand_size=5,
        max_info_tokens=8,
        max_life_tokens=1,
    )


def single_life_loss_deck():
    # Player 0 starts with a rank-1 card in slot 0 while fireworks are empty, so play-0 is invalid.
    deck = np.zeros((50, 2), dtype=int)
    deck[:, 0] = 0
    deck[:, 1] = 0
    deck[0] = np.array([0, 1])
    return jnp.array(deck)


def single_life_actions(env):
    return {
        "agent_0": env.hand_size,  # play card at slot 0
        "agent_1": env.num_moves - 1,  # noop for the non-acting player
    }


def legacy_reset_deck(env, key):
    colors = jnp.arange(env.num_colors)
    ranks = jnp.arange(env.num_ranks)
    ranks = jnp.repeat(ranks, env.num_cards_of_rank)
    color_rank_pairs = jnp.dstack(jnp.meshgrid(colors, ranks)).reshape(-1, 2)
    _, deck_key = jax.random.split(key)
    shuffled_pairs = jax.random.permutation(deck_key, color_rank_pairs, axis=0)
    deck = env._one_hot_encode_deck(shuffled_pairs)
    return deck.at[:env.num_agents * env.hand_size].set(
        jnp.zeros((env.num_colors, env.num_ranks))
    )


def test_fixed_player_order_preserves_legacy_deck_mapping():
    env = HanabiEnv(shuffle_player_order=False)
    key = jax.random.PRNGKey(42)

    _, state = env.reset(key)

    assert jnp.array_equal(state.seat_order, jnp.arange(env.num_agents))
    assert jnp.array_equal(state.deck, legacy_reset_deck(env, key))


def test_shuffled_player_order_is_deterministic_and_keeps_agent_keys_stable():
    env = HanabiEnv(num_agents=4)
    key = jax.random.PRNGKey(42)

    obs, state = env.reset(key)
    obs_again, state_again = env.reset(key)
    legal_moves = env.get_legal_moves(state)

    assert env.agents == ["agent_0", "agent_1", "agent_2", "agent_3"]
    assert list(obs.keys()) == env.agents
    assert list(obs_again.keys()) == env.agents
    assert list(legal_moves.keys()) == env.agents
    assert list(env.action_spaces.keys()) == env.agents
    assert jnp.array_equal(state.seat_order, state_again.seat_order)
    assert jnp.array_equal(state.seat_order, jnp.array([3, 1, 0, 2]))
    assert jnp.array_equal(jnp.sort(state.seat_order), jnp.arange(env.num_agents))


def test_injected_deck_reset_keeps_fixed_seat_order_and_dealt_cards():
    env = HanabiEnv()
    deck = get_decks()[0]

    _, state = env.reset_from_deck(deck)

    assert jnp.array_equal(state.seat_order, jnp.arange(env.num_agents))
    assert jnp.array_equal(state.player_hands[0], deck[: env.hand_size])


def test_shuffled_player_order_routes_current_seat_action_to_assigned_agent_key():
    env = HanabiEnv(num_agents=2)
    _, state = env.reset(jax.random.PRNGKey(42))

    assert jnp.array_equal(state.seat_order, jnp.array([1, 0]))
    assert int(jnp.argmax(state.cur_player_idx)) == 0

    _, next_state, _, _, _ = env.step_env(
        jax.random.PRNGKey(0),
        state,
        {
            "agent_0": 0,  # discard slot 0, wrong player for current seat
            "agent_1": env.hand_size,  # play slot 0, assigned to current seat
        },
    )

    assert int(jnp.sum(next_state.life_tokens)) == env.max_life_tokens - 1


def test_shuffled_player_order_routes_legal_moves_to_assigned_agent_key():
    env = HanabiEnv(num_agents=2)
    _, state = env.reset(jax.random.PRNGKey(42))

    legal_moves = env.get_legal_moves(state)

    assert jnp.array_equal(state.seat_order, jnp.array([1, 0]))
    assert bool(legal_moves["agent_1"][env.play_action_range].any())
    assert not bool(legal_moves["agent_1"][-1])
    assert not bool(legal_moves["agent_0"][env.play_action_range].any())
    assert bool(legal_moves["agent_0"][-1])


def test_shuffled_player_order_routes_observations_to_assigned_agent_key():
    env = HanabiEnv(num_agents=2)
    obs, state = env.reset(jax.random.PRNGKey(42))
    physical_seat_state = state.replace(seat_order=jnp.arange(env.num_agents))
    physical_seat_obs = env.get_obs(
        physical_seat_state,
        physical_seat_state,
        action=env.num_moves - 1,
    )

    assert jnp.array_equal(state.seat_order, jnp.array([1, 0]))
    assert jnp.array_equal(obs["agent_1"], physical_seat_obs["agent_0"])
    assert jnp.array_equal(obs["agent_0"], physical_seat_obs["agent_1"])


def test_step_game_terminates_immediately_when_last_life_is_lost():
    env = make_single_life_env()
    state = env.reset_game_from_deck_of_pairs(single_life_loss_deck())

    next_state, _ = env.step_game(state, aidx=0, action=env.hand_size)

    assert bool(next_state.out_of_lives)
    assert bool(next_state.terminal)
    assert int(jnp.sum(next_state.life_tokens)) == 0


def test_step_env_reports_done_on_losing_final_life():
    env = make_single_life_env()
    _, state = env.reset_from_deck_of_pairs(single_life_loss_deck())

    _, next_state, rewards, dones, info = env.step_env(
        jax.random.PRNGKey(0), state, single_life_actions(env)
    )

    assert bool(next_state.terminal)
    assert bool(dones["agent_0"])
    assert bool(dones["agent_1"])
    assert bool(dones["__all__"])
    assert rewards["__all__"] == rewards["agent_0"] == rewards["agent_1"]
    assert info == {}


def test_step_autoresets_after_final_life_loss():
    env = make_single_life_env()
    _, state = env.reset_from_deck_of_pairs(single_life_loss_deck())

    _, next_state, _, dones, _ = env.step(
        jax.random.PRNGKey(0), state, single_life_actions(env)
    )

    assert bool(dones["__all__"])
    assert not bool(next_state.terminal)
    assert not bool(next_state.out_of_lives)
    assert int(jnp.sum(next_state.life_tokens)) == env.max_life_tokens


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
