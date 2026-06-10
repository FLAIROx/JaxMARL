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


# ---------------------------------------------------------------------------
# Tests for asymmetric num_colors != num_ranks (regression tests for the
# repeat-axis bug in _hint_fn that caused crashes on non-square configs).
# ---------------------------------------------------------------------------

def make_asymmetric_env(num_colors=3, num_ranks=5):
    """Create a HanabiEnv where num_colors != num_ranks."""
    # num_cards_of_rank must have exactly num_ranks entries
    default_cards = np.array([3, 2, 2, 2, 1])
    if num_ranks <= len(default_cards):
        num_cards_of_rank = default_cards[:num_ranks]
    else:
        num_cards_of_rank = np.concatenate([
            default_cards, np.full(num_ranks - len(default_cards), 2)
        ])
    return HanabiEnv(
        num_agents=2,
        num_colors=num_colors,
        num_ranks=num_ranks,
        hand_size=5,
        max_info_tokens=8,
        max_life_tokens=3,
        num_cards_of_rank=num_cards_of_rank,
    )


def test_asymmetric_reset_and_step_smoke():
    """Reset and take every action type with num_colors != num_ranks without crashing."""
    for num_colors, num_ranks in [(3, 5), (5, 3)]:
        env = make_asymmetric_env(num_colors=num_colors, num_ranks=num_ranks)
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        # verify observation shapes are correct
        for agent in env.agents:
            assert obs[agent].shape == (env.obs_size,), (
                f"obs shape mismatch for {num_colors}c/{num_ranks}r: "
                f"got {obs[agent].shape}, expected ({env.obs_size},)"
            )

        # take one action of each type: discard, play, color hint, rank hint
        actions_to_try = [
            0,                              # discard card 0
            env.hand_size,                  # play card 0
            env.color_action_range[0],      # hint color to other player
            env.rank_action_range[0],       # hint rank to other player
        ]
        for action in actions_to_try:
            key, subkey = jax.random.split(key)
            cur_player = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            actions = {agent: env.num_moves - 1 for agent in env.agents}  # noop
            actions[env.agents[int(cur_player)]] = int(action)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

    print("test_asymmetric_reset_and_step_smoke passed")


def test_asymmetric_color_hint_knowledge():
    """After a color hint with num_colors != num_ranks, card_knowledge must have
    the correct shape and eliminate the right possibilities."""
    env = make_asymmetric_env(num_colors=3, num_ranks=5)
    key = jax.random.PRNGKey(0)
    state = env.reset_game(key)

    # player 0 gives a color-0 hint to player 1
    color_hint_action = int(env.color_action_range[0])  # hint color 0 to next player
    new_state, reward = env.step_game(state, aidx=0, action=color_hint_action)

    # card_knowledge for player 1 should still have shape (hand_size, num_colors * num_ranks)
    p1_knowledge = new_state.card_knowledge[1]
    assert p1_knowledge.shape == (env.hand_size, env.num_colors * env.num_ranks), (
        f"knowledge shape wrong: {p1_knowledge.shape}"
    )

    # for cards that DON'T match color 0, the color-0 columns should be zeroed out
    p1_hand = new_state.player_hands[1]
    card_colors = jnp.sum(p1_hand, axis=2)  # (hand_size, num_colors)
    hint_color_vec = jnp.zeros(env.num_colors).at[0].set(1)  # color 0
    matches = jnp.matmul(card_colors, hint_color_vec)  # which cards have color 0

    knowledge_reshaped = p1_knowledge.reshape(env.hand_size, env.num_colors, env.num_ranks)
    for card_idx in range(env.hand_size):
        if not p1_hand[card_idx].any():
            continue  # skip empty card slots
        if matches[card_idx] == 0:
            # card does NOT have color 0 → color-0 row should be all zeros
            assert jnp.all(knowledge_reshaped[card_idx, 0, :] == 0), (
                f"card {card_idx} doesn't match color 0 but knowledge wasn't zeroed"
            )
        else:
            # card HAS color 0 → other color rows should be all zeros
            for c in range(1, env.num_colors):
                assert jnp.all(knowledge_reshaped[card_idx, c, :] == 0), (
                    f"card {card_idx} matches color 0 but color {c} wasn't zeroed"
                )

    print("test_asymmetric_color_hint_knowledge passed")


def test_asymmetric_rank_hint_knowledge():
    """After a rank hint with num_colors != num_ranks, card_knowledge must have
    the correct shape and eliminate the right possibilities."""
    env = make_asymmetric_env(num_colors=3, num_ranks=5)
    key = jax.random.PRNGKey(0)
    state = env.reset_game(key)

    # player 0 gives a rank-0 hint to player 1
    rank_hint_action = int(env.rank_action_range[0])  # hint rank 0 to next player
    new_state, reward = env.step_game(state, aidx=0, action=rank_hint_action)

    p1_knowledge = new_state.card_knowledge[1]
    assert p1_knowledge.shape == (env.hand_size, env.num_colors * env.num_ranks)

    p1_hand = new_state.player_hands[1]
    card_ranks = jnp.sum(p1_hand, axis=1)  # (hand_size, num_ranks)
    hint_rank_vec = jnp.zeros(env.num_ranks).at[0].set(1)
    matches = jnp.matmul(card_ranks, hint_rank_vec)

    knowledge_reshaped = p1_knowledge.reshape(env.hand_size, env.num_colors, env.num_ranks)
    for card_idx in range(env.hand_size):
        if not p1_hand[card_idx].any():
            continue
        if matches[card_idx] == 0:
            # card does NOT have rank 0 → rank-0 column should be all zeros
            assert jnp.all(knowledge_reshaped[card_idx, :, 0] == 0), (
                f"card {card_idx} doesn't match rank 0 but knowledge wasn't zeroed"
            )
        else:
            # card HAS rank 0 → other rank columns should be all zeros
            for r in range(1, env.num_ranks):
                assert jnp.all(knowledge_reshaped[card_idx, :, r] == 0), (
                    f"card {card_idx} matches rank 0 but rank {r} wasn't zeroed"
                )

    print("test_asymmetric_rank_hint_knowledge passed")


def test_asymmetric_full_game_rollout():
    """Run a complete game with num_colors != num_ranks using random legal actions."""
    for num_colors, num_ranks in [(3, 5), (5, 3), (2, 4)]:
        env = make_asymmetric_env(num_colors=num_colors, num_ranks=num_ranks)
        key = jax.random.PRNGKey(99)
        obs, state = env.reset(key)

        max_steps = env.deck_size + env.num_agents + 10  # generous upper bound
        for step_i in range(max_steps):
            if state.terminal:
                break

            key, subkey, action_key = jax.random.split(key, 3)
            cur_player = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
            legal_moves = env.get_legal_moves(state)
            cur_legal = legal_moves[env.agents[cur_player]]

            # pick a random legal action
            legal_indices = jnp.where(cur_legal, size=env.num_moves)[0]
            num_legal = int(cur_legal.sum())
            action_idx = jax.random.randint(action_key, (), 0, max(num_legal, 1))
            action = int(legal_indices[action_idx])

            actions = {agent: env.num_moves - 1 for agent in env.agents}
            actions[env.agents[cur_player]] = action
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

        assert step_i > 0, f"Game with {num_colors}c/{num_ranks}r ended immediately"

    print("test_asymmetric_full_game_rollout passed")


def main():
    test_injected_decks()
    test_asymmetric_reset_and_step_smoke()
    test_asymmetric_color_hint_knowledge()
    test_asymmetric_rank_hint_knowledge()
    test_asymmetric_full_game_rollout()

if __name__ == "__main__":
    main()
