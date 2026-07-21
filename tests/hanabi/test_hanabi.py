import json
import os

import jax
import numpy as np
import pytest
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
    return deck.at[: env.num_agents * env.hand_size].set(
        jnp.zeros((env.num_colors, env.num_ranks))
    )


# Seat orders used by the routing tests. Seat 0 is the acting seat at reset, so
# every entry puts some agent other than agent_0 there: a permutation that left
# that seat alone would satisfy the routing assertions even with the seat
# mapping removed entirely.
SHUFFLED_SEAT_ORDERS = {
    2: [1, 0],
    3: [1, 2, 0],
    4: [3, 2, 0, 1],
    5: [3, 4, 2, 0, 1],
}


def seated(state, seat_order):
    """Put agents in an explicit seat order.

    seat_order only maps seats to agent keys, and cards are dealt to seats
    independently of it, so overriding it yields exactly the state a shuffled
    reset would produce -- without depending on what the RNG happens to permute.
    """
    return state.replace(seat_order=jnp.array(seat_order))


def test_fixed_player_order_preserves_legacy_deck_mapping():
    env = HanabiEnv(shuffle_player_order=False)
    key = jax.random.PRNGKey(42)

    _, state = env.reset(key)

    assert jnp.array_equal(state.seat_order, jnp.arange(env.num_agents))
    assert jnp.array_equal(state.deck, legacy_reset_deck(env, key))


@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_shuffled_player_order_is_deterministic_and_keeps_agent_keys_stable(num_agents):
    """The shuffled reset yields a valid permutation, reproducibly.

    Which permutation a given key produces is left to the RNG; the routing
    tests pin the orders they need explicitly instead of asserting on it.
    """
    env = HanabiEnv(num_agents=num_agents, shuffle_player_order=True)
    key = jax.random.PRNGKey(42)

    obs, state = env.reset(key)
    obs_again, state_again = env.reset(key)
    legal_moves = env.get_legal_moves(state)

    assert env.agents == [f"agent_{i}" for i in range(num_agents)]
    assert list(obs.keys()) == env.agents
    assert list(obs_again.keys()) == env.agents
    assert list(legal_moves.keys()) == env.agents
    assert list(env.action_spaces.keys()) == env.agents
    assert jnp.array_equal(state.seat_order, state_again.seat_order)
    assert jnp.array_equal(jnp.sort(state.seat_order), jnp.arange(num_agents))


@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_shuffling_only_reassigns_seats_and_leaves_the_deal_untouched(num_agents):
    """A shuffled reset differs from a fixed one only in seat_order.

    This is what lets the routing tests below build a shuffled state by
    overriding seat_order rather than fishing for a seed.
    """
    key = jax.random.PRNGKey(42)
    _, fixed = HanabiEnv(num_agents=num_agents).reset(key)
    _, shuffled = HanabiEnv(num_agents=num_agents, shuffle_player_order=True).reset(key)

    assert jnp.array_equal(shuffled.deck, fixed.deck)
    assert jnp.array_equal(shuffled.player_hands, fixed.player_hands)
    assert jnp.array_equal(shuffled.cur_player_idx, fixed.cur_player_idx)
    assert jnp.array_equal(
        seated(fixed, shuffled.seat_order).seat_order, shuffled.seat_order
    )


def test_injected_deck_reset_keeps_fixed_seat_order_and_dealt_cards():
    env = HanabiEnv()
    deck = get_decks()[0]

    _, state = env.reset_from_deck(deck)

    assert jnp.array_equal(state.seat_order, jnp.arange(env.num_agents))
    assert jnp.array_equal(state.player_hands[0], deck[: env.hand_size])


def test_shuffled_player_order_routes_current_seat_action_to_assigned_agent_key():
    env = HanabiEnv(num_agents=2)
    _, state = env.reset(jax.random.PRNGKey(42))
    state = seated(state, SHUFFLED_SEAT_ORDERS[2])  # agent_1 holds the acting seat

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
    state = seated(state, SHUFFLED_SEAT_ORDERS[2])  # agent_1 holds the acting seat

    legal_moves = env.get_legal_moves(state)

    assert bool(legal_moves["agent_1"][env.play_action_range].any())
    assert not bool(legal_moves["agent_1"][-1])
    assert not bool(legal_moves["agent_0"][env.play_action_range].any())
    assert bool(legal_moves["agent_0"][-1])


def test_shuffled_player_order_routes_observations_to_assigned_agent_key():
    env = HanabiEnv(num_agents=2)
    physical_seat_obs, state = env.reset(jax.random.PRNGKey(42))

    shuffled = seated(state, SHUFFLED_SEAT_ORDERS[2])  # agent_1 holds the acting seat
    obs = env.get_obs(shuffled, shuffled, action=env.num_moves - 1)

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
    print("Hanabi Test: test_injected_decks")
    actions_seq = get_action_sequences()
    decks = get_decks()
    true_scores = get_scores()
    scores = jax.jit(jax.vmap(get_injected_score))(decks, actions_seq)
    assert (true_scores == scores).all(), (
        "The injected decks-actions didn't produce the expeceted scores"
    )
    print("Test passed")


# ---------------------------------------------------------------------------
# New fixtures
# ---------------------------------------------------------------------------


def all_r1_deck():
    """50-card deck where every card is R1 (color 0, rank 0)."""
    return jnp.zeros((env.deck_size, 2), dtype=int)


def deck_with_first_card(color: int, rank: int):
    """50-card deck with (color, rank) as card 0; remainder are R1."""
    deck = np.zeros((env.deck_size, 2), dtype=int)
    deck[0] = [color, rank]
    return jnp.array(deck)


# ---------------------------------------------------------------------------
# Observation shape
# ---------------------------------------------------------------------------


def test_obs_shape_after_reset():
    """Each agent's observation has exactly obs_size features after reset."""
    key = jax.random.PRNGKey(0)
    obs, _ = env.reset(key)
    for agent in env.agents:
        assert obs[agent].shape == (env.obs_size,)


# ---------------------------------------------------------------------------
# Fireworks and scoring
# ---------------------------------------------------------------------------


def test_correct_play_advances_fireworks_and_score():
    """Playing a valid card increments fireworks and score by 1."""
    state = env.reset_game_from_deck_of_pairs(deck_with_first_card(0, 0))  # R1
    next_state, reward = env.step_game(state, aidx=0, action=env.hand_size)
    assert int(next_state.fireworks[0].sum()) == 1
    assert int(next_state.score) == 1
    assert int(reward) == 1


def test_wrong_play_loses_a_life():
    """Playing an invalid card spends a life token without advancing fireworks."""
    state = env.reset_game_from_deck_of_pairs(
        deck_with_first_card(0, 1)
    )  # R2 on empty fireworks
    next_state, reward = env.step_game(state, aidx=0, action=env.hand_size)
    assert int(next_state.life_tokens.sum()) == env.max_life_tokens - 1
    assert int(next_state.fireworks.sum()) == 0
    assert int(reward) == 0


def test_perfect_score_terminates_game():
    """Playing the 25th card triggers terminal and fills all fireworks."""
    state = env.reset_game_from_deck_of_pairs(deck_with_first_card(4, 4))  # B5
    near_perfect = jnp.ones((env.num_colors, env.num_ranks)).at[4, 4].set(0)
    state = state.replace(fireworks=near_perfect)
    next_state, _ = env.step_game(state, aidx=0, action=env.hand_size)
    assert bool(next_state.terminal)
    assert int(next_state.fireworks.sum()) == env.num_colors * env.num_ranks


# ---------------------------------------------------------------------------
# Info token mechanics
# ---------------------------------------------------------------------------


def test_hint_spends_info_token():
    """Giving a hint reduces the info token count by one."""
    state = env.reset_game_from_deck_of_pairs(all_r1_deck())
    initial_tokens = int(state.info_tokens.sum())
    hint_action = 2 * env.hand_size  # first color-hint action (hint R to player 1)
    next_state, _ = env.step_game(state, aidx=0, action=hint_action)
    assert int(next_state.info_tokens.sum()) == initial_tokens - 1


def test_discard_gains_info_token():
    """Discarding when tokens are not full restores one token."""
    state = env.reset_game_from_deck_of_pairs(all_r1_deck())
    tokens = state.info_tokens.at[env.max_info_tokens - 1].set(0)
    state = state.replace(info_tokens=tokens)
    assert int(state.info_tokens.sum()) == env.max_info_tokens - 1
    next_state, _ = env.step_game(state, aidx=0, action=0)  # discard card 0
    assert int(next_state.info_tokens.sum()) == env.max_info_tokens


# ---------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------


def test_discard_is_illegal_when_info_tokens_full():
    """No discard action is legal for the acting player at the start of a game."""
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    legal = env.get_legal_moves(state)
    acting_seat = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
    acting_player = env.agents[int(state.seat_order[acting_seat])]
    assert not bool(legal[acting_player][env.discard_action_range].any())


def test_noop_is_legal_only_for_non_acting_player():
    """Noop is legal exactly for agents who are not the current player."""
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    legal = env.get_legal_moves(state)
    noop = env.num_moves - 1
    acting_seat = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
    acting_idx = int(state.seat_order[acting_seat])
    for i, agent in enumerate(env.agents):
        if i == acting_idx:
            assert not bool(legal[agent][noop])
        else:
            assert bool(legal[agent][noop])


# ---------------------------------------------------------------------------
# Last-round countdown
# ---------------------------------------------------------------------------


def test_last_round_count_increments_when_deck_empty():
    """last_round_count increases by 1 per turn once the deck is exhausted."""
    state = env.reset_game_from_deck_of_pairs(all_r1_deck())
    state = state.replace(num_cards_dealt=env.deck_size)
    hint_action = 2 * env.hand_size
    next_state, _ = env.step_game(state, aidx=0, action=hint_action)
    assert int(next_state.last_round_count) == 1
    assert not bool(next_state.terminal)


def test_last_round_terminates_game():
    """Game ends when last_round_count reaches num_agents + 1."""
    state = env.reset_game_from_deck_of_pairs(all_r1_deck())
    state = state.replace(
        num_cards_dealt=env.deck_size,
        last_round_count=env.num_agents,  # one step below the terminal threshold
    )
    hint_action = 2 * env.hand_size
    next_state, _ = env.step_game(state, aidx=0, action=hint_action)
    assert bool(next_state.terminal)
    assert int(next_state.last_round_count) == env.num_agents + 1


# ---------------------------------------------------------------------------
# Multi-player configurations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_multi_player_reset_and_step(num_agents, shuffled):
    """reset and step work for all valid player counts; obs shapes are correct.

    Run with seats in order and permuted: under a non-identity seat order,
    get_legal_moves and step_env must still agree on which agent key is acting.
    """
    env_mp = HanabiEnv(num_agents=num_agents)
    key = jax.random.PRNGKey(0)
    obs, state = env_mp.reset(key)

    if shuffled:
        state = seated(state, SHUFFLED_SEAT_ORDERS[num_agents])
        obs = env_mp.get_obs(state, state, action=env_mp.num_moves - 1)

    assert len(obs) == num_agents
    for agent in env_mp.agents:
        assert obs[agent].shape == (env_mp.obs_size,)

    assert jnp.array_equal(jnp.sort(state.seat_order), jnp.arange(num_agents))

    acting_seat = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
    acting_idx = int(state.seat_order[acting_seat])
    acting_agent = env_mp.agents[acting_idx]

    # the table must move the acting seat, or the assertions below would hold
    # even if seat mapping were dropped entirely
    assert (acting_idx != acting_seat) == shuffled

    # legal moves must be keyed by agent, not seat: exactly the acting agent is
    # barred from noop, and every other agent is restricted to it
    noop = env_mp.num_moves - 1
    legal = env_mp.get_legal_moves(state)
    for agent in env_mp.agents:
        assert bool(legal[agent][noop]) == (agent != acting_agent)

    actions = {agent: noop for agent in env_mp.agents}
    actions[acting_agent] = int(jnp.argmax(legal[acting_agent]))  # first legal move

    obs2, state2, rewards, dones, _ = env_mp.step(key, state, actions)

    assert len(obs2) == num_agents
    assert "__all__" in dones
    assert "__all__" in rewards

    # the acting agent's move was actually executed, so play passed to the next seat
    assert int(jnp.argmax(state2.cur_player_idx)) == (acting_seat + 1) % num_agents


def main():
    test_injected_decks()


if __name__ == "__main__":
    main()
