import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct
from typing import Tuple, Dict
from functools import partial
from gymnax.environments.spaces import Discrete
from smax.environments.multi_agent_env import MultiAgentEnv


@struct.dataclass
class State:
    legal_moves: chex.Array
    deck_size: int
    deck: chex.Array
    discard_pile: chex.Array
    fireworks: chex.Array
    player_hands: chex.Array
    info_tokens: chex.Array
    terminal: bool
    life_tokens: chex.Array
    card_knowledge: chex.Array
    num_cards_dealt: int
    num_cards_discarded: int
    last_moves: chex.Array
    cur_player_idx: chex.Array


class HanabiGame(MultiAgentEnv):

    def __init__(self, num_agents=2, num_colors=5, num_ranks=5, hand_size=5, max_info_tokens=8, max_life_tokens=3,
                 num_cards_of_rank=np.array([3, 2, 2, 2, 1]), agents=None, action_spaces=None, observation_spaces=None,
                 obs_size=None, num_moves=None):
        super().__init__(num_agents)

        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
        self.max_info_tokens = max_info_tokens
        self.max_life_tokens = max_life_tokens
        self.num_cards_of_rank = num_cards_of_rank

        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert len(
                agents) == num_agents, f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents

        if num_moves is None:
            self.num_moves = np.sum(np.array([
                # discard, play
                hand_size * 2,
                # hint color, rank
                num_agents * (num_colors + num_ranks)
            ])).squeeze()

        if obs_size is None:
            self.obs_size = (
                    (hand_size * (num_colors + num_ranks)) +
                    ((num_agents - 1) * hand_size * num_colors * num_ranks) +
                    self.num_moves + (num_colors * num_ranks) + max_info_tokens +
                    max_life_tokens + (num_agents * self.num_moves) + num_agents
            )

        self.action_set = jnp.arange(self.num_moves)
        if action_spaces is None:
            self.action_spaces = {i: Discrete(self.num_moves) for i in self.agents}
        if observation_spaces is None:
            self.observation_spaces = {i: Discrete(self.obs_size) for i in self.agents}

    def get_legal_moves(self, hands: chex.Array, fireworks: chex.Array, info_tokens: chex.Array,
                        cur_player: int) -> chex.Array:

        def _get_player_legal_moves(carry, unused):
            """
            Legal moves encoding in order:
            - discard for all cards in hand
            - play for all cards in hand
            - hint for all colors and ranks for all other players
            """
            move_idx = 1
            hands, fireworks, info_tokens, aidx = carry
            legal_moves = jnp.zeros(self.num_moves)
            # discard always legal
            legal_moves = legal_moves.at[1:self.hand_size].set(1)
            move_idx += self.hand_size
            # play moves always legal
            legal_moves = legal_moves.at[move_idx:move_idx + self.hand_size].set(1)
            move_idx += self.hand_size
            # hints depend on other player cards
            other_hands = jnp.delete(hands, aidx, axis=0, assume_unique_indices=True)

            def _get_hints_for_hand(carry, unused):
                aidx, other_hands = carry
                hand = other_hands[aidx]
                card_counts = jnp.sum(hand, axis=0)
                color_counts = jnp.sum(card_counts, axis=1)
                rank_counts = jnp.sum(card_counts, axis=0)
                colors_present = jnp.where(color_counts > 0, 1, 0)
                ranks_present = jnp.where(rank_counts > 0, 1, 0)
                valid_hints = jnp.concatenate([colors_present, ranks_present])
                carry = (aidx + 1, other_hands)

                return carry, valid_hints

            _, valid_hints = lax.scan(_get_hints_for_hand, (0, other_hands), None, self.num_agents - 1)
            valid_hints = jnp.roll(valid_hints, aidx)
            num_hints = (self.num_agents - 1) * (self.num_colors + self.num_ranks)
            valid_hints = jnp.concatenate(valid_hints, axis=0)
            info_tokens_available = (jnp.sum(info_tokens) != 0)
            valid_hints *= info_tokens_available
            legal_moves = legal_moves.at[move_idx:move_idx + num_hints].set(valid_hints)

            # only enable noop if not current player
            not_cur_player = (aidx != cur_player)
            legal_moves -= legal_moves * not_cur_player
            legal_moves = legal_moves.at[0].set(not_cur_player)

            return (hands, fireworks, info_tokens, aidx + 1), legal_moves

        _, legal_moves = lax.scan(_get_player_legal_moves, (hands, fireworks, info_tokens, 0), None, self.num_agents)

        return legal_moves

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:

        def _gen_cards(aidx, unused):
            color, rank = shuffled_pairs[aidx]
            card = jnp.zeros((self.num_colors, self.num_ranks))
            card = card.at[color, rank].set(1)
            return aidx + 1, card

        colors = jnp.arange(self.num_colors)
        ranks = jnp.arange(self.num_ranks)
        ranks = jnp.repeat(ranks, self.num_cards_of_rank)
        color_rank_pairs = jnp.dstack(jnp.meshgrid(colors, ranks)).reshape(-1, 2)
        key, _key = jax.random.split(key)
        shuffled_pairs = jax.random.permutation(_key, color_rank_pairs, axis=0)
        deck_size = np.sum(self.num_cards_of_rank) * self.num_colors
        _, deck = lax.scan(_gen_cards, 0, None, deck_size)

        # top of deck is first array element, not last
        def _deal_cards(aidx, unused):
            start = aidx * self.hand_size
            hand = lax.dynamic_slice(deck, (start, 0, 0), (self.hand_size, self.num_colors, self.num_ranks))
            return aidx + 1, hand

        _, hands = lax.scan(_deal_cards, 0, None, self.num_agents)

        # start off with all color, rank combinations being possible for all cards
        card_knowledge = jnp.ones((self.num_agents, self.hand_size, self.num_colors + self.num_ranks))

        # remove dealt cards from deck
        num_cards_dealt = self.num_agents * self.hand_size
        deck = deck.at[:num_cards_dealt].set(jnp.zeros((self.num_colors, self.num_ranks)))

        # init discard pile
        discard_pile = jnp.zeros_like(deck)
        num_cards_discarded = 0

        # thermometer encoded
        life_tokens = jnp.ones(self.max_life_tokens)
        info_tokens = jnp.ones(self.max_info_tokens).astype(int)
        fireworks = jnp.zeros((self.num_colors, self.num_ranks))

        # other state variable inits
        score = 0
        cur_player_idx = jnp.zeros(self.num_agents).at[0].set(1)
        terminal = False

        legal_moves = self.get_legal_moves(hands, fireworks, info_tokens, 0)

        last_moves = jnp.zeros((self.num_agents, self.num_moves))

        state = State(
            legal_moves=legal_moves,
            deck_size=deck_size,
            deck=deck,
            discard_pile=discard_pile,
            fireworks=fireworks,
            player_hands=hands,
            info_tokens=info_tokens,
            terminal=terminal,
            life_tokens=life_tokens,
            card_knowledge=card_knowledge,
            num_cards_dealt=num_cards_dealt,
            num_cards_discarded=num_cards_discarded,
            last_moves=last_moves,
            cur_player_idx=cur_player_idx
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> chex.Array:
        """
        Card knowledge observation: includes per card information of past hints
        as well as simple inferred knowledge.
        Currently only returns obs of current player.
        """
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        hands = state.player_hands
        other_hands = jnp.delete(hands, aidx, axis=0, assume_unique_indices=True)
        other_hands = jnp.roll(other_hands, aidx, axis=0)
        other_hands = jnp.reshape(other_hands, (-1,))
        knowledge = state.card_knowledge.at[aidx].get()
        knowledge = jnp.reshape(knowledge, (-1,))
        legal_moves = state.legal_moves.at[aidx].get()
        fireworks = jnp.reshape(state.fireworks, (-1,))
        last_moves = jnp.reshape(state.last_moves, (-1,))
        obs = jnp.concatenate([knowledge, other_hands, legal_moves, last_moves, fireworks,
                               state.info_tokens, state.life_tokens, state.cur_player_idx])
        return obs

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State,
                 action: int) -> Tuple[chex.Array, State, Dict, Dict, Dict]:
        state, reward = self.step_agent(key, state, action)

        done = self.terminal(state)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        rewards = {agent: reward for agent in self.agents}
        rewards["__all__"] = reward

        info = {}

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            rewards,
            dones,
            info
        )

    def step_agent(self, key: chex.PRNGKey, state: State,
                   action: int) -> Tuple[State, int]:
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        reward = 0

        is_discard = (action < self.hand_size)
        is_hint = ((2 * self.hand_size) <= action)

        def _discard_play_fn(state, action):
            # action -= 1
            hand_before = state.player_hands.at[aidx].get()
            card_idx = ((is_discard * action) + (jnp.logical_not(is_discard) * (action - self.hand_size))
                        ).astype(int)
            card = hand_before.at[card_idx].get()
            # discard selected card if discard action
            discard_card = jnp.zeros_like(card) + (is_discard * card)
            discard_pile = state.discard_pile.at[state.num_cards_discarded].set(discard_card)
            num_cards_discarded = state.num_cards_discarded + is_discard
            infos_remaining = jnp.sum(state.info_tokens)
            infos_depleted = (infos_remaining < self.max_info_tokens)
            new_infos = (infos_remaining + (is_discard * infos_depleted)).astype(int)
            info_tokens = state.info_tokens.at[new_infos - 1].set(1)
            # play selected card if play action
            color, rank = jnp.nonzero(card, size=1)
            color_fireworks = state.fireworks.at[color].get()
            is_valid_play = (rank == jnp.sum(color_fireworks))
            make_play = jnp.logical_and(is_valid_play, jnp.logical_not(is_discard)).squeeze(0)
            color_fireworks = color_fireworks.at[0, jnp.sum(color_fireworks).astype(int)].set(make_play)
            fireworks = state.fireworks.at[color].set(color_fireworks)
            # remove life token if invalid play
            life_lost = jnp.logical_and(jnp.logical_not(is_valid_play),
                                        jnp.logical_not(is_discard)).squeeze(0)
            num_life_tokens = jnp.sum(state.life_tokens).astype(int)
            life_tokens = state.life_tokens.at[num_life_tokens - 1].set(jnp.logical_not(life_lost))
            # remove knowledge of selected card
            player_knowledge = state.card_knowledge.at[aidx].get()
            player_knowledge = player_knowledge.at[card_idx].set(jnp.ones(
                self.num_colors + self.num_ranks))
            card_knowledge = state.card_knowledge.at[aidx].set(player_knowledge)
            # deal a new card
            new_card = state.deck.at[state.num_cards_dealt].get()
            new_hand = hand_before.at[card_idx].set(new_card)

            hands = state.player_hands.at[aidx].set(new_hand)
            deck = state.deck.at[state.num_cards_dealt].set(jnp.zeros_like(card))
            num_cards_dealt = state.num_cards_dealt + 1

            return state.replace(
                deck=deck,
                discard_pile=discard_pile,
                player_hands=hands,
                card_knowledge=card_knowledge,
                fireworks=fireworks,
                info_tokens=info_tokens,
                life_tokens=life_tokens,
                num_cards_dealt=num_cards_dealt,
                num_cards_discarded=num_cards_discarded
            )

        def _hint_fn(state, action):
            # action -= 1
            action_idx = action - (2 * self.hand_size)
            hints_per_player = self.num_colors + self.num_ranks
            # get player hint is being given to
            hint_player_before = jnp.floor(action_idx / hints_per_player).astype(int)
            hint_player = ((aidx + 1 + hint_player_before) % self.num_agents).astype(int)
            hint_idx = (action_idx % hints_per_player).astype(int)
            # define hint as possibilities to remove
            hint = jnp.ones(hints_per_player)
            hint = hint.at[hint_idx].set(0)
            is_color_hint = (hint_idx < self.num_colors)
            is_rank_hint = jnp.logical_not(is_color_hint)
            hint_color = (hint.at[:self.num_colors].get() * is_color_hint).astype(int)
            hint_rank = (hint.at[self.num_colors:].get() * is_rank_hint).astype(int)
            # define negative hint as removal of one possibility
            neg_hint = jnp.zeros(hints_per_player)
            neg_hint = neg_hint.at[hint_idx].set(1)
            neg_hint_color = (neg_hint.at[:self.num_colors].get() * is_color_hint).astype(int)
            neg_hint_rank = (neg_hint.at[self.num_colors:].get() * is_rank_hint).astype(int)
            # get current card knowledge of relevant player
            cur_knowledge = state.card_knowledge.at[hint_player].get()
            cur_color_knowledge = cur_knowledge.at[:, :self.num_colors].get()
            cur_rank_knowledge = cur_knowledge.at[:, self.num_colors:].get()
            # get relevant player's hand
            cards = state.player_hands.at[hint_player].get()
            card_colors = jnp.sum(cards, axis=2)
            card_ranks = jnp.sum(cards, axis=1)
            # check which cards have hinted color/rank
            color_hint_matches = jnp.matmul(card_colors, hint_color)
            rank_hint_matches = jnp.matmul(card_ranks, hint_rank)
            # flip 1s and 0s because we are removing possibilities
            color_hint_matches_flipped = 1 - color_hint_matches
            rank_hint_matches_flipped = 1 - rank_hint_matches
            # update relevant player's card knowledge
            num_colors_poss = jnp.sum(cur_color_knowledge, axis=1)
            color_unknown = jnp.where(num_colors_poss == 1, 0, 1)
            updated_color_knowledge = cur_color_knowledge - jnp.outer(color_unknown * color_hint_matches_flipped,
                                                                      hint_color)
            num_ranks_poss = jnp.sum(cur_rank_knowledge, axis=1)
            rank_unknown = jnp.where(num_ranks_poss == 1, 0, 1)
            updated_rank_knowledge = cur_rank_knowledge - jnp.outer(rank_unknown * rank_hint_matches_flipped,
                                                                    hint_rank)
            # update card knowledge with negative information
            num_colors_poss = jnp.sum(cur_color_knowledge, axis=1)
            color_unknown = jnp.where(num_colors_poss == 1, 0, 1)
            updated_color_knowledge = updated_color_knowledge - jnp.outer(color_unknown * color_hint_matches,
                                                                      neg_hint_color)
            num_ranks_poss = jnp.sum(cur_rank_knowledge, axis=1)
            rank_unknown = jnp.where(num_ranks_poss == 1, 0, 1)
            updated_rank_knowledge = updated_rank_knowledge - jnp.outer(rank_unknown * rank_hint_matches,
                                                                    neg_hint_rank)
            updated_knowledge = jnp.concatenate([updated_color_knowledge, updated_rank_knowledge], axis=1)
            card_knowledge = state.card_knowledge.at[hint_player].set(updated_knowledge)
            # remove an info token
            num_info_tokens = jnp.sum(state.info_tokens).astype(int) - 1
            info_tokens = jnp.arange(self.max_info_tokens)
            info_tokens = (info_tokens < num_info_tokens).astype(int)
            return state.replace(
                card_knowledge=card_knowledge,
                info_tokens=info_tokens
            )

        fireworks_before = jnp.sum(state.fireworks, axis=(0, 1))
        state = lax.cond(is_hint, _hint_fn, _discard_play_fn, state, action)
        fireworks_after = jnp.sum(state.fireworks, axis=(0, 1))
        out_of_lives = (jnp.sum(state.life_tokens) == 0)
        game_won = (fireworks_after == (self.num_colors * self.num_ranks))
        deck_empty = (state.num_cards_dealt >= state.deck_size)
        terminal = jnp.logical_or(jnp.logical_or(out_of_lives, game_won), deck_empty)
        cur_player = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        legal_moves = self.get_legal_moves(state.player_hands, state.fireworks, state.info_tokens, cur_player)
        last_moves = state.last_moves.at[aidx, :].set(0)
        last_moves = last_moves.at[aidx, action].set(1)
        reward += (jnp.logical_not(out_of_lives) * (fireworks_after - fireworks_before))
        aidx = (aidx + 1) % self.num_agents

        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)

        return state.replace(terminal=terminal,
                             legal_moves=legal_moves,
                             last_moves=last_moves,
                             cur_player_idx=cur_player_idx), reward

    def terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        return state.terminal

    @property
    def name(self) -> str:
        """Environment name."""
        return "Hanabi"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.num_moves

    def observation_space(self, agent: str):
        """ Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """ Action space for a given agent."""
        return self.action_spaces[agent]