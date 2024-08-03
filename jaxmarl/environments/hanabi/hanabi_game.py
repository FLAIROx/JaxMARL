"""
This class models the game dynamics of Hanabi (reset and step of the game).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct
from typing import Tuple, Dict
from functools import partial
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


@struct.dataclass
class State:
    deck: chex.Array
    discard_pile: chex.Array
    fireworks: chex.Array
    player_hands: chex.Array
    info_tokens: chex.Array
    terminal: bool
    life_tokens: chex.Array
    card_knowledge: chex.Array
    colors_revealed: chex.Array
    ranks_revealed: chex.Array
    num_cards_dealt: int
    num_cards_discarded: int
    cur_player_idx: chex.Array
    out_of_lives: bool
    last_round_count: int
    bombed: bool
    remaining_deck_size: chex.Array
    turn: int
    score: int


class HanabiGame(MultiAgentEnv):

    def __init__(
        self,
        num_agents=2,
        num_colors=5,
        num_ranks=5,
        hand_size=5,
        max_info_tokens=8,
        max_life_tokens=3,
        num_cards_of_rank=np.array([3, 2, 2, 2, 1]),
        color_map=["R", "Y", "G", "W", "B"],
    ):
        super().__init__(num_agents)

        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)
        self.num_colors = num_colors
        self.num_ranks = num_ranks
        self.hand_size = hand_size
        self.max_info_tokens = max_info_tokens
        self.max_life_tokens = max_life_tokens
        self.num_cards_of_rank = num_cards_of_rank
        self.deck_size = np.sum(num_cards_of_rank) * num_colors
        self.color_map = color_map

        # action ranges - useful to know
        self.discard_action_range = jnp.arange(
            0,
            self.hand_size
        )
        self.play_action_range = jnp.arange(
            self.hand_size,
            2 * self.hand_size
        )
        self.color_action_range = jnp.arange(
            2 * self.hand_size,
            2 * self.hand_size + (self.num_agents - 1) * self.num_colors
        )
        self.rank_action_range = jnp.arange(
            2 * self.hand_size + (self.num_agents - 1) * self.num_colors,
            2 * self.hand_size + (self.num_agents - 1) * (self.num_colors + self.num_ranks)
        )

    @partial(jax.jit, static_argnums=[0])
    def get_first_state(self, deck: chex.Array) -> State:
        """Get the initial state of the game"""

        def _deal_cards(aidx, unused):
            """Deals cards to players from top of deck"""
            # top of deck is first array element
            start = aidx * self.hand_size
            hand = lax.dynamic_slice(
                deck, (start, 0, 0), (self.hand_size, self.num_colors, self.num_ranks)
            )

            return aidx + 1, hand

        _, hands = lax.scan(_deal_cards, 0, None, self.num_agents)
        num_cards_dealt = self.num_agents * self.hand_size

        # start off with all (color, rank) combinations being possible for all cards
        card_knowledge = jnp.ones(
            (self.num_agents, self.hand_size, (self.num_colors * self.num_ranks))
        )
        colors_revealed = jnp.zeros((self.num_agents, self.hand_size, self.num_colors))
        ranks_revealed = jnp.zeros((self.num_agents, self.hand_size, self.num_ranks))

        # init discard pile
        discard_pile = jnp.zeros_like(deck)
        num_cards_discarded = 0

        # remove dealt cards from deck
        deck = deck.at[:num_cards_dealt].set(
            jnp.zeros((self.num_colors, self.num_ranks))
        )
        remaining_deck_size = jnp.zeros(self.deck_size).at[:-num_cards_dealt].set(1)

        # thermometer encoded
        life_tokens = jnp.ones(self.max_life_tokens)
        info_tokens = jnp.ones(self.max_info_tokens).astype(int)
        fireworks = jnp.zeros((self.num_colors, self.num_ranks))

        # other state variable inits
        cur_player_idx = jnp.zeros(self.num_agents).at[0].set(1)
        terminal = False
        out_of_lives = False
        bombed = False
        last_round_count = 0

        state = State(
            deck=deck,
            discard_pile=discard_pile,
            fireworks=fireworks,
            player_hands=hands,
            info_tokens=info_tokens,
            terminal=terminal,
            life_tokens=life_tokens,
            card_knowledge=card_knowledge,
            colors_revealed=colors_revealed,
            ranks_revealed=ranks_revealed,
            num_cards_dealt=num_cards_dealt,
            num_cards_discarded=num_cards_discarded,
            cur_player_idx=cur_player_idx,
            out_of_lives=out_of_lives,
            last_round_count=last_round_count,
            bombed=bombed,
            remaining_deck_size=remaining_deck_size,
            turn=0,
            score=0,
        )

        return state

    @partial(jax.jit, static_argnums=[0])
    def reset_game(self, key: chex.PRNGKey) -> State:
        """Create a random deck and return the first state of the game"""

        # get all possible (colour, rank) pairs, including repetitions given num_cards_of_rank
        colors = jnp.arange(self.num_colors)
        ranks = jnp.arange(self.num_ranks)
        ranks = jnp.repeat(ranks, self.num_cards_of_rank)
        color_rank_pairs = jnp.dstack(jnp.meshgrid(colors, ranks)).reshape(-1, 2)
        # randomly shuffle (colour, rank) pairs
        key, _key = jax.random.split(key)
        shuffled_pairs = jax.random.permutation(_key, color_rank_pairs, axis=0)
        # generate one-hot encoded deck
        deck = self._one_hot_encode_deck(shuffled_pairs)
        return self.get_first_state(deck)

    @partial(jax.jit, static_argnums=[0])
    def reset_game_from_deck(self, deck: chex.Array) -> State:
        """Reset the game from a given deck of one-hot encoded cards."""
        return self.get_first_state(deck)

    @partial(jax.jit, static_argnums=[0])
    def reset_game_from_deck_of_pairs(self, deck_of_pairs: chex.Array):
        """Reset the game from a given deck of (color, value) pairs."""
        deck = self._one_hot_encode_deck(deck_of_pairs)
        return self.get_first_state(deck)

    @partial(jax.jit, static_argnums=[0])
    def step_game(
        self,
        state: State,
        aidx: int,
        action: int,
    ) -> Tuple[State, int]:
        """
        Execute the current player's action and its consequences
        """
        # check move type
        is_discard = self._is_discard(action)
        is_hint = self._is_hint(action)
        # initialise reward for move
        reward = 0

        def _discard_play_fn(state: State, action: int):
            """Discard or play selected card according to action selection"""
            # get hand and card info
            hand_before = state.player_hands.at[aidx].get()
            card_idx = (
                (is_discard * action)
                + (jnp.logical_not(is_discard) * (action - self.hand_size))
            ).astype(int)
            card = hand_before.at[card_idx].get()

            # gain an info token for discarding if discard action
            infos_remaining = jnp.sum(state.info_tokens)
            infos_depleted = infos_remaining < self.max_info_tokens
            new_infos = infos_remaining + (is_discard * infos_depleted)
            info_tokens = jnp.where(
                new_infos > 0,
                state.info_tokens.at[new_infos - 1].set(1),
                state.info_tokens
            )

            # play selected card if play action
            color, rank = jnp.nonzero(card, size=1)
            color_fireworks = state.fireworks.at[color].get()
            is_valid_play = rank == jnp.sum(color_fireworks)
            make_play = jnp.logical_and(
                is_valid_play, jnp.logical_not(is_discard)
            ).squeeze(0)

            # gain another info token if completed a color
            is_final_card = jnp.logical_and(
                make_play, rank == self.num_ranks - 1
            ).squeeze(0)
            infos_remaining = jnp.sum(info_tokens)
            infos_depleted = infos_remaining < self.max_info_tokens
            new_infos = infos_remaining + (is_final_card * infos_depleted)
            info_tokens = jnp.where(
                new_infos > 0,
                info_tokens.at[new_infos - 1].set(1),
                info_tokens
            )

            # increment fireworks if valid play action
            color_fireworks = color_fireworks.at[
                0, jnp.sum(color_fireworks).astype(int)
            ].set(make_play)
            fireworks = state.fireworks.at[color].set(color_fireworks)

            # the card must be discarded if action is discard or the play action is not valid
            discard_card = ((~is_valid_play) | (is_discard)).squeeze(0)
            discarded_card = jnp.zeros_like(card) + (discard_card * card)
            discard_pile = state.discard_pile.at[state.num_cards_discarded].set(
                discarded_card
            )
            num_cards_discarded = state.num_cards_discarded + discard_card

            # remove life token if invalid play
            life_lost = jnp.logical_not(
                jnp.logical_or(is_valid_play, is_discard)
            ).squeeze(0)
            num_life_tokens = jnp.sum(state.life_tokens).astype(int)
            life_tokens = jnp.where(
                life_lost,
                state.life_tokens.at[num_life_tokens - 1].set(0),
                state.life_tokens
            )

            # color hint knowledge removal
            player_colors_revealed = state.colors_revealed.at[aidx].get()
            player_colors_revealed = jnp.delete(
                player_colors_revealed, card_idx, axis=0, assume_unique_indices=True
            )
            player_colors_revealed = jnp.append(
                player_colors_revealed, jnp.zeros((1, self.num_colors)), axis=0
            )
            colors_revealed = state.colors_revealed.at[aidx].set(player_colors_revealed)
            # rank hint knowledge removal
            player_ranks_revealed = state.ranks_revealed.at[aidx].get()
            player_ranks_revealed = jnp.delete(
                player_ranks_revealed, card_idx, axis=0, assume_unique_indices=True
            )
            player_ranks_revealed = jnp.append(
                player_ranks_revealed, jnp.zeros((1, self.num_ranks)), axis=0
            )
            ranks_revealed = state.ranks_revealed.at[aidx].set(player_ranks_revealed)

            # deal a new card
            # check if in last round
            in_last_round = state.last_round_count > 0
            # deal empty card from top of deck if in last round
            new_card = state.deck.at[state.num_cards_dealt].get()
            hand_without_old = jnp.delete(
                hand_before, card_idx, axis=0, assume_unique_indices=True
            )
            new_hand = jnp.append(hand_without_old, new_card[jnp.newaxis, :, :], axis=0)
            hands = state.player_hands.at[aidx].set(new_hand)
            deck = state.deck.at[state.num_cards_dealt].set(jnp.zeros_like(card))
            # don't increment if in last round
            num_cards_dealt = lax.select(
                in_last_round, state.num_cards_dealt, state.num_cards_dealt + 1
            )
            remaining_deck_size = state.remaining_deck_size.at[-num_cards_dealt].set(0)

            # remove knowledge of selected card
            player_knowledge = state.card_knowledge.at[aidx].get()
            player_knowledge = jnp.delete(
                player_knowledge, card_idx, axis=0, assume_unique_indices=True
            )
            new_card_knowledge = jnp.where(
                new_card.any(),
                jnp.ones((1, self.num_colors * self.num_ranks)),
                jnp.zeros((1, self.num_colors * self.num_ranks)),
            )
            player_knowledge = jnp.append(
                player_knowledge,
                new_card_knowledge,
                axis=0,
            )
            card_knowledge = state.card_knowledge.at[aidx].set(player_knowledge)

            return state.replace(
                deck=deck,
                discard_pile=discard_pile,
                player_hands=hands,
                card_knowledge=card_knowledge,
                colors_revealed=colors_revealed,
                ranks_revealed=ranks_revealed,
                fireworks=fireworks,
                info_tokens=info_tokens,
                life_tokens=life_tokens,
                num_cards_dealt=num_cards_dealt,
                num_cards_discarded=num_cards_discarded,
                remaining_deck_size=remaining_deck_size,
            )

        def _hint_fn(state: State, action: int):
            is_color_hint = self._is_hint_color(action)
            is_rank_hint = self._is_hint_rank(action)

            hint_player, hint_idx = self._get_target_player_and_hint_index(aidx, action)

            hint_color = jnp.zeros(self.num_colors, dtype=int).at[hint_idx].set(1) * is_color_hint
            hint_rank = jnp.zeros(self.num_ranks, dtype=int).at[hint_idx].set(1) * is_rank_hint

            # get current card knowledge of relevant player
            cur_knowledge = state.card_knowledge.at[hint_player].get()

            # get relevant player's hand
            cards = state.player_hands.at[hint_player].get()
            card_colors = jnp.sum(cards, axis=2)
            card_ranks = jnp.sum(cards, axis=1)

            # check which cards have hinted color/rank
            color_hint_matches = jnp.matmul(card_colors, hint_color)
            rank_hint_matches = jnp.matmul(card_ranks, hint_rank)

            negative_color_hints = jnp.outer(1 - color_hint_matches, hint_color)
            negative_color_hints = jnp.repeat(
                negative_color_hints, self.num_colors, axis=1
            ).reshape(cur_knowledge.shape)
            negative_rank_hints = jnp.outer(1 - rank_hint_matches, hint_rank)
            negative_rank_hints = jnp.repeat(
                negative_rank_hints, self.num_ranks, axis=0
            ).reshape(cur_knowledge.shape)

            color_mask = (color_hint_matches * jnp.ones((self.num_colors, self.hand_size))).transpose()
            rank_mask = (rank_hint_matches * jnp.ones((self.num_ranks, self.hand_size))).transpose()

            color_hints = color_mask * (1 - hint_color * jnp.ones((self.hand_size, self.num_colors)))
            color_hints = jnp.repeat(color_hints, self.num_colors, axis=1).reshape(
                cur_knowledge.shape
            )
            rank_hints = rank_mask * (1 - hint_rank * jnp.ones((self.hand_size, self.num_ranks)))
            rank_hints = jnp.repeat(rank_hints, self.num_ranks, axis=0).reshape(
                cur_knowledge.shape
            )

            total_color_hint = color_hints + negative_color_hints
            total_rank_hint = rank_hints + negative_rank_hints

            new_color_knowledge = (
                cur_knowledge - (is_color_hint * total_color_hint)
            ).clip(min=0)
            new_rank_knowledge = (
                cur_knowledge - (is_rank_hint * total_rank_hint)
            ).clip(min=0)
            new_knowledge = jnp.where(
                is_color_hint, new_color_knowledge, new_rank_knowledge
            )
            card_knowledge = state.card_knowledge.at[hint_player].set(new_knowledge)

            colors_revealed_player = jnp.outer(color_hint_matches, hint_color)
            colors_revealed_player = (
                state.colors_revealed.at[hint_player].get() + colors_revealed_player
            ).clip(max=1)
            colors_revealed = state.colors_revealed.at[hint_player].set(
                colors_revealed_player
            )
            ranks_revealed_player = jnp.outer(rank_hint_matches, hint_rank)
            ranks_revealed_player = (
                state.ranks_revealed.at[hint_player].get() + ranks_revealed_player
            ).clip(max=1)
            ranks_revealed = state.ranks_revealed.at[hint_player].set(
                ranks_revealed_player
            )

            # remove an info token
            num_info_tokens = jnp.sum(state.info_tokens).astype(int) - 1
            info_tokens = jnp.arange(self.max_info_tokens)
            info_tokens = (info_tokens < num_info_tokens).astype(int)
            return state.replace(
                card_knowledge=card_knowledge,
                info_tokens=info_tokens,
                colors_revealed=colors_revealed,
                ranks_revealed=ranks_revealed,
            )

        # update fireworks
        fireworks_before = jnp.sum(state.fireworks, axis=(0, 1))
        state = lax.cond(is_hint, _hint_fn, _discard_play_fn, state, action)
        fireworks_after = jnp.sum(state.fireworks, axis=(0, 1))

        # check if lives left
        num_lives = jnp.sum(state.life_tokens)
        out_of_lives = num_lives == 0

        # check if terminal
        game_won = fireworks_after == (self.num_colors * self.num_ranks)
        deck_empty = state.num_cards_dealt >= self.deck_size
        last_round_count = state.last_round_count + deck_empty
        last_round_done = last_round_count == self.num_agents + 1
        terminal = jnp.logical_or(
            jnp.logical_or(state.out_of_lives, game_won), last_round_done
        )

        # define reward as difference in fireworks
        reward += jnp.logical_not(out_of_lives) * (fireworks_after - fireworks_before)
        # bomb-0 scoring
        reward -= out_of_lives * fireworks_after * jnp.logical_not(state.bombed)
        bombed = jnp.logical_or(out_of_lives, state.bombed)
        aidx = (aidx + 1) % self.num_agents

        cur_player_idx = jnp.zeros(self.num_agents).at[aidx].set(1)

        return (
            state.replace(
                terminal=terminal,
                cur_player_idx=cur_player_idx,
                out_of_lives=out_of_lives,
                last_round_count=last_round_count,
                bombed=bombed,
                turn=state.turn + 1,
                score=state.score + reward.astype(int),
            ),
            reward,
        )

    @partial(jax.jit, static_argnums=[0])
    def _one_hot_encode_deck(self, deck_of_pairs: chex.Array) -> chex.Array:
        """Generates one-hot card encodings given (color, rank) pairs."""

        def _gen_cards(aidx, _):
            color, rank = deck_of_pairs[aidx]
            card = jnp.zeros((self.num_colors, self.num_ranks))
            card = card.at[color, rank].set(1)

            return aidx + 1, card

        _, deck = lax.scan(_gen_cards, 0, None, self.deck_size)
        return deck

    @partial(jax.jit, static_argnums=[0])
    def _get_target_player_and_hint_index(self, aidx: int, action: int):
        """
        Determines the target player and the hint index based on the action of the current agent.
        In case you need a one hot encoded representation use:
        jax.nn.one_hot(hint_idx, num_classes=self.)

        Parameters:
        aidx (int): The index of the agent currently taking an action.
        Integer value that uniquely identifies each agent in the game.

        action (int): The raw, unnormalized action in the range 0 to num_actions.
        Represents the specific action that the agent has decided to take.

        Returns:
        target_player (int): The player who is the target of the hint.
        hint_idx (int): The index of the hint that was given.
        For example if the hint is about the value '1', which is the first possible rank action, the hint_idx will be 0.
        Similarly, if color map is ["R", "Y", "G", "W", "B"], these map to values [0, 1, 2, 3, 4] respectively.

        Note:
        This function assumes that the action given is a hint action.
        If an action that is not a hint is given, the behavior of this function will be unexpected.
        """
        is_hint_color = self._is_hint_color(action)
        # adjust the action - we start with a zero index removing the preceeding actions
        action_idx = action - 2 * self.hand_size
        action_idx = jnp.where(
            is_hint_color,
            action_idx,
            action_idx - (self.num_agents - 1) * self.num_colors
        )

        # get the index of hint (rank/color) played
        hint_idx = jnp.where(
            is_hint_color,
            jnp.mod(action_idx, self.num_colors),
            jnp.mod(action_idx, self.num_ranks)
        )

        # get the player to hint.
        target_player_absolute = jnp.where(
            is_hint_color,
            jnp.floor_divide(action_idx, self.num_colors),
            jnp.floor_divide(action_idx, self.num_ranks)
        )
        # adjust for the player who is hinting - wrap around
        target_player = jnp.mod(target_player_absolute + aidx + 1, self.num_agents).astype(int)

        return target_player, hint_idx

    @partial(jax.jit, static_argnums=[0])
    def _is_discard(self, action: int):
        """
        Determines is the action is the discard action where action is the integer value of the action.
        The ranges are defined in `self.discard_action_range`.
        """
        return jnp.logical_and(
            action >= self.discard_action_range[0],
            action <= self.discard_action_range[-1],
        )

    @partial(jax.jit, static_argnums=[0])
    def _is_play(self, action: int):
        """
        Determines is the action is the discard action where action is the integer value of the action.
        The ranges are defined in `self.play_action_range`.
        """
        return jnp.logical_and(
            action >= self.play_action_range[0],
            action <= self.play_action_range[-1],
        )

    @partial(jax.jit, static_argnums=[0])
    def _is_hint_color(self, action: int):
        """
        Determines is the action is the play action where action is the integer value of the action.
        The ranges are defined in `self.color_action_range`.
        """
        return jnp.logical_and(
            action >= self.color_action_range[0],
            action <= self.color_action_range[-1],
        )

    @partial(jax.jit, static_argnums=[0])
    def _is_hint_rank(self, action: int):
        """
        Determines is the action is the play action where action is the integer value of the action.
        The ranges are defined in `self.rank_action_range`.
        """
        return jnp.logical_and(
            action >= self.rank_action_range[0],
            action <= self.rank_action_range[-1],
        )

    @partial(jax.jit, static_argnums=[0])
    def _is_hint(self, action: int):
        """
        Determines is the action is the hint action where action is the integer value of the action.
        """
        return jnp.logical_or(
            self._is_hint_color(action),
            self._is_hint_rank(action),
        )

    @partial(jax.jit, static_argnums=[0])
    def _is_play_or_discard(self, action: int):
        """
        Determines if the action is either a play or discard action. Action is in the integer representation.
        """
        return jnp.logical_or(
            self._is_play(action),
            self._is_discard(action),
        )
