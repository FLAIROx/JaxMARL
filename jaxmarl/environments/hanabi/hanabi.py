"""
JaxMarl Hanabi Environment
"""
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import chex
from typing import Tuple, Dict
from functools import partial
from gymnax.environments.spaces import Discrete
from .hanabi_game import HanabiGame, State


class HanabiEnv(HanabiGame):

    def __init__(
        self,
        num_agents=2,
        num_colors=5,
        num_ranks=5,
        hand_size=5,
        max_info_tokens=8,
        max_life_tokens=3,
        num_cards_of_rank=np.array([3, 2, 2, 2, 1]),
        agents=None,
        action_spaces=None,
        observation_spaces=None,
        num_moves=None,
    ):
        super().__init__(
            num_agents=num_agents,
            num_colors=num_colors,
            num_ranks=num_ranks,
            hand_size=hand_size,
            max_info_tokens=max_info_tokens,
            max_life_tokens=max_life_tokens,
            num_cards_of_rank=num_cards_of_rank,
        )

        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert (
                len(agents) == num_agents
            ), f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents

        if num_moves is None:
            self.num_moves = np.sum(
                np.array(
                    [
                        # discard, play
                        hand_size * 2,
                        # hint color, rank
                        (num_agents - 1) * (num_colors + num_ranks),
                        # noop
                        1,
                    ]
                )
            ).squeeze()

        self.action_set = jnp.arange(self.num_moves)
        self.action_encoding = {}
        for i, a in enumerate(self.action_set):
            if a < hand_size:
                move_type = f'D{i%hand_size}'
            elif a < hand_size*2:
                move_type = f'P{i%hand_size}'
            elif a < hand_size*2 + num_colors:
                move_type = f'H{self.color_map[i - hand_size*2]}' 
            elif a < hand_size*2 + (num_colors + num_ranks):
                move_type = f'H{i - (hand_size*2+num_colors)+1}' 
            else:
                move_type = 'N'
            self.action_encoding[int(a)] = move_type

        # useful ranges to know the type of the action
        self.discard_action_range = jnp.arange(0, self.hand_size)
        self.play_action_range = jnp.arange(self.hand_size, 2 * self.hand_size)
        self.color_action_range = jnp.arange(
            2 * self.hand_size, 2 * self.hand_size + self.num_colors
        )
        self.rank_action_range = jnp.arange(
            2 * self.hand_size + self.num_colors,
            2 * self.hand_size + self.num_colors + self.num_ranks,
        )

        # number of features
        self.hands_n_feats = (
            (self.num_agents - 1) * self.hand_size * self.num_colors * self.num_ranks
            + self.num_agents  # hands of all the other agents + agents' missing cards
        )
        self.board_n_feats = (
            (self.deck_size - self.num_agents * self.hand_size)
            + self.num_colors * self.num_ranks  # deck-initial cards, thermometer
            + self.max_info_tokens  # fireworks, OH
            + self.max_life_tokens  # info tokens, OH  # life tokens, OH
        )
        self.discards_n_feats = self.num_colors * self.num_cards_of_rank.sum()
        self.last_action_n_feats = (
            self.num_agents
            + 4  # acting player index
            + self.num_agents  # move type
            + self.num_colors  # target player index
            + self.num_ranks  # color revealed
            + self.hand_size  # rank revalued
            + self.hand_size  # reveal outcome
            + self.num_colors * self.num_ranks  # position played/discared
            + 1  # card played/discarded
            + 1  # card played score  # card played added info toke
        )
        self.v0_belief_n_feats = (
            self.num_agents
            * self.hand_size
            * (  # feats for each card, mine and other players
                self.num_colors * self.num_ranks + self.num_colors + self.num_ranks
            )  # 35 feats per card (25deductions+10hints)
        )

        self.obs_size = (
            self.hands_n_feats
            + self.board_n_feats
            + self.discards_n_feats
            + self.last_action_n_feats
            + self.v0_belief_n_feats
        )

        if action_spaces is None:
            self.action_spaces = {i: Discrete(self.num_moves) for i in self.agents}
        if observation_spaces is None:
            self.observation_spaces = {i: Discrete(self.obs_size) for i in self.agents}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:
        """Reset the environment and return the initial observation."""
        state = self.reset_game(key)
        obs = self.get_obs(state, state, action=20)
        return obs, state
    
    @partial(jax.jit, static_argnums=[0])
    def reset_from_deck(self, key: chex.PRNGKey, deck:chex.Array) -> Tuple[Dict, State]:
        """Inject a deck in the game. Useful for testing."""
        state = self.reset_game_from_deck(key, deck)
        obs = self.get_obs(state, state, action=20)
        return obs, state

    @partial(jax.jit, static_argnums=[0])
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict,
    ) -> Tuple[chex.Array, State, Dict, Dict, Dict]:
        """Execute the environment step."""

        # get actions as array
        actions = jnp.array([actions[i] for i in self.agents])
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        action = actions.at[aidx].get()

        # execute the current player's action and its consequences
        old_state = state
        new_state, reward = self.step_game(state, aidx, action)

        done = self.terminal(new_state)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        rewards = {agent: reward for agent in self.agents}
        rewards["__all__"] = reward

        info = {}

        obs = lax.stop_gradient(self.get_obs(new_state, old_state, action))

        return (obs, lax.stop_gradient(new_state), rewards, dones, info)

    @partial(jax.jit, static_argnums=[0])
    def get_obs(
        self, new_state: State, old_state: State, action: chex.Array = 20
    ) -> Dict:
        """Get all agents' observations."""

        # no agent-specific obs
        board_fats = self.get_board_fats(new_state)
        discard_feats = self._binarize_discard_pile(new_state.discard_pile)

        def _observe(aidx: int):

            # HANDS FEATURES: my masked hand, other agents hands, missing cards per agent
            hands_from_self = jnp.roll(
                new_state.player_hands, -aidx, axis=0
            )  # make the current player hand first
            other_hands = jnp.delete(
                hands_from_self, 0, axis=0, assume_unique_indices=True
            ).ravel()
            missing_cards = ~hands_from_self.any(
                axis=(1, 2, 3)
            )  # check if some player is missing a card
            hands_feats = jnp.concatenate((other_hands, missing_cards))

            # LAST ACTION FEATS
            last_action_feats = jnp.where(
                new_state.turn
                == 0,  # no features if first turn because no actions were made
                jnp.zeros(self.last_action_n_feats),
                self.get_last_action_feats(aidx, old_state, new_state, action),
            )

            # BELIEF FEATS
            belief_v0_feats = self.get_v0_belief_feats(aidx, new_state)

            return jnp.concatenate(
                (
                    hands_feats,
                    board_fats,
                    discard_feats,
                    last_action_feats,
                    belief_v0_feats,
                )
            )

        obs = jax.vmap(_observe)(self.agent_range)

        return {a: obs[i] for i, a in enumerate(self.agents)}

    def get_legal_moves(self, state: State) -> chex.Array:
        """Get all agents' legal moves"""

        @partial(jax.vmap, in_axes=[0, None])
        def _legal_moves(aidx: int, state: State) -> chex.Array:
            """
            Legal moves encoding in order:
            - discard for all cards in hand
            - play for all cards in hand
            - hint for all colors and ranks for all other players
            """
            move_idx = 0
            legal_moves = jnp.zeros(self.num_moves)
            hands = state.player_hands
            info_tokens = state.info_tokens
            # discard legal when discard tokens are not full
            is_not_max_info_tokens = jnp.sum(state.info_tokens) < 8
            legal_moves = legal_moves.at[move_idx : move_idx + self.hand_size].set(
                is_not_max_info_tokens
            )
            move_idx += self.hand_size
            # play moves always legal
            legal_moves = legal_moves.at[move_idx : move_idx + self.hand_size].set(1)
            move_idx += self.hand_size
            # hints depend on other player cards
            other_hands = jnp.delete(hands, aidx, axis=0, assume_unique_indices=True)

            def _get_hints_for_hand(carry, unused):
                """Generates valid hints given hand"""
                aidx, other_hands = carry
                hand = other_hands[aidx]

                # get occurrences of each card
                card_counts = jnp.sum(hand, axis=0)
                # get occurrences of each color
                color_counts = jnp.sum(card_counts, axis=1)
                # get occurrences of each rank
                rank_counts = jnp.sum(card_counts, axis=0)
                # check which colors/ranks in hand
                colors_present = jnp.where(color_counts > 0, 1, 0)
                ranks_present = jnp.where(rank_counts > 0, 1, 0)

                valid_hints = jnp.concatenate([colors_present, ranks_present])
                carry = (aidx + 1, other_hands)

                return carry, valid_hints

            _, valid_hints = lax.scan(
                _get_hints_for_hand, (0, other_hands), None, self.num_agents - 1
            )
            # make other player positions relative to current player
            valid_hints = jnp.roll(valid_hints, -aidx, axis=0)
            # include valid hints in legal moves
            num_hints = (self.num_agents - 1) * (self.num_colors + self.num_ranks)
            valid_hints = jnp.concatenate(valid_hints, axis=0)
            info_tokens_available = jnp.sum(info_tokens) != 0
            valid_hints *= info_tokens_available
            legal_moves = legal_moves.at[move_idx : move_idx + num_hints].set(
                valid_hints
            )

            # only enable noop if not current player
            cur_player = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            not_cur_player = aidx != cur_player
            legal_moves -= legal_moves * not_cur_player
            legal_moves = legal_moves.at[-1].set(not_cur_player)

            return legal_moves

        legal_moves = _legal_moves(self.agent_range, state)

        return {a: legal_moves[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def get_last_action_feats_(
        self, aidx: int, old_state: State, new_state: State, action: chex.Array
    ):
        """Get the features of the last action taken"""

        acting_player_index = old_state.cur_player_idx  # absolute OH index
        target_player_index = new_state.cur_player_idx  # absolute OH index
        acting_player_relative_index = jnp.roll(
            acting_player_index, -aidx
        )  # relative OH index
        target_player_relative_index = jnp.roll(
            target_player_index, -aidx
        )  # relative OH index

        # in obl the encoding order here is: play, discard, reveal_c, reveal_r
        move_type = jnp.where(  # hard encoded but hey ho let's go
            (action >= 0) & (action < 5),  # discard
            jnp.array([0, 1, 0, 0]),
            jnp.where(
                (action >= 5) & (action < 10),  # play
                jnp.array([1, 0, 0, 0]),
                jnp.where(
                    (action >= 10) & (action < 15),  # reveal_c
                    jnp.array([0, 0, 1, 0]),
                    jnp.where(
                        (action >= 15) & (action < 20),  # reveal_r
                        jnp.array([0, 0, 0, 1]),
                        jnp.array([0, 0, 0, 0]),  # invalid
                    ),
                ),
            ),
        )

        target_player_relative_index_feat = jnp.where(
            action >= 10,  # only for hint actions
            target_player_relative_index,
            jnp.zeros(self.num_agents),
        )

        # get the hand of the target player
        target_hand = new_state.player_hands[
            jnp.nonzero(target_player_index, size=1)[0][0]
        ]

        color_revealed = jnp.where(  # which color was revealed by action (oh)?
            action == self.color_action_range,
            1.0,
            jnp.zeros(self.color_action_range.size),
        )

        rank_revealed = jnp.where(  # which rank was revealed by action (oh)?
            action == self.rank_action_range,
            1.0,
            jnp.zeros(self.rank_action_range.size),
        )

        # cards that have the color that was revealed
        color_revealed_cards = jnp.where(
            (target_hand.sum(axis=(2)) == color_revealed).all(
                axis=1
            ),  # color of the card==color reveled
            1,
            0,
        )

        # cards that have the color that was revealed
        rank_revealed_cards = jnp.where(
            (target_hand.sum(axis=(1)) == rank_revealed).all(
                axis=1
            ),  # color of the card==color reveled
            1,
            0,
        )

        # cards that are caught by the hint
        reveal_outcome = color_revealed_cards | rank_revealed_cards

        # card that was played-discarded
        pos_played_discarded = jnp.where(
            action < 2 * self.hand_size,
            jnp.arange(self.hand_size) == action % self.hand_size,
            jnp.zeros(self.hand_size),
        )
        actor_hand_before = old_state.player_hands[
            jnp.nonzero(acting_player_index, size=1)[0][0]
        ]

        played_discarded_card = jnp.where(
            pos_played_discarded.any(),
            actor_hand_before[jnp.nonzero(pos_played_discarded, size=1)[0][0]].ravel(),
            jnp.zeros(
                self.num_colors * self.num_ranks
            ),  # all zeros if no card was played
        )

        # effect of playing the card in the game
        card_played_score = (
            new_state.fireworks.sum(axis=(0, 1)) - old_state.fireworks.sum(axis=(0, 1))
        ) != 0

        # "added info token" boolean is present only when you get an info from playing the 5 of the color
        added_info_tokens = jnp.where(
            (action >= 5) & (action < 10),
            new_state.info_tokens.sum() > old_state.info_tokens.sum(),
            0,
        )

        feats = {
            "acting_player_relative_index": acting_player_relative_index,
            "move_type": move_type,
            "target_player_relative_index": target_player_relative_index_feat,
            "color_revealed": color_revealed,
            "rank_revealed": rank_revealed,
            "reveal_outcome": reveal_outcome,
            "pos_played_discarded": pos_played_discarded,
            "played_discarded_card": played_discarded_card,
            "card_played_score": card_played_score[np.newaxis],  # scalar
            "added_info_tokens": added_info_tokens[np.newaxis],  # scalar
        }

        return feats
    
    @partial(jax.jit, static_argnums=[0])
    def get_last_action_feats(
        self, aidx: int, old_state: State, new_state: State, action: chex.Array
    ):
        """Get the features of the last action taken"""
        last_action = self.get_last_action_feats_(aidx, old_state, new_state, action)
        last_action = jnp.concatenate((
            last_action['acting_player_relative_index'],
            last_action['move_type'],
            last_action['target_player_relative_index'],
            last_action['color_revealed'],
            last_action['rank_revealed'],
            last_action['reveal_outcome'],
            last_action['pos_played_discarded'],
            last_action['played_discarded_card'],
            last_action['card_played_score'],
            last_action['added_info_tokens'],
        ))

        return last_action


    @partial(jax.jit, static_argnums=[0])
    def get_board_fats(self, state: State):
        """Get the features of the board."""
        # by default the fireworks are incremental, i.e. [1,1,0,0,0] one and two are in the board
        # must be OH of only the highest rank, i.e. [0,1,0,0,0]
        keep_only_last_one = lambda x: jnp.where(
            jnp.arange(x.size) < (x.size - 1 - jnp.argmax(jnp.flip(x))),  # last argmax
            0,
            x,
        )

        fireworks = jax.vmap(keep_only_last_one)(state.fireworks)
        deck = jnp.any(jnp.any(state.deck, axis=1), axis=1).astype(int)
        deck = deck[
            -1 : self.num_agents * self.hand_size - 1 : -1
        ]  # avoid the first cards at beginning of episode and reset the order
        board_feats = jnp.concatenate(
            (deck, fireworks.ravel(), state.info_tokens, state.life_tokens)
        )

        return board_feats

    @partial(jax.jit, static_argnums=[0])
    def get_full_deck(self):
        """Get the full deck of cards."""
        def _gen_cards(aidx):
            """Generates one-hot card encodings given (color, rank) pairs"""
            color, rank = color_rank_pairs[aidx]
            card = jnp.zeros((self.num_colors, self.num_ranks))
            card = card.at[color, rank].set(1)

            return card

        colors = jnp.arange(self.num_colors)
        ranks = jnp.arange(self.num_ranks)
        ranks = jnp.repeat(ranks, self.num_cards_of_rank)
        color_rank_pairs = jnp.dstack(jnp.meshgrid(colors, ranks)).reshape(-1, 2)

        full_deck = jax.vmap(_gen_cards)(jnp.arange(self.deck_size))
        return full_deck

    @partial(jax.jit, static_argnums=[0])
    def get_v0_belief_feats(self, aidx: int, state: State):
        """Get the belief of the agent about the player hands."""

        full_deck = self.get_full_deck()

        def belief_per_hand(knowledge, color_hint, rank_hint):
            count = (
                full_deck.sum(axis=0).ravel()
                - state.discard_pile.sum(axis=0).ravel()
                - state.fireworks.ravel()
            )  # count of the remaining cards
            normalized_knowledge = knowledge * count
            normalized_knowledge /= normalized_knowledge.sum(axis=1)[:, np.newaxis]
            return jnp.concatenate(
                (normalized_knowledge, color_hint, rank_hint), axis=-1
            ).ravel()

        # compute my belief and the beliefs of other players, starting from self cards
        rel_pos = lambda x: jnp.roll(x, -aidx, axis=0)
        belief = jax.vmap(belief_per_hand)(
            rel_pos(state.card_knowledge),
            rel_pos(state.colors_revealed),
            rel_pos(state.ranks_revealed),
        )

        return belief.ravel()

    @partial(jax.jit, static_argnums=[0])
    def _binarize_discard_pile(self, discard_pile: chex.Array):
        """Binarize the discard pile to reduce dimensionality."""

        def binarize_ranks(n_ranks):
            tree = jax.tree_util.tree_map(
                lambda n_rank_present, max_ranks: jnp.where(
                    jnp.arange(max_ranks) >= n_rank_present,
                    jnp.zeros(max_ranks),
                    jnp.ones(max_ranks),
                ),
                [x for x in n_ranks],
                [x for x in self.num_cards_of_rank],
            )
            return jnp.concatenate(tree)

        binarized_pile = jax.vmap(binarize_ranks)(discard_pile.sum(axis=0)).ravel()

        return binarized_pile

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
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    def render_obs(self, obs: dict):
        # print the dictionary of agents observations
        for i, (agent, obs) in enumerate(obs.items()):
            print(f"Obs for {agent}")
            j = 0
            print("hand feats", obs[: self.hands_n_feats])
            j += self.hands_n_feats
            print("board feats", obs[j : j + self.board_n_feats])
            j += self.board_n_feats
            print("discard feats", obs[j : j + self.discards_n_feats])
            j += self.discards_n_feats
            print("last action feats", obs[j : j + self.last_action_n_feats])
            j += self.last_action_n_feats
            beliefs = obs[-self.v0_belief_n_feats :].reshape(
                self.num_agents, self.hand_size, -1
            )
            for z, b in enumerate(beliefs):
                b = b[:, : self.num_colors * self.num_ranks].reshape(
                    -1, self.num_colors, self.num_ranks
                )
                print(
                    f"agent {i} representation of the belief of its {z}th-relative agent:",
                    b,
                )
    
    def card_to_string(self, card: chex.Array) -> str:
        # transforms a card matrix to string
        if ~card.any():  # empyt card
            return ""
        color = jnp.argmax(card.sum(axis=1), axis=0)
        rank = jnp.argmax(card.sum(axis=0), axis=0)
        return f"{self.color_map[color]}{rank+1}"

    def render(self, state: State):
        """Render the state of the game as a string in console."""

        def get_actor_hand_str(aidx: int) -> str:
            # get the index of an actor and returns its hand (with knowledge per card) as string
            # TODO: missing the first numbers, don't know what they are

            colors_revealed = np.array(state.colors_revealed[aidx])
            ranks_revealed = np.array(state.ranks_revealed[aidx])
            knowledge = np.array(
                state.card_knowledge[aidx].reshape(
                    self.hand_size, self.num_colors, self.num_ranks
                )
            )
            actor_hand = np.array(state.player_hands[aidx])

            def get_card_knowledge_str(card_idx: int) -> str:
                color_hint = colors_revealed[card_idx]
                rank_hint = ranks_revealed[card_idx]
                card_hint = (
                    f"{'X' if ~color_hint.any() else self.color_map[jnp.argmax(color_hint)]}"
                    + f"{'X' if ~rank_hint.any() else jnp.argmax(rank_hint)+1}"
                )

                color_knowledge = knowledge[card_idx].any(axis=1)
                rank_knowledge = knowledge[card_idx].any(axis=0)
                color_knowledge_str = "".join(
                    c
                    for c, bool_idx in zip(self.color_map, color_knowledge)
                    if bool_idx
                )
                rank_knowledge_str = "".join(
                    str(r) for r in jnp.where(rank_knowledge)[0] + 1
                )
                card_knowledge = color_knowledge_str + rank_knowledge_str

                return f"{card_hint}|{card_knowledge}"

            actor_hand_str = [
                f"{i} {self.card_to_string(actor_hand[card_idx])} || {get_card_knowledge_str(card_idx)}"
                for i, card_idx in enumerate(range(self.hand_size))
            ]

            return actor_hand_str

        keep_only_last_one = lambda x: jnp.where(
            jnp.arange(x.size) < (x.size - 1 - jnp.argmax(jnp.flip(x))),  # last argmax
            0,
            x,
        )
        fireworks = jax.vmap(keep_only_last_one)(state.fireworks)
        fireworks_cards = [
            jnp.zeros((self.num_colors, self.num_ranks)).at[i].set(fireworks[i])
            for i in range(self.num_colors)
        ]

        board_info = {
            "turn": state.turn,
            "score": state.score,
            "information": int(state.info_tokens.sum()),
            "lives": int(state.life_tokens.sum()),
            "deck": int(state.deck.sum()),
            "discards": " ".join(self.card_to_string(card) for card in state.discard_pile),
            "fireworks": " ".join(self.card_to_string(card) for card in fireworks_cards),
        }

        for i, (k, v) in enumerate(board_info.items()):
            print(f"{k.capitalize()}: {v}")
            if i == 0:
                print()

        current_player = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        for aidx in range(self.num_agents):
            print(
                f"Actor {aidx} Hand:"
                + ("<-- current player" if aidx == current_player else "")
            )
            for card_str in get_actor_hand_str(aidx):
                print(card_str)

    def get_obs_str(self, new_state, old_state=None, action=20, include_belief=False, best_belief=5):
        """Get the observation as a string."""
        
        output = ""

        def get_actor_hand_str(aidx: int, belief: chex.Array=None, mask_hand=False) -> str:
            # get the index of an actor and returns its hand (with knowledge per card) as string
            # TODO: missing the first numbers, don't know what they are

            colors_revealed = np.array(new_state.colors_revealed[aidx])
            ranks_revealed = np.array(new_state.ranks_revealed[aidx])
            knowledge = np.array(
                new_state.card_knowledge[aidx].reshape(
                    self.hand_size, self.num_colors, self.num_ranks
                )
            )
            actor_hand = np.array(new_state.player_hands[aidx])

            def get_card_knowledge_str(card_idx: int) -> str:
                color_hint = colors_revealed[card_idx]
                rank_hint = ranks_revealed[card_idx]
                card_hint = (
                    f"{'' if ~color_hint.any() else self.color_map[jnp.argmax(color_hint)]}"
                    + f"{'' if ~rank_hint.any() else jnp.argmax(rank_hint)+1}"
                )

                color_knowledge = knowledge[card_idx].any(axis=1)
                rank_knowledge = knowledge[card_idx].any(axis=0)
                color_knowledge_str = "".join(
                    c
                    for c, bool_idx in zip(self.color_map, color_knowledge)
                    if bool_idx
                )
                rank_knowledge_str = "".join(
                    str(r) for r in jnp.where(rank_knowledge)[0] + 1
                )
                card_knowledge = color_knowledge_str + rank_knowledge_str
                card_knowledge_str = f"Hints: {card_hint}, Possible: {card_knowledge}"

                if belief is not None:
                    card_belief = belief[card_idx]
                    best_belief_idx = jnp.argsort(-card_belief)[:best_belief]
                    card_belief = " ".join(
                        f"{color}{rank+1}: {card_belief[i]:.3f}"
                        for i, (color, rank) in enumerate(
                            itertools.product(self.color_map, range(self.num_ranks))
                        ) if i in best_belief_idx
                    )
                    card_knowledge_str += f", Belief: [{card_belief}]"

                return card_knowledge_str

            actor_hand_str = [
                f"{i} {'' if mask_hand else f'Card: {self.card_to_string(actor_hand[card_idx])}, ' }{get_card_knowledge_str(card_idx)}"
                for i, card_idx in enumerate(range(self.hand_size))
            ]

            return actor_hand_str

        keep_only_last_one = lambda x: jnp.where(
            jnp.arange(x.size) < (x.size - 1 - jnp.argmax(jnp.flip(x))),  # last argmax
            0,
            x,
        )
        fireworks = jax.vmap(keep_only_last_one)(new_state.fireworks)
        fireworks_cards = [
            jnp.zeros((self.num_colors, self.num_ranks)).at[i].set(fireworks[i])
            for i in range(self.num_colors)
        ]

        # board features
        board_info = {
            "turn": new_state.turn,
            "score": new_state.score,
            "information available": int(new_state.info_tokens.sum()),
            "lives available": int(new_state.life_tokens.sum()),
            "deck remaining cards": int(new_state.deck.sum()),
            "discards": " ".join(self.card_to_string(card) for card in new_state.discard_pile),
            "fireworks": " ".join(self.card_to_string(card) for card in fireworks_cards),
        }

        for i, (k, v) in enumerate(board_info.items()):
            output += f"{k.capitalize()}: {v}\n"
            if i == 0:
                output += "\n"

        # hands features
        current_player = jnp.nonzero(new_state.cur_player_idx, size=1)[0][0]
        belief = self.get_v0_belief_feats(current_player, new_state)
        belief = belief.reshape(self.num_agents, self.hand_size, -1)[
            :, :, : self.num_colors * self.num_ranks
        ]  # remove hint features
        belief = jnp.roll(belief, current_player, axis=0)  # reset absolute order
        for aidx in range(self.num_agents):
            output += ("Your Hand:" if aidx == current_player else "Other Hand:") + "\n"
            for card_str in get_actor_hand_str(
                aidx, mask_hand=aidx == current_player, belief=belief[aidx] if include_belief else None
            ):
                output += card_str + "\n"


        # last action feature
        if old_state is None:
            old_state = new_state
        last_action_feats = self.get_last_action_feats_(current_player, old_state, new_state, action)

        move_type = self.action_encoding[int(action)]
        output += f"Last action: {move_type}\n"

        if move_type[0] == 'H': # hint move
            reveal_outcome = np.where(last_action_feats['reveal_outcome'])[0]
            output += f"Cards afected: {reveal_outcome}\n"
        elif move_type[0] in ['D','P']:
            card_played = self.card_to_string(last_action_feats['played_discarded_card'].reshape(self.num_colors, self.num_ranks))
            output += f"Card Played: {card_played}\n"
        if move_type[0] == 'P':
            card_scored = last_action_feats['card_played_score'][0]
            card_added_info = last_action_feats['added_info_tokens'][0]
            output += f"Scored: {card_scored}\n"
            output += f"Added Info: {card_added_info}\n"

        # available actions
        legal_moves = self.get_legal_moves(new_state)[self.agents[current_player]]
        legal_actions = [self.action_encoding[int(a)] for a in np.where(legal_moves)[0]]
        output += f"Legal Actions: {legal_actions}\n"

        return output