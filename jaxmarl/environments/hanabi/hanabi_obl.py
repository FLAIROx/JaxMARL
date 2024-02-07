import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct
from typing import Tuple, Dict
from functools import partial
from gymnax.environments.spaces import Discrete
from .hanabi import HanabiGame, State


class HanabiOBL(HanabiGame):

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
        obs_size=None,
        num_moves=None,
        debug=False
    ):
        
        super().__init__(
            num_agents=num_agents,
            num_colors=num_colors,
            num_ranks=num_ranks,
            hand_size=hand_size,
            max_info_tokens=max_info_tokens,
            max_life_tokens=max_life_tokens,
            num_cards_of_rank=num_cards_of_rank,
            agents=agents,
            action_spaces=action_spaces,
            observation_spaces=observation_spaces,
            obs_size=obs_size,
            num_moves=num_moves
        )

        self.debug = debug

        # useful ranges to know the type of the action
        self.discard_action_range = jnp.arange(0, self.hand_size)
        self.play_action_range    = jnp.arange(self.hand_size, 2*self.hand_size)
        self.color_action_range   = jnp.arange(2*self.hand_size, 2*self.hand_size+self.num_colors)
        self.rank_action_range    = jnp.arange(2*self.hand_size+self.num_colors, 2*self.hand_size+self.num_colors+self.num_ranks)

        # number of features
        self.hands_n_feats = (
            self.num_agents*self.hand_size*self.num_colors*self.num_ranks+ # hands of all the agents
            self.num_agents # agents' missing cards
        )
        self.board_n_feats = (
            (self.deck_size-self.num_agents*self.hand_size)+ # deck-initial cards, thermometer
            self.num_colors*self.num_ranks+ # fireworks, OH
            self.max_info_tokens+#info tokens, OH
            self.max_life_tokens#life tokens, OH
        )
        self.discards_n_feats = self.num_colors*self.num_cards_of_rank.sum()
        self.last_action_n_feats = (
            self.num_agents+ # acting player index
            4+ # move type
            self.num_agents+ # target player index
            self.num_colors+ # color revealed
            self.num_ranks+ # rank revalued
            self.hand_size+ # reveal outcome
            self.hand_size+ # position played/discared
            self.num_colors*self.num_ranks+ # card played/discarded
            1 + # card played score
            1 # card played added info toke
        )
        self.v0_belief_n_feats = (
            self.num_agents*self.hand_size* # feats for each card, mine and other players
            (self.num_colors*self.num_ranks+self.num_colors+self.num_ranks) # 35 feats per card (25deductions+10hints)
        )
        self.obs_size = (
            self.hands_n_feats+
            self.board_n_feats+
            self.discards_n_feats+
            self.last_action_n_feats+
            self.v0_belief_n_feats
        )
        self.observation_spaces = {i: Discrete(self.obs_size) for i in self.agents}

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: Dict,
                 ) -> Tuple[chex.Array, State, Dict, Dict, Dict]:


        # get actions as array
        actions = jnp.array([actions[i] for i in self.agents])
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        action = actions.at[aidx].get()
        action = action[0] - 1 # TODO: remove this indexing (need to change ppo for that)

        # execute the current player's action and its consequences
        old_state = state
        new_state, reward = self.step_agent(key, state, aidx, action)

        done = self.terminal(new_state)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        rewards = {agent: reward for agent in self.agents}
        rewards["__all__"] = reward

        info = {}

        return (
            lax.stop_gradient(self.get_obs(old_state, new_state, action)),
            lax.stop_gradient(new_state),
            rewards,
            dones,
            info
        )
    
    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, new_state:State, old_state:State=None, action:chex.Array=0) -> Dict:

        # TODO: remove parameters placeholders (now for the get_obs in the original reset function not passing the additional params)
        if old_state is None:
            old_state = new_state
    
        # no agent-specific obs
        board_fats = self.get_board_fats(new_state)
        discard_feats = self._binarize_discard_pile(new_state.discard_pile)

        def _observe(aidx:int):

            # HANDS FEATURES: my masked hand, other agents hands, missing cards per agent
            hands_from_self = jnp.roll(new_state.player_hands, -aidx, axis=0) # make the current player hand first
            hands = hands_from_self.at[0].set(0).ravel() # mask the own hand and ravel
            missing_cards = ~hands_from_self.any(axis=(1,2,3)) # check if some player is missing a card
            hands_feats = jnp.concatenate((hands,missing_cards))

            # LAST ACTION FEATS
            last_action_feats = jnp.where( 
                new_state.turn==0, # no features if first turn because no actions were made
                jnp.zeros(self.last_action_n_feats),
                self.get_last_action_feats(aidx, old_state, new_state, action)
            )

            # BELIEF FEATS
            belief_v0_feats = self.get_v0_belief_feats(aidx, new_state)

            return jnp.concatenate((
                hands_feats,
                board_fats,
                discard_feats,
                last_action_feats,
                belief_v0_feats,
            ))
        
        obs = jax.vmap(_observe)(self.agent_range)

        return {a: obs[i] for i, a in enumerate(self.agents)}


    @partial(jax.jit, static_argnums=[0])
    def get_last_action_feats(self, aidx:int, old_state: State, new_state:State, action:chex.Array):

        acting_player_index = old_state.cur_player_idx # absolute OH index
        target_player_index = new_state.cur_player_idx # absolute OH index
        acting_player_relative_index = jnp.roll(acting_player_index,-aidx) # relative OH index 
        target_player_relative_index = jnp.roll(target_player_index,-aidx) # relative OH index 

        # in obl the encoding order here is: play, discard, reveal_c, reveal_r
        move_type = jnp.where( # hard encoded but hey ho let's go
            (action>=0)&(action<5), # discard 
            jnp.array([0,1,0,0]),
            jnp.where(
                (action>=5)&(action<10), # play
                jnp.array([1,0,0,0]),
                jnp.where(
                    (action>=10)&(action<15), # reveal_c
                    jnp.array([0,0,1,0]),
                    jnp.where(
                        (action>=15)&(action<20), # reveal_r
                        jnp.array([0,0,0,1]),
                        jnp.array([0,0,0,0]), # invalid
                    )
                )
            )
        )
        
        # get the hand of the target player
        target_hand = new_state.player_hands[jnp.nonzero(target_player_index, size=1)[0][0]]
        
        color_revealed = jnp.where( # which color was revealed by action (oh)?
            action == self.color_action_range,
            1.,
            jnp.zeros(self.color_action_range.size)
        )
        
        rank_revealed = jnp.where( # which rank was revealed by action (oh)?
            action == self.rank_action_range,
            1.,
            jnp.zeros(self.rank_action_range.size)
        )
        
        # cards that have the color that was revealed
        color_revealed_cards = jnp.where(
            (target_hand.sum(axis=(2)) == color_revealed).all(axis=1), # color of the card==color reveled
            1,
            0
        )
        
        # cards that have the color that was revealed
        rank_revealed_cards = jnp.where(
            (target_hand.sum(axis=(1)) == rank_revealed).all(axis=1), # color of the card==color reveled
            1,
            0
        )
        
        # cards that are caught by the hint
        reveal_outcome = color_revealed_cards|rank_revealed_cards
        
        # card that was played-discarded 
        pos_played_discarded = jnp.where(
            action < 2*self.hand_size,
            jnp.arange(self.hand_size) == action%self.hand_size,
            jnp.zeros(self.hand_size),
        )
        actor_hand_before = old_state.player_hands[jnp.nonzero(acting_player_index, size=1)[0][0]]

        played_discarded_card = jnp.where(
            pos_played_discarded.any(),
            actor_hand_before[jnp.nonzero(pos_played_discarded, size=1)[0][0]].ravel(),
            jnp.zeros(self.num_colors*self.num_ranks) # all zeros if no card was played
        )

        # effect of playing the card in the game
        card_played_score = (new_state.fireworks.sum(axis=(0,1)) - old_state.fireworks.sum(axis=(0,1)))!=0
        added_info_tokens = new_state.info_tokens.sum() > old_state.info_tokens.sum()

        if self.debug:
            print({
                'acting_player_relative_index':acting_player_relative_index,
                'move_type':move_type,
                'target_player_relative_index':target_player_relative_index,
                'color_revealed':color_revealed,
                'rank_revealed':rank_revealed,
                'reveal_outcome':reveal_outcome,
                'pos_played_discarded':pos_played_discarded,
                'played_discarded_card':played_discarded_card.reshape(self.num_colors, self.num_ranks),
                'card_played_score':card_played_score[np.newaxis], # scalar
                'added_info_tokens':added_info_tokens[np.newaxis], # scalar 
            })
        
        last_action = jnp.concatenate((
            acting_player_relative_index,
            move_type,
            target_player_relative_index,
            color_revealed,
            rank_revealed,
            reveal_outcome,
            pos_played_discarded,
            played_discarded_card,
            card_played_score[np.newaxis], # scalar
            added_info_tokens[np.newaxis], # scalar 
        ))

        return last_action
    
    @partial(jax.jit, static_argnums=[0])
    def get_board_fats(self, state:State):
        
        # by default the fireworks are incremental, i.e. [1,1,0,0,0] one and two are in the board
        # must be OH of only the highest rank, i.e. [0,1,0,0,0]
        keep_only_last_one = lambda x: jnp.where(
            jnp.arange(x.size)<(x.size - 1 - jnp.argmax(jnp.flip(x))), # last argmax
            0,
            x
        )

        fireworks = jax.vmap(keep_only_last_one)(state.fireworks)
        deck = jnp.any(jnp.any(state.deck, axis=1), axis=1).astype(int)
        deck = deck[-1:self.num_agents*self.hand_size-1:-1] # avoid the first cards at beginning of episode and reset the order 
        board_feats = jnp.concatenate((deck, fireworks.ravel(), state.life_tokens, state.info_tokens))
        
        return board_feats
    
    @partial(jax.jit, static_argnums=[0])
    def get_v0_belief_feats(self, aidx:int, state:State):

        def belief_per_hand(hand, knowledge, color_hint, rank_hint):
            count = (
                hand.sum(axis=0).ravel()+
                state.deck.sum(axis=0).ravel()
            ) # count of the remaining cards
            normalized_knowledge = knowledge*count
            normalized_knowledge /= normalized_knowledge.sum(axis=1)[:,np.newaxis]
            return jnp.concatenate((
                    normalized_knowledge,
                    color_hint,
                    rank_hint
                ), axis=-1
            ).ravel()
        
        # compute my belief and the beliefs of other players, starting from self cards 
        rel_pos = lambda x: jnp.roll(x, -aidx, axis=0)
        belief = jax.vmap(belief_per_hand)(
            rel_pos(state.player_hands),
            rel_pos(state.card_knowledge),
            rel_pos(state.colors_revealed),
            rel_pos(state.ranks_revealed),
        ).ravel()

        return belief

    
    @partial(jax.jit, static_argnums=[0])
    def _binarize_discard_pile(self, discard_pile:chex.Array):

        def binarize_ranks(n_ranks):
            tree = jax.tree_util.tree_map(
                lambda n_rank_present, max_ranks: jnp.where(
                    jnp.arange(max_ranks)>=n_rank_present, jnp.zeros(max_ranks), jnp.ones(max_ranks)
                ),
                [x for x in n_ranks],
                [x for x in self.num_cards_of_rank]
            )
            return jnp.concatenate(tree)

        return jax.vmap(binarize_ranks)(discard_pile.sum(axis=0)).ravel()
    





