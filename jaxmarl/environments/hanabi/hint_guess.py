import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from itertools import product
import chex
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from gymnax.environments.spaces import Discrete
import copy


@struct.dataclass
class State:
    player_hands: chex.Array
    target: chex.Array
    hint: chex.Array
    guess: chex.Array
    turn: int


class HintGuessGame(MultiAgentEnv):

    def __init__(
        self,
        num_agents=2,
        num_features=2,
        num_classes_per_feature=[3, 3],
        hand_size=5,
        card_encoding="onehot",
        matrix_obs=False,
        agents=None,
        action_spaces=None,
        observation_spaces=None,
    ):
        super().__init__(num_agents)

        assert num_agents == 2, "Environment defined only for 2 agents"

        if agents is None:
            self.agents = ["hinter", "guesser"]
        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)

        self.hand_size = hand_size
        self.num_features = num_features
        self.num_classes_per_feature = num_classes_per_feature
        self.num_cards = np.prod(self.num_classes_per_feature)
        self.matrix_obs = matrix_obs
        self.feature_tree = [np.arange(n_c) for n_c in num_classes_per_feature]

        # generate the deck of one-hot encoded cards
        if card_encoding == "onehot":
            self.encoding_dim = np.sum(num_classes_per_feature)
            self.card_encodings = self.get_onehot_encodings()
        else:
            raise NotImplementedError("Available encodings are: 'onehot'")

        self.obs_size = (
            (self.num_agents * (self.hand_size)+1) * (self.encoding_dim + 2)
        )  # +1 one_hot for if it's hinter or guesser card + 1 one_hot if the obs belong to hinter or guesser
        self.action_dim = (
            np.prod(self.num_classes_per_feature) + 1
        )  # hint-guess one card of the game + nothing
        if action_spaces is None:
            self.action_spaces = {i: Discrete(self.action_dim) for i in self.agents}
        if observation_spaces is None:
            self.observation_spaces = {i: Discrete(self.obs_size) for i in self.agents}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, rng):
        rng_hands, rng_target = jax.random.split(rng)
        player_hands = jax.random.choice(
            rng_hands,
            jnp.arange(self.num_cards),
            shape=(
                self.num_agents,
                self.hand_size,
            ),
        )
        
        # every agent sees the hands in different order
        _rngs = jax.random.split(rng_hands, self.num_agents)
        permuted_hands = jax.vmap(
            lambda rng: jax.random.permutation(rng, player_hands, axis=1)
        )(_rngs)

        # choose one card of the second agent
        target = jax.random.choice(rng_target, player_hands[1])

        state = State(
            player_hands=permuted_hands, target=target, hint=-1, guess=-1, turn=0
        )
        return jax.lax.stop_gradient(self.get_obs(state)), state

    @partial(jax.jit, static_argnums=[0, 2])
    def reset_for_eval(self, rng, reset_mode="exact_match"):
        
        def exact_match(card_multi_set):
            card_flat_set = card_multi_set.flatten()
            hint_flat_id = target_flat_id
            hinter_and_guesser_flat_hand_set = jnp.delete(card_flat_set, target_flat_id, assume_unique_indices=True)
            hinter_flat_rest_of_hand = jax.random.choice(hinter_hand_rngs, 
                                                         hinter_and_guesser_flat_hand_set, 
                                                         shape=(self.hand_size-1,))
            guesser_flat_rest_of_hand = jax.random.choice(guesser_hand_rngs, 
                                                          hinter_and_guesser_flat_hand_set, 
                                                          shape=(self.hand_size-1,))
            return hint_flat_id, hinter_flat_rest_of_hand, guesser_flat_rest_of_hand
        
        def similarity_match(card_multi_set):
            feature_of_interest = jax.random.choice(hint_rng, self.num_features)
            target_index_of_interest = target_multi_id[feature_of_interest]
            # note this hint_set also include target card, need to be removed
            # print(feature_of_interest, target_index_of_interest)
            hint_set = jax.lax.dynamic_index_in_dim(card_multi_set, 
                                                    target_index_of_interest, 
                                                    feature_of_interest, 
                                                    keepdims=False)
            # find the target card id in hint set after slicing from the feature of interest
            target_id = jnp.concatenate((target_multi_id[:feature_of_interest], target_multi_id[feature_of_interest+1:]))
            hint_flat_set = jnp.delete(hint_set, target_id, assume_unique_indices=True).flatten()
            
            non_similar_hand_set = copy.deepcopy(card_multi_set)
            for feature_dim in range(self.num_features):
                non_similar_hand_set = jnp.delete(non_similar_hand_set, 
                                                  target_multi_id[feature_dim], 
                                                  axis=feature_dim,
                                                  assume_unique_indices=True)
                
            non_similar_flat_hand_set = non_similar_hand_set.flatten()
            hinter_flat_id = jax.random.choice(hint_rng, 
                                               hint_flat_set, 
                                               shape=(1,))
            hinter_flat_rest_of_hand = jax.random.choice(hinter_hand_rngs, 
                                                         non_similar_flat_hand_set, 
                                                         shape=(self.hand_size-1,))
            guesser_flat_rest_of_hand = jax.random.choice(guesser_hand_rngs, 
                                                          non_similar_flat_hand_set, 
                                                          shape=(self.hand_size-1,))
            return hinter_flat_id, hinter_flat_rest_of_hand, guesser_flat_rest_of_hand
        
        def mutual_exclusive(card_multi_set):
            non_similar_hand_set = copy.deepcopy(card_multi_set)
            for feature_dim in range(self.num_features):
                non_similar_hand_set = jnp.delete(non_similar_hand_set, 
                                                  target_multi_id[feature_dim], 
                                                  axis=feature_dim,
                                                  assume_unique_indices=True)
            
            non_similar_flat_hand_set = non_similar_hand_set.flatten()
            hint_flat_id = jax.random.choice(hint_rng, 
                                             non_similar_flat_hand_set, 
                                             shape=(1,))
            hinter_and_guesser_flat_rest_of_hand_set = jnp.delete(non_similar_flat_hand_set, hint_flat_id, assume_unique_indices=True)
            # note the rest of hand of both players are the same, so use either of the rngs
            hinter_and_guesser_flat_rest_of_hand = jax.random.choice(hinter_hand_rngs, 
                                                                     hinter_and_guesser_flat_rest_of_hand_set, 
                                                                     shape=(self.hand_size-1,))
            return hint_flat_id, hinter_and_guesser_flat_rest_of_hand, hinter_and_guesser_flat_rest_of_hand
        
        def mutual_exclusice_similarity(card_multi_set):
            # the target will be included by the first slice, thus need to be removed
            similar_cards_of_the_first_feature = jax.lax.dynamic_index_in_dim(card_multi_set, 
                                                                                    target_multi_id[0], 
                                                                                    0, 
                                                                                    keepdims=False)
            target_id = target_multi_id[1:]
            hinter_and_guesser_flat_rest_of_hand_set = jnp.delete(similar_cards_of_the_first_feature, target_id, assume_unique_indices=True).flatten()
            
            # later slices does include the target card
            for feature_dim in range(1, self.num_features):
                similar_cards = jax.lax.dynamic_index_in_dim(card_multi_set, 
                                                             target_multi_id[feature_dim], 
                                                             feature_dim, 
                                                             keepdims=False)
                hinter_and_guesser_flat_rest_of_hand_set = jnp.append(hinter_and_guesser_flat_rest_of_hand_set, 
                                                                      similar_cards.flatten())
                card_multi_set = jnp.delete(card_multi_set, 
                                            target_multi_id[feature_dim], 
                                            axis=feature_dim,
                                            assume_unique_indices=True)
                
            hint_flat_set = card_multi_set.flatten()
            hint_flat_id = jax.random.choice(hint_rng, hint_flat_set, shape=(1,))
            # note the rest of hand of both players are the same, so use either of the rngs
            hinter_and_guesser_flat_rest_of_hand = jax.random.choice(hinter_hand_rngs, 
                                                                     hinter_and_guesser_flat_rest_of_hand_set, 
                                                                     shape=(self.hand_size-1,))
            return hint_flat_id, hinter_and_guesser_flat_rest_of_hand, hinter_and_guesser_flat_rest_of_hand
                
        def shuffle_and_index(rng, players_hands):
            def set_single_hand(hand, index):
                empty_hands = jnp.zeros(5, dtype=jnp.int32)
                return empty_hands.at[index].set(hand)
            """
            generates a permutation mapping for the hands of the players such that the target_card and hint_card are tractable after the permutation
            returns permuted hands, hint_card_index and target_card_index in the permuted hands
            """
            rngs = jax.random.split(rng, 2)
            permutation_index = jax.vmap(jax.random.permutation, in_axes=(0, None))(rngs, 5)
            permuted_hands = jax.vmap(set_single_hand, in_axes=(0, 0))(players_hands, permutation_index)
            return permuted_hands, permutation_index[0, 0], permutation_index[0, 1]

        target_rng, hint_rng, hinter_hand_rngs, guesser_hand_rngs = jax.random.split(rng, 4)
        
        # constants
        target_flat_id = jax.random.choice(target_rng, self.num_cards)
        target_multi_id = jnp.array(jnp.unravel_index(target_flat_id, self.num_classes_per_feature))
        card_multi_set = jnp.arange(self.num_cards).reshape(self.num_classes_per_feature)
        

        if reset_mode == "exact_match":
            hint_flat_id, hinter_flat_rest_of_hand, guesser_flat_rest_of_hand = exact_match(card_multi_set)
        elif reset_mode == "similarity_match":
            hint_flat_id, hinter_flat_rest_of_hand, guesser_flat_rest_of_hand = similarity_match(card_multi_set)
        elif reset_mode == "mutual_exclusive":
            hint_flat_id, hinter_flat_rest_of_hand, guesser_flat_rest_of_hand = mutual_exclusive(card_multi_set)
        elif reset_mode == "mutual_exclusive_similarity":
            hint_flat_id, hinter_flat_rest_of_hand, guesser_flat_rest_of_hand = mutual_exclusice_similarity(card_multi_set)
        else:
            raise ValueError("reset_mode is not supported")
        
        hinter_hand = jnp.append(hint_flat_id, hinter_flat_rest_of_hand)
        guesser_hand = jnp.append(target_flat_id, guesser_flat_rest_of_hand)

        player_hands = jnp.stack((hinter_hand, guesser_hand))
        print(player_hands.shape)
        rngs = jnp.stack((hinter_hand_rngs, guesser_hand_rngs))
        permuted_hands, hints, targets = jax.vmap(shuffle_and_index, in_axes=(0, None), out_axes=(0, 0, 0))(rngs, player_hands)
        state = State(
            player_hands=permuted_hands, target=target_flat_id, hint=-1, guess=-1, turn=0
        )

        return jax.lax.stop_gradient(self.get_obs(state)), state, hints, targets


        

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, rng, state, actions):

        def step_hint(state, actions):
            action = actions["hinter"]
            state = state.replace(
                hint=action,
                turn=1,
            )
            reward = 0
            done = False
            return state, reward, done

        def step_guess(state, actions):
            action = actions["guesser"]
            state = state.replace(
                guess=action,
                turn=2,
            )
            reward = (action == state.target).astype(int)
            done = True
            return state, reward, done

        state, reward, done = jax.lax.cond(
            state.turn == 0, step_hint, step_guess, state, actions
        )

        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        rewards = {agent: reward for agent in self.agents}
        info = {}

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            rewards,
            dones,
            info,
        )

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state):
        """Obs is [one_hot(feat1),one_hot(feat2)...,agent_id_hand,agent_id_obs] per each card in all the hands."""

        target_hint_card = jnp.where(
            state.turn == 0,  # is hint step?
            self.card_encodings[state.target],  # target if is hint step
            self.card_encodings[state.hint],  # hint otherwise
        )
        # add the guess hint card adding as agent_id the current turn (i.e. 00 for hinting turn, 11 for guessing turn)
        target_hint_card = jnp.append(
            target_hint_card, jnp.array([state.turn, state.turn])
        )

        def _get_obs(aidx):
            hands = state.player_hands[aidx]
            card_encodings = self.card_encodings[hands]
            # extra one-hot features: agent_id of relative to the hand and to the obs
            agent_id = jnp.concatenate(
                [jnp.full(self.hand_size, i) for i in jnp.arange(self.num_agents)]
            ).reshape(self.num_agents, self.hand_size, -1)
            is_guesser = jnp.full((self.num_agents, self.hand_size, 1), aidx)
            card_encodings = jnp.concatenate(
                (card_encodings, agent_id, is_guesser), axis=-1
            )
            target_hint_card_masked = jnp.where(
                (aidx == 1)
                & (state.turn == 0),  # mask the target card for the guesser at turn 0
                jnp.zeros_like(target_hint_card),
                target_hint_card,
            )
            card_encodings = jnp.concatenate(
                (jnp.vstack(card_encodings), target_hint_card_masked[np.newaxis])
            )
            # flatten if the obs is not requested as matrix
            card_encodings = (
                card_encodings.ravel() if not self.matrix_obs else card_encodings
            )
            return card_encodings

        obs = jax.vmap(_get_obs)(jnp.arange(self.num_agents))

        return {"hinter": obs[0], "guesser": obs[1]}

    @partial(jax.jit, static_argnums=[0])
    def get_legal_moves(self, state):
        """Legal moves in first step are the features-combinations cards for the hinter, nope for the guesser. Symmetric for second round"""

        actions = jnp.zeros(self.action_dim)
        nope_move = (
            jnp.zeros(self.action_dim).at[-1].set(1)
        )  # only "do nothing" is valid

        hinter_legal_moves = jnp.where(
            state.turn
            == 0,  # for hint turn, available actions are the card types of the hinter
            actions.at[state.player_hands[0][0]].set(1),
            nope_move,  # do nothing in the guessing turn
        )

        guesser_legal_moves = jnp.where(
            state.turn == 1,
            actions.at[state.player_hands[0][1]].set(
                1
            ),  # for guess turn, available actions are the card types of the guesser
            nope_move,  # do nothing in the hinting turn
        )

        return {"hinter": hinter_legal_moves, "guesser": guesser_legal_moves}

    def get_onehot_encodings(self):
        """Concatenation of one_hots for every card feature, f.i., 2feats with 2classes ->[1,0,0,1,0,0,0]"""
        encodings = [
            jax.nn.one_hot(jnp.arange(n_c), n_c) for n_c in self.num_classes_per_feature
        ]
        encodings = jnp.array(
            [jnp.concatenate(combination) for combination in list(product(*encodings))]
        )
        return encodings

    
if __name__ == "__main__":
    jax.config.update("jax_disable_jit", True)
    env = HintGuessGame()
    rng = jax.random.PRNGKey(0)
    _, state, _, _ = env.reset_for_eval(rng, reset_mode="exact_match")
    print(state)
