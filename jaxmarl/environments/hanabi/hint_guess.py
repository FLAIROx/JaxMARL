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
        self.card_feature_space = jnp.array(list(product(*[np.arange(n_c) for n_c in self.num_classes_per_feature])))

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

    @partial(jax.jit, static_argnums=[0, 2, 3])
    def reset_for_eval(self, rng, reset_mode="exact_match", replace=True):
        
        def p_exact_match(masks):
            target_mask, non_target_mask, _, _, _ = masks
            p_hint = target_mask/jnp.sum(target_mask)
            p_hinter_and_guesser_rest_of_hand = non_target_mask/jnp.sum(non_target_mask)
            return p_hint, p_hinter_and_guesser_rest_of_hand
        
        def p_similarity_match(masks):
            _, _, random_similar_feature_exclude_target_mask, _, non_similar_feature_mask = masks
            hint_p = random_similar_feature_exclude_target_mask/jnp.sum(random_similar_feature_exclude_target_mask)
            p_hinter_and_guesser_rest_of_hand = non_similar_feature_mask/jnp.sum(non_similar_feature_mask)
            return hint_p, p_hinter_and_guesser_rest_of_hand
        
        def p_mutual_exclusive(masks):
            _, _, _, _, non_similar_feature_mask = masks
            p_non_sim = non_similar_feature_mask/jnp.sum(non_similar_feature_mask)
            hint_flat_id = jax.random.choice(hint_rng, 
                                             card_space, 
                                             shape=(1,),
                                             p=p_non_sim)
            hint_mask = jax.nn.one_hot(x=hint_flat_id, num_classes=self.num_cards).flatten() # note this is also p_hint, as the chosen card has p=1
            hinter_and_guesser_rest_of_hand_mask = jnp.logical_and(non_similar_feature_mask, jnp.logical_not(hint_mask))
            p_hinter_and_guesser_rest_of_hand = hinter_and_guesser_rest_of_hand_mask/jnp.sum(hinter_and_guesser_rest_of_hand_mask)
            return hint_mask, p_hinter_and_guesser_rest_of_hand
        
        def p_mutual_exclusice_similarity(masks):
            _, _, _, similar_feature_exclude_target_mask, non_similar_feature_mask = masks
            p_hint = non_similar_feature_mask/jnp.sum(non_similar_feature_mask)
            p_hinter_and_guesser_rest_of_hand = similar_feature_exclude_target_mask/jnp.sum(similar_feature_exclude_target_mask)
            print(similar_feature_exclude_target_mask, p_hinter_and_guesser_rest_of_hand)
            return p_hint, p_hinter_and_guesser_rest_of_hand
                
        def shuffle_and_index(rng, players_hands):
            def set_single_hand(hand, index):
                empty_hands = jnp.zeros(5, dtype=jnp.int32)
                return empty_hands.at[index].set(hand)
            """
            generates a permutation mapping for the hands of the players such that the target_card and hint_card are tractable after the permutation
            returns permuted hands, hint_card_index and target_card_index in the permuted hands of hinter and guesser
            """
            rngs = jax.random.split(rng, 2)
            permutation_index = jax.vmap(jax.random.permutation, in_axes=(0, None))(rngs, 5)
            permuted_hands = jax.vmap(set_single_hand, in_axes=(0, 0))(players_hands, permutation_index)
            return permuted_hands, permutation_index[0, 0], permutation_index[1, 0]

        target_rng, hint_rng, hinter_hand_rngs, guesser_hand_rngs = jax.random.split(rng, 4)
        
        # target randomisation
        target_flat_id = jax.random.choice(target_rng, self.num_cards)
        target_multi_id = jnp.array(jnp.unravel_index(target_flat_id, self.num_classes_per_feature))
        
        #copy card space to ensure env is not modified
        card_space = jnp.arange(self.num_cards)
        card_feature_space = self.card_feature_space

        # generate mask for exact match and non_exact match
        target_mask = jnp.where(target_flat_id 
                                == card_space, 
                                1, 
                                0).flatten()
        non_target_mask = 1 - target_mask

        # generate mask for similar cards for a randomly selected feature
        random_feature_of_interest = jax.random.choice(hint_rng, self.num_features)
        random_similar_feature_mask = jnp.where(card_feature_space[:, random_feature_of_interest] 
                                                == target_multi_id[random_feature_of_interest],
                                                1, 
                                                0).flatten()
        random_similar_feature_exclude_target_mask = non_target_mask * random_similar_feature_mask
        
        # generate mask for all non-similar cards for all features
        similar_feature_mask = jnp.zeros(self.num_cards)
        non_similar_feature_mask = jnp.ones(self.num_cards)
        for feature_dim in range(self.num_features):
            # + is logical or operation, * is logical and operation
            similar_feature_mask = similar_feature_mask + jnp.where(card_feature_space[:, feature_dim] 
                                                                    == target_multi_id[feature_dim],
                                                                    1, 
                                                                    0).flatten()
            non_similar_feature_mask = non_similar_feature_mask * jnp.where(card_feature_space[:, feature_dim] 
                                                                            != target_multi_id[feature_dim],
                                                                            1, 
                                                                            0).flatten()
        similar_feature_mask_exculde_target = similar_feature_mask * non_target_mask
            
        masks = (target_mask, non_target_mask, random_similar_feature_exclude_target_mask, similar_feature_mask_exculde_target, non_similar_feature_mask)
        p_reset_modes = {
            "exact_match": p_exact_match,
            "similarity_match": p_similarity_match,
            "mutual_exclusive": p_mutual_exclusive,
            "mutual_exclusive_similarity": p_mutual_exclusice_similarity,
        }
        if reset_mode in p_reset_modes:
            p_hint, p_other = p_reset_modes[reset_mode](masks)
        else:
            raise ValueError("reset_mode is not supported")

        hinter_flat_id = jax.random.choice(hint_rng, 
                                               card_space, 
                                               shape=(1,),
                                               replace=replace,
                                               p=p_hint)

        hinter_flat_rest_of_hand = jax.random.choice(hinter_hand_rngs, 
                                                         card_space, 
                                                         shape=(self.hand_size-1,),
                                                         replace=replace,
                                                         p=p_other)
        if reset_mode == "mutual_exclusive" or reset_mode == "mutual_exclusive_similarity":
            guesser_flat_rest_of_hand = hinter_flat_rest_of_hand
        else:
            guesser_flat_rest_of_hand = jax.random.choice(guesser_hand_rngs, 
                                                            card_space, 
                                                            shape=(self.hand_size-1,),
                                                            replace=replace,
                                                            p=p_other)
        
        hinter_hand = jnp.append(hinter_flat_id, hinter_flat_rest_of_hand)
        guesser_hand = jnp.append(target_flat_id, guesser_flat_rest_of_hand)

        player_hands = jnp.stack((hinter_hand, guesser_hand))
        rngs = jnp.stack((hinter_hand_rngs, guesser_hand_rngs))
        permuted_hands, hint_indices, target_indices = jax.vmap(shuffle_and_index, in_axes=(0, None), out_axes=(0, 0, 0))(rngs, player_hands)
        state = State(
            player_hands=permuted_hands, target=target_flat_id, hint=-1, guess=-1, turn=0
        )

        return jax.lax.stop_gradient(self.get_obs(state)), state, hint_indices, target_indices

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
    # jax.config.update("jax_disable_jit", True)
    env = HintGuessGame()
    rng = jax.random.PRNGKey(10)
    # reset_modes: exact_match, similarity_match, mutual_exclusive, mutual_exclusive_similarity
    _, state, hints, targets = env.reset_for_eval(rng, reset_mode="similarity_match", replace=True)
    print(jnp.arange(9).reshape(3, 3))
    print(state)
    print(hints)
    print(targets)
