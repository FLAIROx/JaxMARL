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

        def get_safe_probability_from_mask(mask):
            def true_fn(mask):
                return jnp.full(self.num_cards, -1, dtype=jnp.float32)
            return jax.lax.cond(jnp.sum(mask) == 0, true_fn, lambda x: x/x.sum(), mask)

        def mask_selected_card(args):
            hand_masks, selected_cards = args
            masked_mask = jax.vmap(lambda mask, card: mask.at[card].set(0))(hand_masks, selected_cards)
            return masked_mask
        
        def get_random_pair(masks, rng, replace=False):
            """
            This function returns a random pair of cards based on masks such that there is no relation between the cards.
            """
            h_rng, g_rng = jax.random.split(rng, 2)
            task_specific_mask, h_hand_mask, g_hand_mask = masks
            h_hand_mask = task_specific_mask * h_hand_mask
            h_card = jax.random.choice(h_rng, self.num_cards, p=get_safe_probability_from_mask(h_hand_mask))
            g_hand_mask = task_specific_mask * g_hand_mask
            g_card = jax.random.choice(g_rng, self.num_cards, p=get_safe_probability_from_mask(g_hand_mask))
            selected_cards = jnp.array([h_card, g_card])
            hand_masks = jnp.stack([h_hand_mask, g_hand_mask], axis=0)
            updated_masks = jax.lax.cond(replace, 
                                      mask_selected_card, 
                                      lambda args: args[0], 
                                      (hand_masks, selected_cards))
            return updated_masks, selected_cards

        def get_identical_pair(masks, rng, replace=False):
            """
            This function returns a pair of cards that are identical in all the features.
            """
            task_specific_mask, h_hand_mask, g_hand_mask = masks
            both_hand_available = h_hand_mask * g_hand_mask
            card_mask = task_specific_mask * both_hand_available
            card_drawn = jax.random.choice(rng, jnp.arange(self.num_cards), p=get_safe_probability_from_mask(card_mask))
            selected_cards = jnp.array([card_drawn, card_drawn])
            hand_masks = jnp.stack([h_hand_mask, g_hand_mask], axis=0)
            updated_masks = jax.lax.cond(replace, 
                                      mask_selected_card, 
                                      lambda args: args[0], 
                                      (hand_masks, selected_cards))
            return updated_masks, selected_cards

        def get_similar_pair(masks, rng, replace=False):
            """
            This function returns a pair of cards that are identical in at least one feature but not the same card
            """
            f_rng, h_rng, g_rng = jax.random.split(rng, 3)
            task_specific_mask, h_hand_mask, g_hand_mask = masks
            h_card_mask = task_specific_mask * h_hand_mask
            # Robustness issue: need to solve the problem where there might not be a possible corresponding card for the other player
            # Solution: need to backtrack a set of possible hints from guesser's playable cards
            
            h_card = jax.random.choice(h_rng, self.num_cards, p=get_safe_probability_from_mask(h_card_mask))
            h_card_feature_vector = self.card_feature_space[h_card, :]
            interested_feature_class = jax.random.choice(f_rng, self.num_features)
            interested_feature_mask = jnp.where(self.card_feature_space[:, interested_feature_class] 
                                                == h_card_feature_vector[interested_feature_class],
                                                1, 
                                                0).flatten()
            non_h_cards = 1 - jax.nn.one_hot(h_card, self.num_cards)
            g_card_mask = task_specific_mask * g_hand_mask * interested_feature_mask * non_h_cards
            print(task_specific_mask, g_hand_mask, interested_feature_mask, non_h_cards)
            g_card = jax.random.choice(g_rng, self.num_cards, p=get_safe_probability_from_mask(g_card_mask))
            selected_cards = jnp.array([h_card, g_card])
            hand_masks = jnp.stack([h_hand_mask, g_hand_mask], axis=0)
            updated_masks = jax.lax.cond(replace, 
                                      mask_selected_card, 
                                      lambda args: args[0], 
                                      (hand_masks, selected_cards))
            return updated_masks, selected_cards
            
 
        def get_non_simillar_pair(masks, rng, replace=False):
            """
            This function returns a pair of cards that are different in all features.
            """
            def get_non_sim_feature_mask(card):
                def feature_level_non_similar_mask(label, arr):
                    return jnp.where(label != arr, 1, 0)
                card_feature_vector = self.card_feature_space[card, :]
                card_non_similar_masks = jax.vmap(feature_level_non_similar_mask, in_axes=(0, 1))(card_feature_vector, self.card_feature_space)
                return jnp.prod(card_non_similar_masks, axis=0)
            h_rng, g_rng = jax.random.split(rng, 2)
            task_specific_mask, h_hand_mask, g_hand_mask = masks
            h_card_mask = task_specific_mask * h_hand_mask
            h_card = jax.random.choice(h_rng, self.num_cards, p=get_safe_probability_from_mask(h_card_mask))
            h_non_sim_feature_mask = get_non_sim_feature_mask(h_card)
            g_card_mask = task_specific_mask * g_hand_mask * h_non_sim_feature_mask
            g_card = jax.random.choice(g_rng, self.num_cards, p=get_safe_probability_from_mask(g_card_mask))
            selected_cards = jnp.array([h_card, g_card])
            hand_masks = jnp.stack([h_hand_mask, g_hand_mask], axis=0)
            updated_masks = jax.lax.cond(replace, 
                                      mask_selected_card, 
                                      lambda args: args[0], 
                                      (hand_masks, selected_cards))
            return updated_masks, selected_cards
        
        def get_similarity_mask(card):
            def feature_level_similar_mask(label, arr):
                return jnp.where(label == arr, 1, 0)
            # card_similar_masks is an ndarray indicating the which cards are similar on which features of the target
            # it has shape feature x num_cards
            card_feature_vector = self.card_feature_space[card, :]
            card_similar_masks = jax.vmap(feature_level_similar_mask, in_axes=(0, 1))(card_feature_vector, self.card_feature_space)
            return card_similar_masks
            
        feature_masks = jnp.ones((self.num_features, self.num_cards), dtype=jnp.int32)
        h_hand_mask = jnp.ones(self.num_cards, dtype=jnp.int32)
        g_hand_mask = jnp.ones(self.num_cards, dtype=jnp.int32)
        _, rng, rng_hand, rng_view = jax.random.split(rng, 4)

        # when generating hands, the hint/target pair are first generated, then the rest of the hands are generated
        if reset_mode == "exact_match":
            hint_target_fn = get_identical_pair
            rest_of_hands_fn = get_random_pair
        elif reset_mode == "similarity_match":
            hint_target_fn = get_similar_pair
            rest_of_hands_fn = get_random_pair
        elif reset_mode == "mutual_exclusive":
            hint_target_fn = get_non_simillar_pair
            rest_of_hands_fn = get_identical_pair
        elif reset_mode == "mutual_exclusive_similarity":
            hint_target_fn = get_non_simillar_pair
            rest_of_hands_fn = get_similar_pair
        else:
            raise ValueError("Invalid reset mode")
        
        # during the target/hint generation phase, there is no restictions on what target can be, thus feature masks are all 1
        # replace is forced to be true to guarentee that hint/target are only used once from individual set
        hand_masks, selected_cards = hint_target_fn((jnp.ones(self.num_cards, dtype=jnp.int32), h_hand_mask, g_hand_mask), rng, replace=True)
        h_hand_mask, g_hand_mask = hand_masks[0, :], hand_masks[1, :]

        # set the target card in hinter's hand to 0 to avoid duplication if they are not the same
        h_hand_mask = h_hand_mask * g_hand_mask

        # similarly, set the hint card in guesser's hand to 0 to avoid duplication if they are not the same, e.g., if hint is 6, then 6 must not be an option for guesser's rest of the hand
        g_hand_mask = g_hand_mask * h_hand_mask


        hint, target = selected_cards[0], selected_cards[1]
        similarity_mask = get_similarity_mask(target)

        if reset_mode == "exact_match":
            # there is no other restriction on values of rest of the hands, apart from hand masks on hint/target
            task_specific_mask = jnp.ones(self.num_cards, dtype=jnp.int32)
        elif reset_mode == "similarity_match":
            # apart from hint/target, the rest of the hands pairs should also have no feature that is same as target
            task_specific_mask = 1 - jnp.any(similarity_mask, axis=0).astype(jnp.int32)
        elif reset_mode == "mutual_exclusive":
            # apart from hint/target, the rest of the hands pairs should also have no feature that is same as target
            task_specific_mask = 1 - jnp.any(similarity_mask, axis=0).astype(jnp.int32)
        elif reset_mode == "mutual_exclusive_similarity":
            # there are two possibilities here, we can set the rest of the hand to be "no feature idential", like the two previous cases
            # or we can let the rest of the hands to have one/more similar feature with the target, as far as they are not the target
            # first option
            task_specific_mask = 1 - jnp.any(similarity_mask, axis=0).astype(jnp.int32)
            # second option
            # task_specific_mask = jnp.ones(self.num_cards, dtype=jnp.int32)


        hinter_hand = hint
        guesser_hand = target
        for _ in range(self.hand_size - 1):
            _, rng = jax.random.split(rng)
            hand_masks, selected_cards = rest_of_hands_fn((task_specific_mask, h_hand_mask, g_hand_mask), rng, replace=replace)
            h_hand_mask, g_hand_mask = hand_masks[0, :], hand_masks[1, :]
            hinter_card, guesser_card = selected_cards[0], selected_cards[1]
            hinter_hand = jnp.append(hinter_hand, hinter_card)
            guesser_hand = jnp.append(guesser_hand, guesser_card)

        # shuffle the cards
        _rngs = jax.random.split(rng_hand, 2)
        hinter_hand = jax.random.permutation(_rngs[0], hinter_hand)
        guesser_hand = jax.random.permutation(_rngs[1], guesser_hand)
        
        player_hands = jnp.stack([hinter_hand, guesser_hand])
        # assert player_hands.shape == (2, self.hand_size)
        
        # shuffle the views
        _rngs = jax.random.split(rng_view, self.num_agents)
        permuted_hands = jax.vmap(
            lambda rng: jax.random.permutation(rng, player_hands, axis=1)
        )(_rngs)
        state = State(
            player_hands=permuted_hands, target=target, hint=-1, guess=-1, turn=0
        )

        return jax.lax.stop_gradient(self.get_obs(state)), state, hint


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
    rng = jax.random.PRNGKey(20)
    # reset_modes: exact_match, similarity_match, mutual_exclusive, mutual_exclusive_similarity
    _, state, hint = env.reset_for_eval(rng, reset_mode="mutual_exclusive_similarity", replace=False)
    print(jnp.arange(9).reshape(3, 3))
    print(state)
    print("the ideal hint is: ", hint)