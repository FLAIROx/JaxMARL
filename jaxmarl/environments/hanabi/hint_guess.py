import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from itertools import product
import chex
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from gymnax.environments.spaces import Discrete


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
