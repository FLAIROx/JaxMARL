"""
In this file is defined a simple OBL agent that aligns with the original torch version and
can be used with the (flax-translated) OBL original weights.

This implementation had the goal to perform inference with pretrained params, for this reason
is kept as minimal as possible.
"""

import numpy as np
import flax
import jax
from jax import numpy as jnp
import flax.linen as nn
from functools import partial
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
)
from chex import Array, PRNGKey
from flax.linen import initializers
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.activation import sigmoid, tanh
from flax.linen.module import compact, nowrap
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict


class TorchAlignedLSTMCell(nn.RNNCellBase):

    features: int
    gate_fn: Callable[..., Any] = sigmoid
    activation_fn: Callable[..., Any] = tanh
    kernel_init = default_kernel_init
    recurrent_kernel_init = initializers.orthogonal()
    bias_init = initializers.zeros_init()
    carry_init = initializers.zeros_init()

    @compact
    def __call__(self, carry, inputs):

        c, h = carry
        hidden_features = h.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        dense_h = partial(
            Dense,
            features=hidden_features,
            use_bias=True,
        )
        dense_i = partial(
            Dense,
            features=hidden_features,
            use_bias=True,
        )
        i = self.gate_fn(dense_i(name="ii")(inputs) + dense_h(name="hi")(h))
        f = self.gate_fn(dense_i(name="if")(inputs) + dense_h(name="hf")(h))
        g = self.activation_fn(dense_i(name="ig")(inputs) + dense_h(name="hg")(h))
        o = self.gate_fn(dense_i(name="io")(inputs) + dense_h(name="ho")(h))
        new_c = f * c + i * g
        new_h = o * self.activation_fn(new_c)
        return (new_c, new_h), new_h

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_dims + (self.features,)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class TorchAlignedLSTM(nn.RNNCellBase):

    num_layers: int
    features: int

    @compact
    def __call__(self, carry, inputs):

        new_hs = []
        new_cs = []
        for l in range(self.num_layers):
            new_carry, y = TorchAlignedLSTMCell(self.features, name=f"l{l}")(
                jax.tree_map(lambda x: x[l], carry), inputs
            )
            new_cs.append(new_carry[0])
            new_hs.append(new_carry[1])
            inputs = y

        new_final_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_final_carry, y

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        batch_dims = input_shape[:-1]
        key1, key2 = jax.random.split(rng)
        mem_shape = (self.num_layers,) + batch_dims + (self.features,)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class SimpleOBLAgent_(nn.Module):

    hid_dim: int = 512
    out_dim: int = 21
    num_lstm_layer: int = 2
    num_ff_layer: int = 1

    @compact
    def __call__(self, carry, inputs):

        priv_s, publ_s = inputs

        # private net
        priv_o = nn.Sequential(
            [
                nn.Dense(self.hid_dim, name="priv_net_dense_0"),
                nn.relu,
                nn.Dense(self.hid_dim, name="priv_net_dense_1"),
                nn.relu,
                nn.Dense(self.hid_dim, name="priv_net_dense_2"),
                nn.relu,
            ]
        )(priv_s)

        # public net (MLP+lstm)
        x = nn.Sequential([nn.Dense(self.hid_dim, name="publ_net_dense_0"), nn.relu])(
            publ_s
        )
        carry, publ_o = TorchAlignedLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim, name="lstm"
        )(carry, x)

        o = priv_o * publ_o

        a = nn.Dense(self.out_dim, name="fc_a")(o)

        return carry, a

    def greedy_act(self, params, carry, inputs):

        priv_s, publ_s, legal_move = inputs
        carry, adv = self.apply(params, carry, (priv_s, publ_s))

        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = jnp.argmax(legal_adv, axis=-1)

        return carry, greedy_action

    @nowrap
    def initialize_carry(self, batch_dims: Tuple[int, ...]) -> Tuple[Array, Array]:
        mem_shape = (self.num_lstm_layer,) + batch_dims + (self.hid_dim,)
        c = jnp.zeros(mem_shape)
        h = jnp.zeros(mem_shape)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class SimpleOBLAgent(nn.Module):

    hid_dim: int = 512
    out_dim: int = 21
    num_lstm_layer: int = 2
    num_ff_layer: int = 1

    @compact
    def __call__(self, carry, inputs):

        priv_s, publ_s = inputs

        # private net
        priv_o = nn.Sequential(
            [
                nn.Dense(self.hid_dim, name="priv_net_dense_0"),
                nn.relu,
                nn.Dense(self.hid_dim, name="priv_net_dense_1"),
                nn.relu,
                nn.Dense(self.hid_dim, name="priv_net_dense_2"),
                nn.relu,
            ]
        )(priv_s)

        # public net (MLP+lstm)
        x = nn.Sequential([nn.Dense(self.hid_dim, name="publ_net_dense_0"), nn.relu])(
            publ_s
        )
        carry, publ_o = TorchAlignedLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim, name="lstm"
        )(carry, x)

        o = priv_o * publ_o

        a = nn.Dense(self.out_dim, name="fc_a")(o)

        return carry, a

    @partial(jax.jit, static_argnums=[0])
    def greedy_act(self, params, obs):
        # aligned as much as possible with obl r2d2 agent

        legal_move = obs["legal_move"]
        inputs = (obs["priv_s"], obs["publ_s"])
        carry = (obs["c0"], obs["h0"])

        carry, adv = self.apply(params, carry, inputs)

        #print("adv:", adv)

        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = jnp.argmax(legal_adv, axis=-1)

        #print("legal adv:", legal_adv)

        reply = {}
        reply["a"] = greedy_action
        reply["c0"] = carry[0]
        reply["h0"] = carry[1]

        return reply

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        return TorchAlignedLSTM(
            num_layers=self.num_lstm_layer,
            features=self.hid_dim,
        ).initialize_carry(rng, input_shape)

    @property
    def num_feature_axes(self) -> int:
        return 1


def load_params(filename):
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


class OBLFlaxAgent:

    def __init__(self, weight_file, player_idx):
        self.player_id = player_idx
        self.params = load_params(weight_file)
        self.model = SimpleOBLAgent()
        self._hid = {
            "h0": jnp.zeros((2, 2, 512)),  # num_layer, batch_size, dim
            "c0": jnp.zeros((2, 2, 512)),
        }

    def act(self, obs, legal_moves, curr_player):

        obs = self._batchify(obs)
        legal_moves = self._batchify(legal_moves)
        #legal_moves = jnp.roll(legal_moves, -1, axis=1)

        flax_input = {
            "priv_s": obs,  # batch*agents, ...
            "publ_s": obs[..., 125:],  # batch*agents, remove private info (other agents' hands)
            "h0": self._hid["h0"],  # num_layer, batch*agents, dim
            "c0": self._hid["c0"],  # num_layer, batch*agents, dim
            "legal_move": legal_moves,
        }

        act_result = self.model.greedy_act(self.params, flax_input)
        actions = act_result.pop("a")
        self._hid = act_result
        #actions = np.where(actions + 1 == 21, 0, actions + 1)
        return np.array(actions)

    def _batchify(self, x_dict):
        return jnp.stack([x_dict[agent] for agent in ['agent_0','agent_1']])
