"""
In this file is defined a simple OBL agent that aligns with the original torch version and
can be used with the (flax-translated) OBL original weights.

This implementation had the goal to perform inference with pretrained params, for this reason
is kept as minimal as possible.
"""

import jax
from jax import numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Tuple
from chex import Array, PRNGKey
from flax.linen.module import compact, nowrap

class MultiLayerLSTM(nn.RNNCellBase):

    num_layers: int
    features: int

    @compact
    def __call__(self, carry, inputs):

        new_hs = []
        new_cs = []
        for l in range(self.num_layers):
            new_carry, y = nn.LSTMCell(self.features, name=f"l{l}")(
                jax.tree_map(lambda x: x[l], carry), inputs
            )
            new_cs.append(new_carry[0])
            new_hs.append(new_carry[1])
            inputs = y

        new_final_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_final_carry, y

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, batch_dims: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        mem_shape = (self.num_layers,) + batch_dims + (self.features,)
        c = jnp.zeros(mem_shape)
        h = jnp.zeros(mem_shape)
        return (c, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class OBLAgentR2D2(nn.Module):

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
        carry, publ_o = MultiLayerLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim, name="lstm"
        )(carry, x)

        o = priv_o * publ_o

        a = nn.Dense(self.out_dim, name="fc_a")(o)

        return carry, a

    @partial(jax.jit, static_argnums=[0])
    def greedy_act(self, params, carry, inputs):

        obs, legal_move = inputs
        priv_s = obs
        publ_s = obs[..., 125:]

        carry, adv = self.apply(params, carry, (priv_s, publ_s))

        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = jnp.argmax(legal_adv, axis=-1)

        return carry, greedy_action

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, batch_dims: Tuple[int, ...]
    ) -> Tuple[Array, Array]:
        return MultiLayerLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim
        ).initialize_carry(rng, batch_dims)



def example():

    from jaxmarl import make
    from jaxmarl.wrappers.baselines import load_params

    weight_file = "./obl-r2d2-flax/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors"
    params = load_params(weight_file)

    agent = OBLAgentR2D2()
    agent_carry = agent.initialize_carry(jax.random.PRNGKey(0), batch_dims=(2,))
    
    rng = jax.random.PRNGKey(0)
    env = make('hanabi')
    obs, env_state = env.reset(rng)
    env.render(env_state)

    batchify = lambda x: jnp.stack([x[agent] for agent in env.agents])
    unbatchify = lambda x: {agent:x[i] for i, agent in enumerate(env.agents)}

    agent_input = (
        batchify(obs),
        batchify(env.get_legal_moves(env_state))
    )
    agent_carry, actions = agent.greedy_act(params, agent_carry, agent_input)
    actions = unbatchify(actions)

    obs, env_state, rewards, done, info = env.step(rng, env_state, actions)

    print('actions:', {agent:env.action_encoding[int(a)] for agent, a in actions.items()})
    env.render(env_state)

if __name__ == "__main__":
    example()


