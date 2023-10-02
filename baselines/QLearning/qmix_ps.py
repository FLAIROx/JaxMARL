"""
End-to-End JAX Implementation of QMix with Parameters Sharing.

Notice:
- Agents are controlled by a single RNN (parameters sharing).
- Works also with non-homogenous agents (different obs/action spaces).
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network and a target mixer network (hard-updated).
- Loss is the 1-step TD error.
- Adam optimizer is used instead of RMSPROP.
- The environment is reset at the end of each episode.
- Trained with a team reward (reward['__all__'])
- At the moment, last_actions are not included in the agents' observations.

The implementation closely follows the original Pymarl: https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.core import frozen_dict

from .utils import CTRolloutManager, EpsilonGreedy, Transition, UniformBuffer

from functools import partial


class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )
    
class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)

        return hidden, q_vals
    
    @partial(jax.jit, static_argnums=0)
    def homogeneous_pass(self, params, hidden_state, obs, dones):
        """
        - concatenate agents and parallel envs to process them in one batch
        - assumes all agents are homogenous (same obs and action shapes)
        - assumes the first dimension is the time step
        - assumes the other dimensions except the last one can be considered as batches
        - returns a dictionary of q_vals indexed by the agent names
        """
        agents, flatten_agents_obs = zip(*obs.items())
        original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
        batched_input = (
            jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
            jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
        )
        hidden_state, q_vals = self.apply(params, hidden_state, batched_input)
        q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1) # (time_steps, n_agents, n_envs, action_dim)
        q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
        return hidden_state, q_vals
    

class HyperNetwork(nn.Module):
    """HyperNetwork for generating weights of QMix' mixing network."""
    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        return x

class MixingNetwork(nn.Module):
    """
    Mixing network for projecting individual agent Q-values into Q_tot. Follows the original QMix implementation.
    """
    embedding_dim: int
    hypernet_hidden_dim: int

    @nn.compact
    def __call__(self, q_vals, states):
        
        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0)) # (time_steps, batch_size, n_agents)
        
        # hypernetwork
        w_1 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim*n_agents, init_scale=1e-2)(states)
        b_1 = nn.Dense(self.embedding_dim, kernel_init=orthogonal(1e-5), bias_init=constant(0.))(states)
        w_2 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim, init_scale=1e-2)(states)
        b_2 = HyperNetwork(hidden_dim=self.embedding_dim, output_dim=1, init_scale=1e-5)(states)
        
        # monotononicity and reshaping
        w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)
    
        # mix
        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot  = jnp.matmul(hidden, w_2) + b_2
        
        return q_tot.squeeze() # (time_steps, batch_size)


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    def linear_schedule(count):
        frac = 1.0 - (count / config["NUM_UPDATES"])
        return config["LR"] * frac
    
    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"]) # batched env for testing (has different batch size)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {agent: wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(env.agents)}
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            transition = Transition(obs, actions, rewards, dones)
            return env_state, transition
        _, sample_traj = jax.lax.scan(
            _env_sample_step, env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree_map(lambda x: x[:, 0], sample_traj) # remove the NUM_ENV dim
        buffer = UniformBuffer(parallel_envs=config["NUM_ENVS"], batch_size=config["BUFFER_BATCH_SIZE"], max_size=config["BUFFER_SIZE"])
        buffer_state = buffer.reset(sample_traj_unbatched)

        # INIT NETWORK
        # init agent
        agent = AgentRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config['AGENT_HIDDEN_DIM'])
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
            jnp.zeros((1, 1)) # (time_step, batch size)
        )
        init_hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
        agent_params = agent.init(_rng, init_hstate, init_x)

        # init mixer
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((len(env.agents), 1, 1))
        state_size = sample_traj.obs['__all__'].shape[-1]  # get the state shape from the buffer
        init_state = jnp.zeros((1, 1, state_size))
        mixer = MixingNetwork(config['MIXER_EMBEDDING_DIM'], config["MIXER_HYPERNET_HIDDEN_DIM"])
        mixer_params = mixer.init(_rng, init_x, init_state)

        # init optimizer
        network_params = frozen_dict.freeze({'agent':agent_params, 'mixer':mixer_params})
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=linear_schedule, eps=config['EPS_ADAM'], weight_decay=config['WEIGHT_DECAY_ADAM']),
        )
        train_state = TrainState.create(
            apply_fn=None,
            params=network_params,
            tx=tx,
        )
        # target network params
        target_network_params = jax.tree_map(lambda x: jnp.copy(x), train_state.params)

        # INIT EXPLORATION STRATEGY
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )


        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, target_network_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state


            # EPISODE STEP
            def _env_step(step_state, unused):

                params, env_state, last_obs, last_dones, hstate, rng, t = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # SELECT ACTION
                # add a dummy time_step dimension to the agent input
                obs_   = {a:last_obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                # get the q_values from the agent netwoek
                hstate, q_vals = agent.homogeneous_pass(params, hstate, obs_, dones_)
                # remove the dummy time_step dimension and index qs by the valid actions of each agent 
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, wrapped_env.valid_actions)
                # explore with epsilon greedy_exploration
                actions = explorer.choose_actions(valid_q_vals, t, key_a)

                # STEP ENV
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones)

                step_state = (params, env_state, obs, dones, hstate, rng, t+1)
                return step_state, transition


            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_ENVS"])

            step_state = (
                train_state.params['agent'],
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
                time_state['timesteps'] # t is needed to compute epsilon
            )

            step_state, traj_batch = jax.lax.scan(
                _env_step, step_state, None, config["NUM_STEPS"]
            )

            # BUFFER UPDATE: save the collected trajectory in the buffer
            buffer_state = buffer.add(buffer_state, traj_batch)

            # LEARN PHASE
            def q_of_action(q, u):
                """index the q_values with action indices"""
                q_u = jnp.take_along_axis(q, jnp.expand_dims(u, axis=-1), axis=-1)
                return jnp.squeeze(q_u, axis=-1)

            def _loss_fn(params, target_network_params, init_hstate, learn_traj):

                obs_ = {a:learn_traj.obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                _, q_vals = agent.homogeneous_pass(params['agent'], init_hstate, obs_, learn_traj.dones)
                _, target_q_vals = agent.homogeneous_pass(target_network_params['agent'], init_hstate, obs_, learn_traj.dones)

                # get the q_vals of the taken actions (with exploration) for each agent
                chosen_action_qvals = jax.tree_map(
                    lambda q, u: q_of_action(q, u)[:-1], # avoid last timestep
                    q_vals,
                    learn_traj.actions
                )

                # get the target q value of the greedy actions for each agent
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, wrapped_env.valid_actions)
                target_max_qvals = jax.tree_map(
                    lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1))[1:], # avoid first timestep
                    target_q_vals,
                    jax.lax.stop_gradient(valid_q_vals)
                )

                # compute q_tot with the mixer network
                chosen_action_qvals_mix = mixer.apply(
                    params['mixer'], 
                    jnp.stack(list(chosen_action_qvals.values())),
                    learn_traj.obs['__all__'][:-1] # avoid last timestep
                )
                target_max_qvals_mix = mixer.apply(
                    target_network_params['mixer'], 
                    jnp.stack(list(target_max_qvals.values())),
                    learn_traj.obs['__all__'][1:] # avoid first timestep
                )

                # compute target
                targets = (
                    learn_traj.rewards['__all__'][:-1]
                    + config["GAMMA"]*(1-learn_traj.dones['__all__'][:-1])*target_max_qvals_mix
                )

                loss = jnp.mean((chosen_action_qvals_mix - jax.lax.stop_gradient(targets))**2)
                
                return loss


            # sample a batched trajectory from the buffer and set the time step dim in first axis
            rng, _rng = jax.random.split(rng)
            _, learn_traj = buffer.sample(buffer_state, _rng) # (batch_size, max_time_steps, ...)
            learn_traj = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), learn_traj) # (max_time_steps, batch_size, ...)
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["BUFFER_BATCH_SIZE"]) 

            # compute loss and optimize grad
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_network_params, init_hs, learn_traj)
            train_state = train_state.apply_gradients(grads=grads)


            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

            # update the states
            time_state['timesteps'] = step_state[-1]
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            target_network_params = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree_map(lambda x: jnp.copy(x), train_state.params),
                lambda _: target_network_params,
                operand=None
            )

            # update the greedy rewards
            rng, _rng = jax.random.split(rng)
            test_metrics = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics(_rng, train_state.params['agent']),
                lambda _: test_metrics,
                operand=None
            )

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), traj_batch.rewards)
            }
            metrics.update(test_metrics) # add the test metrics dictionary

            if config.get("DEBUG"):

                def callback(info):
                    print(
                        f"""
                        Update {info['updates']}:
                        \t n_timesteps: {info['updates']*config['NUM_ENVS']}
                        \t avg_reward: {info['rewards']}
                        \t loss: {info['loss']}
                        """
                    )

                jax.debug.callback(callback, metrics)

            runner_state = (
                train_state,
                target_network_params,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                rng
            )

            return runner_state, metrics

        def get_greedy_metrics(rng, params):
            """Help function to test greedy policy during training"""
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                obs_   = {a:last_obs[a] for a in env.agents}
                obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                hstate, q_vals = agent.homogeneous_pass(params, hstate, obs_, dones_)
                actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q.squeeze(0)[..., valid_idx], axis=-1), q_vals, wrapped_env.valid_actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in env.agents+['__all__']}
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_TEST_EPISODES"])
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
            )
            step_state, rews_dones = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )
            # compute the episode returns of the first episode that is done for each parallel env
            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                return jnp.where(first_episode_mask, rewards, 0.).sum()
            all_dones = rews_dones[1]['__all__']
            returns = jax.tree_map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rews_dones[0])
            metrics = {
                'test_returns': returns # episode returns
            }
            return metrics

        rng, _rng = jax.random.split(rng)
        test_metrics = get_greedy_metrics(_rng, train_state.params['agent']) # initial greedy metrics

        # train
        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            target_network_params,
            env_state,
            buffer_state,
            time_state,
            init_obs,
            init_dones,
            test_metrics,
            _rng
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {'runner_state':runner_state, 'metrics':metrics}
    
    return train


if __name__ == "__main__":

    from smax import make
    import time
    env_name = "MPE_simple_spread_v3"
    env = make(env_name)
    config = {
        "NUM_ENVS":8,
        "NUM_STEPS": env.max_steps,
        "BUFFER_SIZE":5000,
        "BUFFER_BATCH_SIZE":32,
        "TOTAL_TIMESTEPS":2e6+5e4,
        "AGENT_HIDDEN_DIM":64,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 100000,
        "MIXER_EMBEDDING_DIM": 32,
        "MIXER_HYPERNET_HIDDEN_DIM": 64,
        "MAX_GRAD_NORM": 25,
        "TARGET_UPDATE_INTERVAL": 200, 
        "LR": 0.005,
        "EPS_ADAM":1e-2,
        "WEIGHT_DECAY_ADAM":0.00001,
        "GAMMA": 0.9,
        "DEBUG": False,
        "NUM_TEST_EPISODES":32,
        "TEST_INTERVAL": 5e4
    }

    b = 32 # number of concurrent trainings
    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, b)
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    t0 = time.time()
    outs = jax.block_until_ready(train_vjit(rngs))
    t1 = time.time() - t0
    print(f"time: {t1:.2f} s")

    from matplotlib import pyplot as plt
    def rolling_average_plot(x, y, window_size=50, label=''):
        y = jnp.mean(jnp.reshape(y, (-1, window_size)), axis=1)
        x = x[::window_size]
        plt.plot(x, y, label=label)

    rolling_average_plot(outs['metrics']['timesteps'][0], outs['metrics']['rewards']['__all__'].mean(axis=0))
    plt.xlabel("Timesteps")
    plt.ylabel("Team Returns")
    plt.title(f"{env_name} returns (mean of {b} seeds)")
    plt.show()
