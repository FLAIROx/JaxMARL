"""
End-to-End JAX Implementation of VDN with Parameters Sharing and possibility to control some agents with pretrained networks.

Notice:
- The pretrained network is assumed to follow the same schema of the agent network to train (i.e., AgentRNN below).
- The pretrained agents are frozen, i.e. they don't improve their policy during training.

"""


import jax
import jax.numpy as jnp
import numpy as np

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import os

from smax import make
from baselines.QLearning.utils import CTRolloutManager, EpsilonGreedy, Transition, UniformBuffer, load_params
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



def make_train(config, env, pretrained_agents:dict):

    """pretrained_agents is a dictionary containing some agent networks indexed by agents names"""

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # get the list of the trainable agents names
    agents = [agent for agent in env.agents if agent not in pretrained_agents]
    
    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"], training_agents=agents)
        test_env = CTRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"], training_agents=agents) # batched env for testing
        valid_actions = {k:v for k, v in wrapped_env.valid_actions.items() if k in agents}
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            trainable_actions = {agent: wrapped_env.batch_sample(jax.random.PRNGKey(0), agent) for agent in agents}
            pretrained_actions = {agent: wrapped_env.batch_sample(jax.random.PRNGKey(0), agent) for agent in pretrained_agents}
            actions = {**trainable_actions, **pretrained_actions}
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(jax.random.PRNGKey(0), env_state, actions)
            transition = Transition(obs, trainable_actions, rewards, dones)
            return env_state, transition
        _, sample_traj = jax.lax.scan(
            _env_sample_step, env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree_map(lambda x: x[:, 0], sample_traj) # remove the NUM_ENV dim
        buffer = UniformBuffer(parallel_envs=config["NUM_ENVS"], batch_size=config["BUFFER_BATCH_SIZE"], max_size=config["BUFFER_SIZE"])
        buffer_state = buffer.reset(sample_traj_unbatched)


        # INIT NETWORK
        agent = AgentRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config['AGENT_HIDDEN_DIM'])
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
            jnp.zeros((1, 1)) # (time_step, batch size)
        )
        init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
        network_params = agent.init(_rng, init_hs, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=agent.apply,
            params=network_params,
            tx=tx,
        )
        # target network params
        target_agent_params = jax.tree_map(lambda x: jnp.copy(x), train_state.params)

        # INIT EXPLORATION STRATEGY
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

            # EPISODE STEP
            def _env_step(step_state, unused):

                params, env_state, last_obs, last_dones, hstate, hstate_pretrained, rng, t = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # SELECT ACTION
                # add a dummy time_step dimension to the agent input
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                timed_obs = jax.tree_map(lambda x: x[np.newaxis, :], last_obs)
                
                # get the q_values from the agent network
                obs_   = {a:timed_obs[a] for a in agents} # ensure to pass only the trainable agents to the network
                hstate, q_vals = agent.homogeneous_pass(params, hstate, obs_, dones_)
                # remove the dummy time_step dimension and index qs by the valid actions of each agent 
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, valid_actions)
                # explore with epsilon greedy_exploration
                trainable_actions = explorer.choose_actions(valid_q_vals, t, key_a)

                # add the greedy actions for the pretrained agents (this code could be improved)
                outs  = {a:agent.apply(pretrained_agents[a], hstate_pretrained[i], (timed_obs[a], dones_[a])) for i, a in enumerate(pretrained_agents)}
                q_vals = {k:o[1] for k, o in outs.items()}
                hstate_pretrained = jnp.stack([o[0] for o in outs.values()], axis=0)
                pretrained_actions = jax.tree_util.tree_map(lambda q: jnp.argmax(q.squeeze(0), axis=-1), q_vals)
                actions = {**trainable_actions, **pretrained_actions}
                                                
                # STEP ENV
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, trainable_actions, rewards, dones)

                step_state = (params, env_state, obs, dones, hstate, hstate_pretrained, rng, t+1)
                return step_state, transition


            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(agents)*config["NUM_ENVS"])
            hstate_pretrained = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(pretrained_agents), config["NUM_ENVS"])

            step_state = (
                train_state.params,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                hstate_pretrained,
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

            def _loss_fn(params, target_agent_params, init_hs, learn_traj):

                obs_ = {a:learn_traj.obs[a] for a in agents} # ensure to not pass the global state (obs["__all__"]) to the network
                _, q_vals = agent.homogeneous_pass(params, init_hs, obs_, learn_traj.dones)
                _, target_q_vals = agent.homogeneous_pass(target_agent_params, init_hs, obs_, learn_traj.dones)

                # get the q_vals of the taken actions (with exploration) for each agent
                chosen_action_qvals = jax.tree_map(
                    lambda q, u: q_of_action(q, u)[:-1], # avoid last timestep
                    q_vals,
                    learn_traj.actions
                )

                # get the target q values of the greedy actions
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, valid_actions)
                target_max_qvals = jax.tree_map(
                    lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1))[1:], # get the greedy actions and avoid first timestep
                    target_q_vals,
                    valid_q_vals
                )

                # VDN: computes q_tot as the sum of the agents' individual q values
                chosen_action_qvals_sum = jnp.stack(list(chosen_action_qvals.values())).sum(axis=0)
                target_max_qvals_sum = jnp.stack(list(target_max_qvals.values())).sum(axis=0)

                # compute the centralized targets using the "__all__" rewards and dones
                targets = (
                    learn_traj.rewards['__all__'][:-1]
                    + config["GAMMA"]*(1-learn_traj.dones['__all__'][:-1])*target_max_qvals_sum
                )

                loss = jnp.mean((chosen_action_qvals_sum - jax.lax.stop_gradient(targets))**2)

                return loss


            # sample a batched trajectory from the buffer and set the time step dim in first axis
            rng, _rng = jax.random.split(rng)
            _, learn_traj = buffer.sample(buffer_state, _rng) # (batch_size, max_time_steps, ...)
            learn_traj = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), learn_traj) # (max_time_steps, batch_size, ...)
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(agents)*config["BUFFER_BATCH_SIZE"]) 

            # compute loss and optimize grad
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_agent_params, init_hs, learn_traj)
            train_state = train_state.apply_gradients(grads=grads)


            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in wrapped_env.agents+['__all__']}

            # update the states
            time_state['timesteps'] = step_state[-1]
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            target_agent_params = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree_map(lambda x: jnp.copy(x), train_state.params),
                lambda _: target_agent_params,
                operand=None
            )

            # update the greedy rewards
            rng, _rng = jax.random.split(rng)
            test_metrics = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics(_rng, train_state.params),
                lambda _: test_metrics,
                operand=None
            )

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), traj_batch.rewards),
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
                target_agent_params,
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
                params, env_state, last_obs, last_dones, hstate, hstate_pretrained, rng = step_state
                rng, key_s = jax.random.split(rng)
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                timed_obs = jax.tree_map(lambda x: x[np.newaxis, :], last_obs)
                hstate, q_vals = agent.homogeneous_pass(params, hstate, {a:timed_obs[a] for a in agents}, dones_)
                trainable_actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q.squeeze(0)[..., valid_idx], axis=-1), q_vals, valid_actions)
                # add actions from pretrained agents
                outs  = {a:agent.apply(pretrained_agents[a], hstate_pretrained[i], (timed_obs[a], dones_[a])) for i, a in enumerate(pretrained_agents)}
                q_vals = {k:o[1] for k, o in outs.items()}
                hstate_pretrained = jnp.stack([o[0] for o in outs.values()], axis=0)
                pretrained_actions = jax.tree_util.tree_map(lambda q: jnp.argmax(q.squeeze(0), axis=-1), q_vals)
                actions = {**trainable_actions, **pretrained_actions}
                # step
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                step_state = (params, env_state, obs, dones, hstate, hstate_pretrained, rng)
                return step_state, (rewards, dones)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in env.agents+['__all__']}
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(agents)*config["NUM_TEST_EPISODES"])
            hstate_pretrained = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(pretrained_agents), config["NUM_TEST_EPISODES"])
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                hstate_pretrained,
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
        test_metrics = get_greedy_metrics(_rng, train_state.params) # initial greedy metrics
        
        # train
        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            target_agent_params,
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

@hydra.main(version_base=None, config_path="../config", config_name="vdn_ps_pretrained")
def main(config):
    import matplotlib.pyplot as plt
    import time 
    
    config = OmegaConf.to_container(config)

    env = make(config["ENV_NAME"])
    
    config["TOTAL_TIMESTEPS"] = config["TOTAL_TIMESTEPS"] + 5.0e4
    config["NUM_STEPS"] = env.max_steps
    
    # load the pretrained agents
    load_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'QLearning/checkpoints', config["ENV_NAME"])
    pretrained_params = load_params(f'{load_dir}/iql_ns.safetensors')
    pretrained_agents = ['agent_0'] # in simple tag, agents names are ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0'] and agent_0 is the pray
    pretrained_agents = {a:pretrained_params[a] for a in pretrained_agents}

    b = 4 # number of concurrent trainings
    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, b)
    train_vjit = jax.jit(jax.vmap(make_train(config, env, pretrained_agents)))
    t0 = time.time()
    outs = jax.block_until_ready(train_vjit(rngs))
    t1 = time.time() - t0
    print(f"time: {t1:.2f} s")

    def rolling_average_plot(x, y, window_size=50, label=''):
        y = jnp.mean(jnp.reshape(y, (-1, window_size)), axis=1)
        x = x[::window_size]
        plt.plot(x, y, label=label)

    rolling_average_plot(outs['metrics']['timesteps'][0], outs['metrics']['rewards']['__all__'].mean(axis=0))
    plt.xlabel("Timesteps")
    plt.ylabel("Team Returns")
    plt.title(f"{config['ENV_NAME']} returns (mean of {b} seeds)")
    plt.savefig(f"{config['ENV_NAME']}_pretrained_vdn_ps.png")
    plt.show()


if __name__ == "__main__":
    main()