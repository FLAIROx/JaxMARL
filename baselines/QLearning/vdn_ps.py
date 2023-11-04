"""
End-to-End JAX Implementation of VDN with Parameters Sharing.

Notice:
- Agents are controlled by a single RNN (parameters sharing).
- Works also with non-homogenous agents (different obs/action spaces)
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network (hard-updated).
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
import wandb
import hydra
from omegaconf import OmegaConf

from smax import make
from baselines.QLearning.utils import CTRolloutManager, EpsilonGreedy, Transition, UniformBuffer, ScannedRNN


class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

        return hidden, q_vals


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    
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
        agent = AgentRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
            jnp.zeros((1, 1)) # (time_step, batch size)
        )
        init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
        network_params = agent.init(_rng, init_hs, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config['LR'], eps=config['EPS_ADAM']),
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

        def homogeneous_pass(params, hidden_state, obs, dones):
            # concatenate agents and parallel envs to process them in one batch
            agents, flatten_agents_obs = zip(*obs.items())
            original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
            batched_input = (
                jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
                jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
            )
            hidden_state, q_vals = agent.apply(params, hidden_state, batched_input)
            q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1) # (time_steps, n_agents, n_envs, action_dim)
            q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
            return hidden_state, q_vals

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

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
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
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
                train_state.params,
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

            def _loss_fn(params, target_agent_params, init_hs, learn_traj):

                obs_ = {a:learn_traj.obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                _, q_vals = homogeneous_pass(params, init_hs, obs_, learn_traj.dones)
                _, target_q_vals = homogeneous_pass(target_agent_params, init_hs, obs_, learn_traj.dones)

                # get the q_vals of the taken actions (with exploration) for each agent
                chosen_action_qvals = jax.tree_map(
                    lambda q, u: q_of_action(q, u)[:-1], # avoid last timestep
                    q_vals,
                    learn_traj.actions
                )

                # get the target q values of the greedy actions
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, wrapped_env.valid_actions)
                target_max_qvals = jax.tree_map(
                    lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1))[1:], # get the greedy actions and avoid first timestep
                    target_q_vals,
                    valid_q_vals
                )

                # VDN: computes q_tot as the sum of the agents' individual q values
                chosen_action_qvals_sum = jnp.stack(list(chosen_action_qvals.values())).sum(axis=0)
                target_max_qvals_sum = jnp.stack(list(target_max_qvals.values())).sum(axis=0)

                # compute the centralized targets using the "__all__" rewards and dones
                if config.get('TD_LAMBDA_LOSS', True):
                    # time difference loss
                    def _td_lambda_target(ret, values):
                        reward, done, target_qs = values
                        ret = jnp.where(
                            done,
                            target_qs,
                            ret*config['TD_LAMBDA']*config['GAMMA']
                            + reward
                            + (1-config['TD_LAMBDA'])*config['GAMMA']*(1-done)*target_qs
                        )
                        return ret, ret

                    ret = target_max_qvals_sum[-1] * (1-learn_traj.dones['__all__'][-1])
                    ret, td_targets = jax.lax.scan(
                        _td_lambda_target,
                        ret,
                        (learn_traj.rewards['__all__'][-2::-1], learn_traj.dones['__all__'][-2::-1], target_max_qvals_sum[-1::-1])
                    )
                    targets = td_targets[::-1]
                    loss = jnp.mean(0.5*((chosen_action_qvals_sum - jax.lax.stop_gradient(targets))**2))
                else:
                    # standard DQN loss
                    targets = (
                        learn_traj.rewards['__all__'][:-1]
                        + config['GAMMA']*(1-learn_traj.dones['__all__'][:-1])*target_max_qvals_sum
                    )
                    loss = jnp.mean((chosen_action_qvals_sum - jax.lax.stop_gradient(targets))**2)

                return loss


            # sample a batched trajectory from the buffer and set the time step dim in first axis
            rng, _rng = jax.random.split(rng)
            _, learn_traj = buffer.sample(buffer_state, _rng) # (batch_size, max_time_steps, ...)
            learn_traj = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), learn_traj) # (max_time_steps, batch_size, ...)
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["BUFFER_BATCH_SIZE"]) 

            # compute loss and optimize grad
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_agent_params, init_hs, learn_traj)
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
                lambda _: get_greedy_metrics(_rng, train_state.params, time_state),
                lambda _: test_metrics,
                operand=None
            )

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), traj_batch.rewards) # sum of timesteps, mean of envs
            }
            metrics.update(test_metrics) # add the test metrics dictionary

            if config.get('WANDB_ONLINE_REPORT', False):
                def callback(metrics):
                    wandb.log(
                        {
                            "returns": metrics['rewards']['__all__'].mean(),
                            "test_returns": metrics['test_returns']['__all__'].mean(),
                            "timestep": metrics['timesteps'],
                            "updates": metrics['updates'],
                            "loss": metrics['loss'],
                        }
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

        def get_greedy_metrics(rng, params, time_state):
            """Help function to test greedy policy during training"""
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                obs_   = {a:last_obs[a] for a in env.agents}
                obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
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
            if config.get('VERBOSE', False):
                def callback(timestep, val):
                    print(f"Timestep: {timestep}, return: {val}")
                jax.debug.callback(callback, time_state['timesteps']*config['NUM_ENVS'], returns['__all__'].mean())
            return metrics

        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        test_metrics = get_greedy_metrics(_rng, train_state.params, time_state) # initial greedy metrics
        
        # train
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

@hydra.main(version_base=None, config_path="../config", config_name="vdn_ps")
def main(config):
    config = OmegaConf.to_container(config)

    env = make(config["ENV_NAME"])
    
    config["TOTAL_TIMESTEPS"] = config["TOTAL_TIMESTEPS"] + 5.0e4
    config["NUM_STEPS"] = env.max_steps
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["VDN", "PS", "RNN", config["ENV_NAME"]],
        name=f'vdn_ps_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))


if __name__ == "__main__":
    main()
    
