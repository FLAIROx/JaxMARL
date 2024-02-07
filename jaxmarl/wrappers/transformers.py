import jax
import jax.numpy as jnp
import numpy as np
from .baselines import JaxMARLWrapper, CTRolloutManager
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from functools import partial

class TransformersCTRolloutManager(CTRolloutManager):
    # wraps envs the rollout manager for managing transformers observations and states matrices 
    
    def __init__(self, env: MultiAgentEnv, batch_size:int):

        # SMAX
        if 'smax' in env.name.lower():
            # For SMAX the simplest approach is to post-process the obs, since they already contain a matrix structure 
            super().__init__(env, batch_size, training_agents=None, preprocess_obs=True)
            self._preprocess_obs = self.smax_obs_vec_to_matrix
            self.global_state = self.smax_global_state

        # SPREAD  
        elif 'spread' in env.name.lower():
            # For SPREAD it is simpler to build the obs from scratch as matrices
            super().__init__(env, batch_size, training_agents=None, preprocess_obs=False)
            if any(issubclass(parent, JaxMARLWrapper) for parent in type(env).__mro__): #apply the wrapped get_obs to the child env class
                self._env._env.get_obs = self.spread_wrapped_get_obs
            else:
                self._env.get_obs = self.spread_wrapped_get_obs
            self.global_state = lambda obs, state: obs['world_state']
        
        else:
            raise NotImplementedError('This implemention currently supports only MPE_spread and SMAX')
        

    @partial(jax.jit, static_argnums=0)
    def smax_obs_vec_to_matrix(self, obs, extra_feats):
        # extract the features relative to others and self and make a matrix
        others_feats = obs[:(self._env.obs_size-len(self._env.own_features))].reshape(-1, len(self._env.unit_features))
        self_feats = obs[-len(self._env.own_features):]
        pad_width = [(0, max(0, others_feats.shape[-1] - self_feats.shape[-1]))]
        self_feats = jnp.pad(self_feats, pad_width, mode='constant', constant_values=0)
        rel_feats = jnp.concatenate((others_feats, self_feats[np.newaxis, :]), axis=0)
        # the last obs vector refers to self
        is_self_feat = jnp.zeros(self.num_allies+self.num_enemies).at[-1].set(1)
        # first are teamates, then enemies and finally the self
        is_agent_feat = jnp.concatenate((jnp.ones(self._env.num_allies-1), jnp.zeros(self._env.num_enemies), jnp.ones(1)))
        feats = jnp.concatenate((
            rel_feats,
            is_agent_feat[:, None],
            is_self_feat[:, None],
        ), axis=1)
        return feats
            
    @partial(jax.jit, static_argnums=0)
    def smax_global_state(self, obs, state):
        # extract the main feats that are defined for all the entities
        main_feats = obs['world_state'][:len(self._env.own_features)*(self._env.num_allies+self._env.num_enemies)]
        main_feats = main_feats.reshape(self._env.num_allies+self._env.num_enemies, -1)
        
        # the other feats are is_agent and unit_type, defined for each entity
        other_feats = obs['world_state'][len(self._env.own_features)*(self._env.num_allies+self._env.num_enemies):]
        other_feats = jnp.swapaxes(other_feats.reshape(-1, self._env.num_allies+self._env.num_enemies), 0,1)
        
        return jnp.concatenate((main_feats, other_feats), axis=1)
    
    @partial(jax.jit, static_argnums=0)
    def spread_wrapped_get_obs(self, state):
        """
        - Obs feats: [d_x, d_y, is_self, is_agent]
        - State feats: [x, y, vel_x, vel_y, is_agent]
        """
        # relative position between agents and other entities
        rel_pos = state.p_pos - state.p_pos[:self._env.num_agents, None, :]
        is_self_feat  = (jnp.arange(self._env.num_entities) == jnp.arange(self._env.num_agents)[:, np.newaxis])
        is_agent_feat = jnp.tile(
            jnp.concatenate((jnp.ones(self._env.num_agents), jnp.zeros(self._env.num_landmarks))),
            (self._env.num_agents, 1)
        )
        feats = jnp.concatenate((
            rel_pos,
            is_self_feat[:, :, None],
            is_agent_feat[:, :, None],
        ), axis=2)

        obs = {
            a:feats[i]
            for i, a in enumerate(self._env.agents)
        }
        
        obs['world_state'] = jnp.concatenate((
            state.p_pos,
            state.p_vel,
            is_agent_feat[0][:, None]
        ), axis=1)
        
        return obs