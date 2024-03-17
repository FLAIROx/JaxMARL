import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from jaxmarl.environments.mpe.simple import State, SimpleMPE
from gymnax.environments.spaces import Box
from jaxmarl.environments.mpe.default_params import *


SimpleFacmacMPE3a = lambda: SimpleFacmacMPE(num_good_agents=1, num_adversaries=3, num_landmarks=2,
                            view_radius=1.5, score_function="min")
SimpleFacmacMPE6a = lambda: SimpleFacmacMPE(num_good_agents=2, num_adversaries=6, num_landmarks=4,
                            view_radius=1.5, score_function="min")
SimpleFacmacMPE9a = lambda: SimpleFacmacMPE(num_good_agents=3, num_adversaries=9, num_landmarks=6,
                            view_radius=1.5, score_function="min")

class SimpleFacmacMPE(SimpleMPE):
    def __init__(
        self,
        num_good_agents=1,
        num_adversaries=3,
        num_landmarks=2,
        view_radius=1.5,  # set -1 to deactivate
        score_function="sum"
    ):
        dim_c = 2  # NOTE follows code rather than docs
        action_type = CONTINUOUS_ACT
        view_radius = view_radius if view_radius != -1 else 999999

        num_agents = num_good_agents + num_adversaries
        num_entities = num_agents + num_landmarks

        self.num_good_agents, self.num_adversaries = num_good_agents, num_adversaries

        self.adversaries = ["adversary_{}".format(i) for i in range(num_adversaries)]
        self.good_agents = ["agent_{}".format(i) for i in range(num_good_agents)]
        agents = self.adversaries + self.good_agents

        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        colour = (
            [ADVERSARY_COLOUR] * num_adversaries
            + [AGENT_COLOUR] * num_good_agents
            + [OBS_COLOUR] * num_landmarks
        )

        # Parameters
        rad = jnp.concatenate(
            [
                jnp.full((self.num_adversaries), 0.075),
                jnp.full((self.num_good_agents), 0.05),
                jnp.full((num_landmarks), 0.2),
            ]
        )
        accel = jnp.concatenate(
            [
                jnp.full((self.num_adversaries), 3.0),
                jnp.full((self.num_good_agents), 4.0),
            ]
        )
        max_speed = jnp.concatenate(
            [
                jnp.full((self.num_adversaries), 1.0),
                jnp.full((self.num_good_agents), 1.3),
                jnp.full((num_landmarks), 0.0),
            ]
        )
        collide = jnp.full((num_entities,), True)

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            dim_c=dim_c,
            colour=colour,
            rad=rad,
            accel=accel,
            max_speed=max_speed,
            collide=collide,
        )

        # Overwrite action and observation spaces
        self.observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (16,)) for i in self.adversaries
        }
        self.action_spaces = {i: Box(0.0, 1.0, (5,)) for i in self.adversaries}

        # Introduce partial observability by limiting the agents' view radii
        self.view_radius = jnp.concatenate(
            [
                jnp.full((num_agents), view_radius),
                jnp.full((num_landmarks), 0.0),
            ]
        )

        self.score_function = score_function

    def rewards(self, state: State) -> Dict[str, float]:
        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: int, other_idx: int):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx,
                other_idx,
                state,
            )

        c = _collisions(
            jnp.arange(self.num_good_agents) + self.num_adversaries,
            jnp.arange(self.num_adversaries),
        )  # [agent, adversary, collison]

        def _good(aidx: int, collisions: chex.Array):
            rew = -10 * jnp.sum(collisions[aidx])

            mr = jnp.sum(self.map_bounds_reward(jnp.abs(state.p_pos[aidx])))
            rew -= mr
            return rew

        ad_rew = 10 * jnp.sum(c)

        rew = {a: ad_rew for a in self.adversaries}
        rew.update(
            {
                a: _good(i + self.num_adversaries, c)
                for i, a in enumerate(self.good_agents)
            }
        )
        return rew

    def _prey_policy(self, key: chex.PRNGKey, state: State, aidx: int):
        action = None
        n = 100  # number of positions sampled
        # sample actions randomly from a target circle
        # length = jnp.sqrt(jnp.random.uniform(0, 1, n))
        # angle = jnp.pi * jnp.random.uniform(0, 2, n)

        key, _key = jax.random.split(key)
        length = jnp.sqrt(jax.random.uniform(_key, (n,), minval=0., maxval=1.))
        key, _key = jax.random.split(key)
        angle = jnp.pi * jnp.sqrt(jax.random.uniform(_key, (n,), minval=0., maxval=2.))
        x = length * jnp.cos(angle)
        y = length * jnp.sin(angle)

        # evaluate score for each position
        # check whether positions are reachable
        # sample a few evenly spaced points on the way and see if they collide with anything
        scores = jnp.zeros(n, dtype=jnp.float32)
        n_iter = 5
        if self.score_function == "sum":
            for i in range(n_iter):
                waypoints_length = (length / float(n_iter)) * (i + 1)
                x_wp = waypoints_length * jnp.cos(angle)
                y_wp = waypoints_length * jnp.sin(angle)
                proj_pos = jnp.vstack((x_wp, y_wp)).transpose() + state.p_pos[aidx]
                delta_pos = state.p_pos[None, :, :] - proj_pos[:, None, :]
                dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=2))
                dist_min = self.rad + self.rad[aidx]
                scores = jnp.where((dist < dist_min[None]).sum(axis=1), scores, -9999999)
                if i == n_iter - 1:
                    scores += dist[:, :self.num_adversaries].sum(axis=1)
        elif self.score_function == "min":
            proj_pos = jnp.vstack((x, y)).transpose() + state.p_pos[aidx]
            rel_dis = jnp.sqrt(jnp.sum(jnp.square(state.p_pos[aidx] - state.p_pos[:self.num_adversaries])))
            min_dist_adv_idx = jnp.argmin(rel_dis)
            delta_pos = state.p_pos[:self.num_adversaries][None, :, :] - proj_pos[:, None, :]
            dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=2))
            dist_min = self.rad[:self.num_adversaries] + self.rad[aidx]
            scores = jnp.where((dist < dist_min[None]).sum(axis=1), scores, -9999999)
            scores += dist[:, min_dist_adv_idx]
        else:
            raise Exception("Unknown score function {}".format(self.score_function))
        # move to best position
        best_idx = jnp.argmax(scores)
        chosen_action = jnp.array([x[best_idx], y[best_idx]], dtype=jnp.float32)
        chosen_action = jax.lax.cond(scores[best_idx] < 0, lambda: chosen_action*0.0, lambda: chosen_action)
        return chosen_action

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        u, c = self.set_actions(actions)
        # we throw away num_good_agents now, as num_agents does not differentiate between active and passive agents
        u = u[:-self.num_good_agents]
        for i in range(self.num_good_agents):
            prey_action = self._prey_policy(key, state, u.shape[0]-1+i)
            u = jnp.concatenate([u, prey_action[None]], axis=0)
        if (
            c.shape[1] < self.dim_c
        ):  # This is due to the MPE code carrying around 0s for the communication channels, and due to added prey
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1
            )

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, self.c_noise, self.silent)
        done = jnp.full((self.num_agents), state.step >= self.max_steps)

        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)
        obs = self.get_obs(state)

        info = {}

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0))
        def _common_stats(aidx):
            """Values needed in all observations"""

            landmark_pos = (
                    state.p_pos[self.num_agents:] - state.p_pos[aidx]
            )  # Landmark positions in agent reference frame

            # Zero out unseen agents with other_mask
            other_pos = state.p_pos[: self.num_agents] - state.p_pos[aidx]
            other_vel = state.p_vel[: self.num_agents]

            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents - aidx - 1, axis=0)[
                        : self.num_agents - 1
                        ]
            other_vel = jnp.roll(other_vel, shift=self.num_agents - aidx - 1, axis=0)[
                        : self.num_agents - 1
                        ]

            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            other_vel = jnp.roll(other_vel, shift=aidx, axis=0)

            # mask out entities and other agents not in view radius of agent
            landmark_mask = jnp.sqrt(jnp.sum(landmark_pos ** 2)) > self.view_radius[aidx]
            landmark_pos = jnp.where(landmark_mask, 0.0, landmark_pos)

            other_mask = jnp.sqrt(jnp.sum(other_pos ** 2)) > self.view_radius[aidx]
            other_pos = jnp.where(other_mask, 0.0, other_pos)
            other_vel = jnp.where(other_mask, 0.0, other_vel)

            return landmark_pos, other_pos, other_vel

        landmark_pos, other_pos, other_vel = _common_stats(self.agent_range)

        def _good(aidx):
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                    jnp.zeros_like(other_vel[aidx, -1:].flatten())  # 2 - ensure same dimensionality for all agents!
                ]
            )

        def _adversary(aidx):
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                    other_vel[aidx, -1:].flatten(),  # 2
                ]
            )

        obs = {a: _adversary(i) for i, a in enumerate(self.adversaries)}
        obs.update(
            {a: _good(i + self.num_adversaries) for i, a in enumerate(self.good_agents)}
        )
        return obs

    def get_world_state(self, state: State):
        @partial(jax.vmap, in_axes=(0))
        def _common_stats(aidx):
            """Values needed in all observations"""

            landmark_pos = (
                    state.p_pos[self.num_agents:] - state.p_pos[aidx]
            )  # Landmark positions in agent reference frame

            # Zero out unseen agents with other_mask
            other_pos = state.p_pos[: self.num_agents] - state.p_pos[aidx]
            other_vel = state.p_vel[: self.num_agents]

            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents - aidx - 1, axis=0)[
                        : self.num_agents - 1
                        ]
            other_vel = jnp.roll(other_vel, shift=self.num_agents - aidx - 1, axis=0)[
                        : self.num_agents - 1
                        ]

            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            other_vel = jnp.roll(other_vel, shift=aidx, axis=0)

            return landmark_pos, other_pos, other_vel

        landmark_pos, other_pos, other_vel = _common_stats(self.agent_range)

        def _good(aidx):
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                ]
            )

        def _adversary(aidx):
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                    other_vel[aidx, -1:].flatten(),  # 2
                ]
            )

        obs = {a: _adversary(i) for i, a in enumerate(self.adversaries)}
        obs.update(
            {a: _good(i + self.num_adversaries) for i, a in enumerate(self.good_agents)}
        )
        return obs

if __name__ == "__main__":
    env = SimpleFacmacMPE(0)
    vec_step_env = jax.jit(env.step_env)
    jax.jit(env.step_env)
    import jaxmarl
    env = jaxmarl.make("MPE_simple_pred_prey_v1")
    get_obs = jax.jit(env.get_obs)

    num_envs = 128
    rng = jax.random.PRNGKey(30)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    obsv, env_state = jax.vmap(env.reset)(reset_rng)

    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, num_envs)
    env_act = jnp.zeros((num_envs, 1))
    obsv, env_state, reward, done, info = jax.vmap(env.step)(rng_step, env_state, env_act)
    pass
