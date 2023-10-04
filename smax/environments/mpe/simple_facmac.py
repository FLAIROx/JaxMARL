import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from smax.environments.mpe.simple import State, SimpleMPE
from gymnax.environments.spaces import Box
from smax.environments.mpe.default_params import *


class SimplePredPreyMPE(SimpleMPE):
    def __init__(
        self,
        num_good_agents=1,
        num_adversaries=3,
        view_radius=1.5  # set -1 to deactivate
    ):
        dim_c = 2  # NOTE follows code rather than docs
        action_type = CONTINUOUS_ACT
        num_obs = 2

        num_agents = num_good_agents + num_adversaries
        num_landmarks = num_obs
        num_entities = num_agents + num_landmarks

        self.num_good_agents, self.num_adversaries = num_good_agents, num_adversaries

        self.adversaries = ["adversary_{}".format(i) for i in range(num_adversaries)]
        self.good_agents = ["agent_{}".format(i) for i in range(num_good_agents)]
        agents = self.adversaries + self.good_agents

        landmarks = ["landmark {}".format(i) for i in range(num_obs)]

        colour = (
            [ADVERSARY_COLOUR] * num_adversaries
            + [AGENT_COLOUR] * num_good_agents
            + [OBS_COLOUR] * num_obs
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

        # Introduce partial observability by limiting the agents' view radii
        self.view_radius = jnp.concatenate(
            [
                jnp.full((num_agents), view_radius),
                jnp.full((num_landmarks), 0.0),
            ]
        )

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

    def _prey_policy(self, key: chex.PRNGKey, state: State):
        action = None
        n = 100  # number of positions sampled
        # sample actions randomly from a target circle
        length = jnp.sqrt(jnp.random.uniform(0, 1, n))
        angle = jnp.pi * jnp.random.uniform(0, 2, n)
        x = length * jnp.cos(angle)
        y = length * jnp.sin(angle)
        # evaluate score for each position
        # check whether positions are reachable
        # sample a few evenly spaced points on the way and see if they collide with anything
        scores = jnp.zeros(n, dtype=jnp.float32)
        n_iter = 5
        # if self.score_function == "sum":
        #     for i in range(n_iter):
        #         waypoints_length = (length / float(n_iter)) * (i + 1)
        #         x_wp = waypoints_length * jnp.cos(angle)
        #         y_wp = waypoints_length * jnp.sin(angle)
        #         # proj_pos = jnp.vstack((x_wp, y_wp)).transpose() + agent.state.p_pos
        #         proj_pos = jnp.vstack((x_wp, y_wp)).transpose() + state.p_pos
        #         for _agent in world.agents:
        #             if _agent.name != agent.name:
        #                 delta_pos = _agent.state.p_pos - proj_pos
        #                 dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=1))
        #                 dist_min = _agent.size + agent.size
        #                 scores[dist < dist_min] = -9999999
        #                 if i == n_iter - 1 and _agent.movable:
        #                     scores += dist
        # elif self.score_function == "min":
        #     rel_dis = []
        #     adv_names = []
        #     # adversaries = self.adversaries(world)
        #     adversaries = self.adversaries
        #     proj_pos = jnp.vstack((x, y)).transpose() + state.p_pos  # the position of the 100 sampled points.
        #     for adv in adversaries:
        #         rel_dis.append(jnp.sqrt(jnp.sum(jnp.square(state.p_pos - adv.state.p_pos))))
        #         adv_names.append(adv.name)
        #     min_dis_adv_name = adv_names[jnp.argmin(rel_dis)]
        #     for adv in adversaries:
        #         delta_pos = adv.state.p_pos - proj_pos
        #         dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=1))
        #         dist_min = adv.size + agent.size
        #         scores[dist < dist_min] = -9999999
        #         if adv.name == min_dis_adv_name:
        #             scores += dist
        # else:
        #     raise Exception("Unknown score function {}".format(self.score_function))
        # move to best position
        best_idx = jnp.argmax(scores)
        chosen_action = jnp.array([x[best_idx], y[best_idx]], dtype=jnp.float32)
        if scores[best_idx] < 0:
            chosen_action *= 0.0 # cannot go anywhere
        return chosen_action

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        u, c = self.set_actions(actions)
        prey_action = self._prey_policy(key, state)
        u = jnp.concatenate([u, prey_action], axis=1)
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
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        # for entity in world.landmarks:
        for entity in state.landmarks:
            dist = jnp.sqrt(jnp.sum(jnp.square(entity.state.p_pos - state.p_pos)))
            if not entity.boundary and (agent.view_radius >= 0) and dist <= agent.view_radius:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                entity_pos.append(jnp.array([0., 0.]))
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:  continue
            dist = jnp.sqrt(jnp.sum(jnp.square(other.state.p_pos - agent.state.p_pos)))
            if agent.view_radius >= 0 and dist <= agent.view_radius:
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            else:
                other_pos.append(jnp.array([0., 0.]))
                if not other.adversary:
                    other_vel.append(jnp.array([0., 0.]))
        return jnp.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def _world_step(self, key: chex.PRNGKey, state: State, u: chex.Array):
        p_force = jnp.zeros((self.num_agents, 2))

        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_agents)
        p_force = self._apply_action_force(
            key_noise, p_force, u, self.u_noise, self.moveable[: self.num_agents]
        )
        # jax.debug.print('jax p_force post agent {p_force}', p_force=p_force)

        # apply environment forces
        p_force = jnp.concatenate([p_force, jnp.zeros((self.num_landmarks, 2))])
        p_force = self._apply_environment_force(p_force, state)
        # print('p_force post apply env force', p_force)
        # jax.debug.print('jax p_force final: {p_force}', p_force=p_force)

        # integrate physical state
        p_pos, p_vel = self._integrate_state(
            p_force, state.p_pos, state.p_vel, self.mass, self.moveable, self.max_speed
        )

        return p_pos, p_vel

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
    env = SimplePredPreyMPE(0)
    vec_step_env = jax.jit(env.step_env)
    jax.jit(env.step_env)
    import smax
    env = smax.make("MPE_simple_pred_prey_v1")
    env.get_obs()
    pass
