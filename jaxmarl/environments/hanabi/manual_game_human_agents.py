import chex
from absl import flags, app
import jax
import jax.numpy as jnp

from jaxmarl import make
from jaxmarl.environments.hanabi.hanabi import HanabiEnv

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_players', default=2, help='Number of players playing the game.')
flags.DEFINE_integer('seed', default=0, help='Game seed. Used for deck generation.')

flags.register_validator(
    'num_players',
    lambda value: 2 <= value <= 5,
    message='--num_players must be between 2 and 5'
)


def batchify(x, agents):
    x = jnp.stack([x[a] for a in agents])
    return x.reshape((len(agents), -1))


def unbatchify(x, agents):
    return {a: x[i] for i, a in enumerate(agents)}


def get_user_action(player_legal_moves):
    while True:
        try:
            print("---")
            action = int(input("Pick Action: "))
            is_legal_action = player_legal_moves[action] == 1.0

            print(f"Is legal action: {is_legal_action}")
            if not is_legal_action:
                raise ValueError("Action is not legal.")

            print("---")
            return action
        except:
            print("Illegal action! Try again.")


def main(argv):
    num_players = FLAGS.num_players
    seed = FLAGS.seed

    print(f"Starting game with seed={seed}")
    key = jax.random.PRNGKey(seed)
    hand_size = 5 if num_players <= 3 else 4
    env: HanabiEnv = make("hanabi", hand_size=hand_size, num_agents=num_players)

    print(f"Action encoding for the environment: {env.action_encoding}")

    rng, _rng = jax.random.split(key)
    obs, env_state = env.reset(_rng)

    score = 0
    env_step_jit = jax.jit(env.step)
    while True:
        env.render(env_state)
        legal_moves_dict = env.get_legal_moves(env_state)
        legal_moves = batchify(legal_moves_dict, env.agents)

        cur_player = jnp.where(env_state.cur_player_idx == 1)[0][0]
        cur_player_legal_moves = legal_moves[cur_player]

        print("Legal moves for current player:")
        legal_moves_encoded = {
            i: env.action_encoding[i]
            for i, m
            in enumerate(cur_player_legal_moves)
            if m == 1.
        }
        print(legal_moves_encoded)

        user_action = get_user_action(cur_player_legal_moves)

        actions = jnp.full(
            env.num_agents, env.num_actions - 1
        ).at[cur_player].set(user_action)

        rng, _rng = jax.random.split(rng)
        actions = unbatchify(actions, env.agents)
        obs, env_state, rewards, dones, infos = env_step_jit(
            _rng, env_state, actions
        )
        print(f"Action played: {env.action_encoding[user_action]}")

        score += rewards['__all__']

        if dones['__all__']:
            break

    print(f"Game ended.\nScore: {score}")


if __name__ == '__main__':
    app.run(main)
