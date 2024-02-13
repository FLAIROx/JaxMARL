import jax
from jax import numpy as jnp
jax.config.update('jax_platform_name', 'cpu') # force playing on cpu
from jaxmarl import make
import random
import pprint
import sys
import numpy as np
import argparse
from obl.obl_pytorch import OBLPytorchAgent


class ManualPlayer:
    def act(self, obs) -> int:
        # take action input from user
        while True:
            try:
                print("---")
                action = int(input('Insert manual action: '))
                print("---\n")
                if action>=0&action<=20:
                    break
                else:
                    print('Invalid action.')
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                action = 0
                print('Invalid action.')
        return action


def get_agents(args):
    agents = []

    for player_idx in [0, 1]:
        player_type = getattr(args, f"player{player_idx}")
        weight_file = getattr(args, f"weight{player_idx}")
        if player_type == "manual":
            agents.append(ManualPlayer())
        elif player_type == "obl":
            agents.append(OBLPytorchAgent(args.weight))

    return agents


def play_game(args, action_encoding):
    seed = random.randint(0, 10000)
    print(f"{'-'*10}\nStarting new game with random seed: {seed}\n")

    agents = get_agents(args)
     
    with jax.disable_jit():
        env = make('hanabi', debug=False)
        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)
        obs, state = env.reset(_rng)

        done = False

        print("\n" + "=" * 40 + "\n")

        while not done:
            env.render(state)
            print()

            current_player_idx = np.where(state.cur_player_idx==1)[0][0]
            action = agents[current_player_idx].act(obs)

            print(f"Move played: {action_encoding[action]} ({action})")

            actions = {agent:jnp.array([action]) for agent in env.agents}
            rng, _rng = jax.random.split(rng)
            obs, state, reward, dones, infos = env.step(_rng, state, actions)
            done = dones['__all__']

            print("\n" + "=" * 40 + "\n")

        print('Game Ended.')


def main(args):
    action_encoding = {
        0: "INVALID",
        1: "Discard 0",
        2: "Discard 1",
        3: "Discard 2",
        4: "Discard 3",
        5: "Discard 4",
        6: "Play 0",
        7: "Play 1",
        8: "Play 2",
        9: "Play 3",
        10: "Play 4",
        11: "Reveal player +1 color R",
        12: "Reveal player +1 color Y",
        13: "Reveal player +1 color G",
        14: "Reveal player +1 color W",
        15: "Reveal player +1 color B",
        16: "Reveal player +1 rank 1",
        17: "Reveal player +1 rank 2",
        18: "Reveal player +1 rank 3",
        19: "Reveal player +1 rank 4",
        20: "Reveal player +1 rank 5",
    }

    print('Starting Hanabi. Remember, actions encoding is:')
    pprint.pprint(action_encoding)

    play_game(args, action_encoding)

    new_game = 'y'
    while new_game=='y':
        new_game = input('New Game?')
        if new_game=='y':
            play_game(args, action_encoding)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player0", type=str, default="manual")
    parser.add_argument("--player1", type=str, default="obl")
    parser.add_argument("--weight0", type=str, default=None)
    parser.add_argument("--weight1", type=str, default=None)
    args = parser.parse_args()
    return args

    
if __name__=='__main__':
    args = parse_args()
    main(args)
