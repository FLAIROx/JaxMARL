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

OBL1A_WEIGHT = "obl/models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw"


class ManualPlayer:
    def __init__(self, player_idx):
        self._player_idx = player_idx

    def act(self, env, obs, legal_moves, curr_player) -> int:
        legal_moves = batchify(env, legal_moves)
        legal_moves = jnp.roll(legal_moves, -1, axis=1)

        actions = np.array([0, 0])

        if curr_player != self._player_idx:
            return actions

        print("Legal moves:")
        print(legal_moves[curr_player])

        # take action input from user
        while True:
            try:
                print("---")
                action = int(input('Insert manual action: '))
                print("action legal:", legal_moves[curr_player][action])
                print("---\n")
                if action >= 0 and action <= 20 and legal_moves[curr_player][action] == 1:
                    break
                else:
                    print('Invalid action.')
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                action = 0
                print('Invalid action.')

        action = (action + 1) % 21

        actions[curr_player] = action

        return actions


def get_agents(args):
    agents = []

    for player_idx in [0, 1]:
        player_type = getattr(args, f"player{player_idx}")
        if args.weight is not None:
            weight_file = args.weight
        else:
            weight_file = getattr(args, f"weight{player_idx}")
        if weight_file is None:
            weight_file = OBL1A_WEIGHT
        if player_type == "manual":
            agents.append(ManualPlayer(player_idx))
        elif player_type == "obl":
            agents.append(OBLPytorchAgent(weight_file))

    return agents


def play_game(args, action_encoding):
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 10000)
    print(f"{'-'*10}\nStarting new game with random seed: {seed}\n")

    agents = get_agents(args)
     
    with jax.disable_jit():
        env = make('hanabi', debug=False)
        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)
        obs, state = env.reset(_rng)
        legal_moves = env.get_legal_moves(state)

        done = False

        print("\n" + "=" * 40 + "\n")

        while not done:
            env.render(state)
            print()

            curr_player = np.where(state.cur_player_idx==1)[0][0]
            actions_all = [
                agents[i].act(env, obs, legal_moves, curr_player) 
                for i in range(len(env.agents))
            ]

            actions = actions_all[curr_player]

            played_action = (actions[curr_player] - 1) % 20
            print("played action:", played_action)
            print(f"Move played: {action_encoding[played_action]} ({played_action})")

            actions = {agent:jnp.array([actions[i]]) for i, agent in enumerate(env.agents)}
            rng, _rng = jax.random.split(rng)
            obs, state, reward, dones, infos = env.step(_rng, state, actions)
            legal_moves = env.get_legal_moves(state)
            done = dones['__all__']

            print("\n" + "=" * 40 + "\n")



        print('Game Ended.')


def main(args):
    action_encoding = {
        0: "Discard 0",
        1: "Discard 1",
        2: "Discard 2",
        3: "Discard 3",
        4: "Discard 4",
        5: "Play 0",
        6: "Play 1",
        7: "Play 2",
        8: "Play 3",
        9: "Play 4",
        10: "Reveal player +1 color R",
        11: "Reveal player +1 color Y",
        12: "Reveal player +1 color G",
        13: "Reveal player +1 color W",
        14: "Reveal player +1 color B",
        15: "Reveal player +1 rank 1",
        16: "Reveal player +1 rank 2",
        17: "Reveal player +1 rank 3",
        18: "Reveal player +1 rank 4",
        19: "Reveal player +1 rank 5",
        20: "INVALID",
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
    parser.add_argument("--player0", type=str, default="obl")
    parser.add_argument("--player1", type=str, default="manual")
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--weight0", type=str, default=None)
    parser.add_argument("--weight1", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    print(args)
    return args

def batchify(env, x): 
    return jnp.stack([x[a] for a in env.agents])

    
if __name__=='__main__':
    args = parse_args()
    main(args)
