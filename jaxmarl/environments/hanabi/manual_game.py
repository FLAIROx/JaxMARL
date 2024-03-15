import jax
from jax import numpy as jnp
#jax.config.update('jax_platform_name', 'cpu') # force playing on cpu
from jaxmarl import make
import random
import pprint
import sys
import numpy as np
import argparse
import json

OBL1A_WEIGHT_TORCH = "/app/JaxMARL/jaxmarl/environments/hanabi/obl/models/torch_models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a/model0.pthw"
#OBL1A_WEIGHT_FLAX  = "obl/models/flax_models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors"
OBL1A_WEIGHT_FLAX = "/app/JaxMARL/jaxmarl/environments/hanabi/obl/models/flax_models/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors"

with open('decks_test.json') as f:
    decks_j = json.load(f)

decks_test_rngs = jnp.array([jnp.array(np.array(deck['jax_rng'], dtype=np.uint32))for deck in decks_j])

with open('cpp_deck_actions.json', 'r') as file:
    deck_actions = json.load(file)

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
        
        if player_type == "manual":
            agents.append(ManualPlayer(player_idx))
        elif player_type == 'obl_flax':
            from obl.obl_flax import OBLFlaxAgent
            if weight_file is None:
                weight_file = OBL1A_WEIGHT_FLAX
            agents.append(OBLFlaxAgent(weight_file, player_idx))
        elif player_type == "obl":
            from obl.obl_pytorch import OBLPytorchAgent
            if weight_file is None:
                weight_file = OBL1A_WEIGHT_TORCH
            agents.append(OBLPytorchAgent(weight_file))

    return agents


def play_game(args, action_encoding):
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 10000)
    print(f"{'-'*10}\nStarting new game with random seed: {seed}\n")

    agents = get_agents(args)

    if args.use_jit is not None:
        use_jit = args.use_jit
    else:
        use_jit = True

    with jax.disable_jit(not use_jit):
        env = make('hanabi')
        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)

        # custom seed from the deck test
        _rng = decks_test_rngs[seed]

        # custom actions
        pre_actions = np.array(deck_actions[seed]).astype(int)

        obs, env_state = env.reset(_rng)
        legal_moves = env.get_legal_moves(env_state)

        @jax.jit
        def _step_env(rng, env_state, actions):
            rng, _rng = jax.random.split(rng)
            new_obs, new_env_state, reward, dones, infos = env.step(_rng, env_state, actions)
            new_legal_moves = env.get_legal_moves(new_env_state)
            return rng, new_env_state, new_obs, reward, dones, new_legal_moves
        
        done = False
        cum_rew = 0
        t = 0

        print("\n" + "=" * 40 + "\n")

        while not done:
            env.render(env_state)

            curr_player = np.where(env_state.cur_player_idx==1)[0][0]
            actions_all = [
                agents[i].act(obs, legal_moves, curr_player) 
                for i in range(len(env.agents))
            ]

            actions = actions_all[curr_player]
            #actions = np.zeros(2,dtype=int)
            #actions[curr_player] = pre_actions[t]

            played_action = actions[curr_player]
            print("played action:", played_action)
            print(f"Move played: {action_encoding[played_action]} ({played_action})")

            actions = {agent:jnp.array(actions[i]) for i, agent in enumerate(env.agents)}

            rng, env_state, obs, reward, dones, legal_moves = _step_env(rng, env_state, actions)
            
            done = dones['__all__']
            cum_rew += reward['__all__']
            t += 1

            print("\n" + "=" * 40 + "\n")



        print('Game Ended. Score:', cum_rew)


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
    while False and new_game=='y':
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
    parser.add_argument("--use_jit", type=bool, default=True)
    args = parser.parse_args()
    print(args)
    return args

def batchify(env, x): 
    return jnp.stack([x[a] for a in env.agents])

    
if __name__=='__main__':
    args = parse_args()
    main(args)
