import jax
from jax import numpy as jnp
jax.config.update('jax_platform_name', 'cpu') # force playing on cpu
from jaxmarl import make
import random
import pprint
import sys

def play_game():

    seed = random.randint(0, 10000)
    print(f"{'-'*10}\nStarting new game with random seed: {seed}\n")
     
    with jax.disable_jit():
        env = make('hanabi', debug=True)
        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)
        obs, state = env.reset(_rng)

        env.render(state)
        
        done = False

        while not done:

            # take action input from user
            while True:
                try:
                    action = int(input('Insert next action: '))
                    if action>=0&action<=20:
                        break
                    else:
                        print('Invalid action.')
                except KeyboardInterrupt:
                    sys.exit(0)
                except:
                    print('Invalid action.')

            actions = {agent:jnp.array([action]) for agent in env.agents}
            rng, _rng = jax.random.split(rng)
            obs, state, reward, dones, infos = env.step(_rng, state, actions)
            done = dones['__all__']
            env.render(state, debug=False)

        print('Game Ended.')

def main():

    action_encoding = {
        "Discard 0": 1,
        "Discard 1": 2,
        "Discard 2": 3,
        "Discard 3": 4,
        "Discard 4": 5,
        "Play 0": 6,
        "Play 1": 7,
        "Play 2": 8,
        "Play 3": 9,
        "Play 4": 10,
        "Reveal player +1 color R": 11,
        "Reveal player +1 color Y": 12,
        "Reveal player +1 color G": 13,
        "Reveal player +1 color W": 14,
        "Reveal player +1 color B": 15,
        "Reveal player +1 rank 1": 16,
        "Reveal player +1 rank 2": 17,
        "Reveal player +1 rank 3": 18,
        "Reveal player +1 rank 4": 19,
        "Reveal player +1 rank 5": 20,
        "INVALID": 0
    }


    print('Starting Hanabi. Remember, actions encoding is:')
    pprint.pprint(action_encoding)

    play_game()

    new_game = 'y'
    while new_game=='y':
        new_game = input('New Game?')
        if new_game=='y':
            play_game()

    
if __name__=='__main__':
    main()