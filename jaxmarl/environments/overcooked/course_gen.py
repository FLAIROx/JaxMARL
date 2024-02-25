from overcooked import Actions
import random


def generate_random_actions(n):
    """
    Generate two lists of n randomly generated actions from the Actions enumeration.

    Parameters:
        n (int): Number of actions to generate.

    Returns:
        tuple: Two lists of randomly generated actions.
    """
    actions_1 = [random.choice(list(Actions)) for _ in range(n)]
    actions_2 = [random.choice(list(Actions)) for _ in range(n)]
    return actions_1, actions_2

def write_actions_to_file(actions_1, actions_2, filename):
    """
    Write the two arrays into a file.

    Parameters:
        actions_1 (list): List of actions for player 1.
        actions_2 (list): List of actions for player 2.
        filename (str): Name of the file to write to.
    """
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode) as file:
        for action_1, action_2 in zip(actions_1, actions_2):
            file.write(f"{action_1.value},{action_2.value}\n")

# Example usage:
n = 10  # Number of actions
filename = "actions.txt"  # File to write actions to

actions_1, actions_2 = generate_random_actions(n)
write_actions_to_file(actions_1, actions_2, filename)
