import numpy as np

def reward_function(distances, threshold, w=2):
  """
  Calculates a reward function for a multi-agent system based on distance to landmarks.

  Args:
      distances: A numpy array of shape (n_agents, n_landmarks) representing distances between agents and landmarks.
      threshold: The maximum distance considered "close enough".
      w: A weight between 0 and 1 to balance coverage and distance error (default: 0.5).

  Returns:
      A float representing the reward value.
  """

  n_agents, n_landmarks = distances.shape

  # Coverage term (normalized by number of landmarks)
  coverage = np.sum(np.min(distances, axis=0)) / n_landmarks

  # Distance error term (squared for emphasis)
  distance_error = np.sum((np.maximum(np.min(distances, axis=0) - threshold, 0))**2) / (n_landmarks * threshold**2)
  
  return 1 - distance_error

  return np.sum(np.square(np.maximum(np.min(distances, axis=0) - threshold, 0)))

  return 1 - distance_error


# Testing data points
distances = np.array([
    [120, 400, 400, 400],  # Agent 1 close to all landmarks
    [400, 120, 400, 400],  # Agent 2 far from landmark 1
    [400, 400, 80, 400],  # Agent 3 close to some landmarks
    [400, 400, 400, 120],  # Agent 3 close to some landmarks
])
threshold = 60

# Test reward with different weights
print("Reward with w=0.5 (balanced):", reward_function(distances, threshold))
#print("Reward with w=1.0 (coverage priority):", reward_function(distances, threshold, w=1.0))
#print("Reward with w=0.0 (distance error priority):", reward_function(distances, threshold, w=0.0))
