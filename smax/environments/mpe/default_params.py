""" Default parameters for MPE environments """

# Action types
DISCRETE_ACT = "Discrete"
CONTINUOUS_ACT = "Continuous"

# Environment
MAX_STEPS = 25
AGENT_RADIUS = 0.05
ADVERSARY_RADIUS = 0.075
LANDMARK_RADIUS = 0.2
MASS = 1.0
ACCEL = 5.0
ADVERSARY_ACCEL = 3.0
MAX_SPEED = -1  # defaults to no max speed
DAMPING = 0.25
CONTACT_FORCE=1e2
CONTACT_MARGIN  = 1e-3
DT = 0.1

# Colours
AGENT_COLOUR = (115, 243, 115)
ADVERSARY_COLOUR = (243, 115, 115)
OBS_COLOUR = (64, 64, 64)