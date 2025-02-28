URGENCY_CUTOFF = 40  # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20
POT_COOK_TIME = 20  # Time it takes to cook a pot
INDICATOR_ACTIVATION_TIME = 10  # Time the recipe indicator is active for
INDICATOR_ACTIVATION_COST = 5  # Cost of activating the recipe indicator

SHAPED_REWARDS = {
    "PLACEMENT_IN_POT": 3,
    "POT_START_COOKING": 5,
    # "DISH_PICKUP": 0,
    # "PLATE_PICKUP": 0,
    "DISH_PICKUP": 5,
    "PLATE_PICKUP": 3,
}
