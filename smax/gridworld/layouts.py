import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

cramped_room = {
    "height" : 4,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,9,
                            10,14,
                            15,16,17,18,19])
}
forced_coord = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4,
                            5,7,9,
                            10,12,14,
                            15,17,19,
                            20,21,22,23,24])
}



layouts = {
    "cramped_room" : FrozenDict(cramped_room),
    "forced_coord" : FrozenDict(forced_coord)
}