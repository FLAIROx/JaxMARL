from typing import Dict, List, Tuple, Union
import jax.numpy as jnp

# TODO: programatically generate these mappings from the kinematic trees
#       and add an observation distance parameter to the environment


_agent_action_mapping = {
    "ant_4x2": {
        "agent_0": jnp.array([0, 1]),
        "agent_1": jnp.array([2, 3]),
        "agent_2": jnp.array([4, 5]),
        "agent_3": jnp.array([6, 7]),
    },
    "halfcheetah_6x1": {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
        "agent_3": jnp.array([3]),
        "agent_4": jnp.array([4]),
        "agent_5": jnp.array([5]),
    },
    "hopper_3x1": {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
    },
    "humanoid_9|8": {
        "agent_0": jnp.array([0, 1, 2, 11, 12, 13, 14, 15, 16]),
        "agent_1": jnp.array([3, 4, 5, 6, 7, 8, 9, 10]),
    },
    "walker2d_2x3": {
        "agent_0": jnp.array([0, 1, 2]),
        "agent_1": jnp.array([3, 4, 5]),
    },
    "multiquad_2x4": {
        "agent_0": jnp.array([0, 1, 2, 3]),
        "agent_1": jnp.array([4, 5, 6, 7])
    },
    "quad_1x4": {
        "agent_0": jnp.array([0, 1, 2, 3]),
    },
    "multiquad_3x4": {
        "agent_0": jnp.array([0, 1, 2, 3]),
        "agent_1": jnp.array([4, 5, 6, 7]),
        "agent_2": jnp.array([8, 9, 10, 11])
    }
}


def listerize(ranges: List[Union[int, Tuple[int, int]]]) -> List[int]:
    return [
        i
        for r in ranges
        for i in (range(r[0], r[1] + 1) if isinstance(r, tuple) else [r])
    ]


ranges: Dict[str, Dict[str, List[Union[int, Tuple[int, int]]]]] = {
    "ant_4x2": {
        "agent_0": [(0, 5), 6, 7, 9, 11, (13, 18), 19, 20],
        "agent_1": [(0, 5), 7, 8, 9, 11, (13, 18), 21, 22],
        "agent_2": [(0, 5), 7, 9, 10, 11, (13, 18), 23, 24],
        "agent_3": [(0, 5), 7, 9, 11, 12, (13, 18), 25, 26],
    },
    "halfcheetah_6x1": {
        "agent_0": [(1, 2), 3, 4, 6, (9, 11), 12],
        "agent_1": [(1, 2), 3, 4, 5, (9, 11), 13],
        "agent_2": [(1, 2), 4, 5, (9, 11), 14],
        "agent_3": [(1, 2), 3, 6, 7, (9, 11), 15],
        "agent_4": [(1, 2), 6, 7, 8, (9, 11), 16],
        "agent_5": [(1, 2), 7, 8, (9, 11), 17],
    },
    "hopper_3x1": {
        "agent_0": [(0, 1), 2, 3, (5, 7), 8],
        "agent_1": [(0, 1), 2, 3, 4, (5, 7), 9],
        "agent_2": [(0, 1), 3, 4, (5, 7), 10],
    },
    "humanoid_9|8": {
        "agent_0": [
            (0, 10),
            (12, 14),
            (16, 30),
            (39, 44),
            (55, 94),
            (115, 124),
            (145, 184),
            (191, 214),
            (227, 232),
            (245, 277),
            (286, 291),
            (298, 321),
            (334, 339),
            (352, 375),
        ],
        "agent_1": [
            (0, 15),
            (22, 27),
            (31, 38),
            (85, 144),
            (209, 244),
            (269, 274),
            (278, 285),
            (316, 351),
        ],
    },
    "walker2d_2x3": {
        "agent_0": [0, (2, 5), (8, 9), (11, 13)],
        "agent_1": [0, 2, (5, 9), (14, 16)],
    },
    "multiquad_2x4": {
        "agent_0": [
            (0, 5),
         (6, 29),
         (54, 57),
         (30, 32),
         (42,44)], 
        "agent_1": [
            (0, 5),
         (30, 53),
         (58, 61),
         (6, 8),
         (18,20)], 
    },
    # local quad mappings
    "multiquad_2x4_local": {
        "agent_0": [
            (98,103),
         (9,17),
         (62,76),
         (54, 57),
         (92,94)], 
        "agent_1": [
            (104,109),
         (33, 41),
         (77, 91),
        (58,61),
         (95,97)], 
    },
    ## spherical local quad mappings
    "multiquad_2x4_spherical": {
        "agent_0": [
            (116,118),  # quad1 payload sph coords
            (101,103),  # quad1 lin vel
            (9,17),     # quad1 rot matrix
            (110,112),  # quad1 rel pos sph coords
            (65,76),    # quad1 dynamic state
            (54,57),    # quad1 last action subset
            (122,124),  # quad1->quad2 rel sph coords
        ],
        "agent_1": [
            (119,121),  # quad2 payload sph coords
            (107,109),  # quad2 lin vel
            (33,41),    # quad2 rot matrix
            (113,115),  # quad2 rel pos sph coords
            (80,91),    # quad2 dynamic state
            (58,61),    # quad2 last action subset
            (125,127),  # quad2->quad1 rel sph coords
        ], 
    },
     "multiquad_3x4": {
        "agent_0": [
            (0, 5),
         (6, 29),
         (54, 57),
         (30, 32),
         (42,44)], 
        "agent_1": [
            (0, 5),
         (30, 53),
         (58, 61),
         (6, 8),
         (18,20)], 
        "agent_2": [
            (0, 5),
         (30, 53),
         (58, 61),
         (6, 8),
         (18,20)],
    },
    "quad_1x4": {
        "agent_0": [(0, 24)],  # 24 with actions
    },
}

_agent_observation_mapping = {
    k: {k_: jnp.array(listerize(v_)) for k_, v_ in v.items()} for k, v in ranges.items()
}
