from .environments import (
    SimpleMPE,
    SimpleTagMPE,
    SimpleWorldCommMPE,
    SimpleSpreadMPE
)


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered SMAX environments.")
    
    # 1. MPE PettingZoo Environments
    if env_id == "simple_v2":
        return 
    elif env_id == "simple_tag_v2":
        return SimpleTagMPE(**env_kwargs)
    elif env_id == "simple_world_comm_v2":
        return SimpleWorldCommMPE(**env_kwargs)
    elif env_id == "simple_spread_v2":
        return SimpleSpreadMPE(**env_kwargs)
    
    
registered_envs = [
    "simple_tag_v2",
    "simple_world_comm_v2",
    "simple_spread_v2"
]
    