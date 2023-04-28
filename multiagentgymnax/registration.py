from .environments import (
    SimpleTagEnv,
    SimpleWorldCommEnv,
)


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's infamous env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered SMAX environments.")
    
    # 1. MPE PettingZoo Environments
    if env_id == "simple_tag_v2":
        return SimpleTagEnv(**env_kwargs)
    elif env_id == "simple_world_comm_v2":
        return SimpleWorldCommEnv(**env_kwargs)
    
    
registered_envs = [
    "simple_tag_v2",
    "simple_world_comm_v2"
]
    