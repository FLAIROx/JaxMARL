from .environments import (
    SimpleMPE,
    SimpleTagMPE,
    SimpleWorldCommMPE,
    SimpleSpreadMPE,
    SimpleCryptoMPE,
    SimpleSpeakerListenerMPE,
    SimplePushMPE,
    SimpleAdversaryMPE,
    SimpleReferenceMPE,
    SwitchRiddle
)

def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered SMAX environments.")
    
    # 1. MPE PettingZoo Environments
    if env_id == "MPE_simple_v2":
        env = SimpleMPE(**env_kwargs)
    elif env_id == "MPE_simple_tag_v2":
        env = SimpleTagMPE(**env_kwargs)
    elif env_id == "MPE_simple_world_comm_v2":
        env = SimpleWorldCommMPE(**env_kwargs)
    elif env_id == "MPE_simple_spread_v2":
        env = SimpleSpreadMPE(**env_kwargs)
    elif env_id == "MPE_simple_crypto_v2":
        env = SimpleCryptoMPE(**env_kwargs)
    elif env_id == "MPE_simple_speaker_listener_v3":
        env = SimpleSpeakerListenerMPE(**env_kwargs)
    elif env_id == "MPE_simple_push_v2":
        env = SimplePushMPE(**env_kwargs)
    elif env_id == "MPE_simple_adversary_v2":
        env = SimpleAdversaryMPE(**env_kwargs)
    elif env_id == "MPE_simple_reference_v2":
        env = SimpleReferenceMPE(**env_kwargs)
    
    # 2. Switch Riddle
    elif env_id == "switch_riddle":
        env = SwitchRiddle(**env_kwargs)
    
    return env, env.default_params
    
registered_envs = [
    "MPE_simple_v2",
    "MPE_simple_tag_v2",
    "MPE_simple_world_comm_v2",
    "MPE_simple_spread_v2",
    "MPE_simple_crypto_v2",
    "MPE_simple_speaker_listener_v3",
    "MPE_simple_push_v2",
    "MPE_simple_adversary_v2",
    "MPE_simple_reference_v2",
    "switch_riddle",
]
    