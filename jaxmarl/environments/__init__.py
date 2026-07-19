from .coin_game import CoinGame
from .hanabi import Hanabi
from .jaxnav import JaxNav
from .mpe import (
    SimpleAdversaryMPE,
    SimpleCryptoMPE,
    SimpleFacmacMPE,
    SimpleFacmacMPE3a,
    SimpleFacmacMPE6a,
    SimpleFacmacMPE9a,
    SimpleMPE,
    SimplePushMPE,
    SimpleReferenceMPE,
    SimpleSpeakerListenerMPE,
    SimpleSpreadMPE,
    SimpleTagMPE,
    SimpleWorldCommMPE,
)
from .multi_agent_env import MultiAgentEnv, State
from .overcooked import Overcooked, overcooked_layouts
from .overcooked_v2 import OvercookedV2, overcooked_v2_layouts
from .smax import SMAX, HeuristicEnemySMAX, LearnedPolicyEnemySMAX
from .storm import InTheGrid, InTheGrid_2p, InTheMatrix
from .switch_riddle import SwitchRiddle

# MABrax requires brax, an optional dependency due to the deprecation: `pip install jaxmarl[mabrax]`
try:
    from .mabrax import Ant, HalfCheetah, Hopper, Humanoid, Walker2d

    MABRAX_AVAILABLE = True
except ImportError:
    MABRAX_AVAILABLE = False

# Submoduled environments
try:
    print("Importing submoduled environments...")
    from jaxrobotarium import (
        RWARE,
        ArcticTransport,
        Discovery,
        Foraging,
        MaterialTransport,
        Navigation,
        PredatorPrey,
        Warehouse,
    )

    SUBMODULE_ENVIRONMENTS = True
    print("Submoduled environments imported successfully.")
except ImportError:
    print("Submoduled environments not found. Skipping import.")
    SUBMODULE_ENVIRONMENTS = False

__all__ = [
    "CoinGame",
    "Hanabi",
    "JaxNav",
    "MABRAX_AVAILABLE",
    # mabrax (conditionally available, requires brax)
    "Ant",
    "HalfCheetah",
    "Hopper",
    "Humanoid",
    "Walker2d",
    "SimpleAdversaryMPE",
    "SimpleCryptoMPE",
    "SimpleFacmacMPE",
    "SimpleFacmacMPE3a",
    "SimpleFacmacMPE6a",
    "SimpleFacmacMPE9a",
    "SimpleMPE",
    "SimplePushMPE",
    "SimpleReferenceMPE",
    "SimpleSpeakerListenerMPE",
    "SimpleSpreadMPE",
    "SimpleTagMPE",
    "SimpleWorldCommMPE",
    "MultiAgentEnv",
    "State",
    "Overcooked",
    "overcooked_layouts",
    "OvercookedV2",
    "overcooked_v2_layouts",
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
    "InTheGrid",
    "InTheGrid_2p",
    "InTheMatrix",
    "SwitchRiddle",
    "SUBMODULE_ENVIRONMENTS",
    # jaxrobotarium submodule (conditionally available)
    "RWARE",
    "ArcticTransport",
    "Discovery",
    "Foraging",
    "MaterialTransport",
    "Navigation",
    "PredatorPrey",
    "Warehouse",
]
