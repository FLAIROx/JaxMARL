from .multi_agent_env import MultiAgentEnv, State
from .mpe import (
    SimpleMPE,
    SimpleTagMPE,
    SimpleWorldCommMPE,
    SimpleSpreadMPE,
    SimpleCryptoMPE,
    SimpleSpeakerListenerMPE,
    SimplePushMPE,
    SimpleAdversaryMPE,
    SimpleReferenceMPE,
    SimpleFacmacMPE,
    SimpleFacmacMPE3a,
    SimpleFacmacMPE6a,
    SimpleFacmacMPE9a,
)
from .smax import SMAX, HeuristicEnemySMAX, LearnedPolicyEnemySMAX
from .switch_riddle import SwitchRiddle
from .overcooked import Overcooked, overcooked_layouts
from .overcooked_v2 import OvercookedV2, overcooked_v2_layouts
from .mabrax import Ant, Humanoid, Hopper, Walker2d, HalfCheetah
from .hanabi import Hanabi
from .storm import InTheGrid, InTheGrid_2p, InTheMatrix
from .coin_game import CoinGame
from .jaxnav import JaxNav

# Submoduled environments
try:
    print("Importing submoduled environments...")
    from jaxrobotarium import Navigation, Discovery, MaterialTransport, Warehouse, ArcticTransport, Foraging, RWARE, PredatorPrey
    SUBMODULE_ENVIRONMENTS = True
    print("Submoduled environments imported successfully.")
except ImportError:
    print("Submoduled environments not found. Skipping import.")
    SUBMODULE_ENVIRONMENTS = False