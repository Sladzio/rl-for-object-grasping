try:
    import mpi4py
except ImportError:
    mpi4py = None
from stable_baselines import DQN, TD3

import warnings
# DDPG require MPI to be installed
if mpi4py is None:
    DDPG = None
    warnings.warn("DDPG cannot be used, because mpi4py is not configured correctly")
else:
    from stable_baselines import DDPG

ALGOS = {
    'DQN': DQN,
    'DDPG': DDPG,
    'TD3': TD3
}