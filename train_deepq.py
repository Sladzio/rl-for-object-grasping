import os

import numpy as np

import object_data
from CustomMonitor import CustomMonitor
from custom_callbacks import MeanHundredEpsTensorboardCallback, SuccessRateTensorboardCallback
from envs import PandaGraspGymEnv
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines.her import HERGoalEnvWrapper

best_mean_reward, n_steps = -np.inf, 0
log_dir = "tmp/"

os.makedirs(log_dir, exist_ok=True)


def get_environment():
    env = PandaGraspGymEnv(urdf_root=object_data.getDataPath(), is_rendering=False, use_ik=True, is_discrete=True,
                           num_controlled_joints=7, is_target_position_fixed=False, max_step_count=1000,
                           reward_type='dense')
    return env


panda_env = CustomMonitor(get_environment(), log_dir)
every_n_steps_callback = CheckpointCallback(50000, "./logs/")
mean_hundred_eps_callback = MeanHundredEpsTensorboardCallback(log_dir)
succ_rate_callback = SuccessRateTensorboardCallback(log_dir)
time_steps = 10000000
seed = 100

model = DQN(LnMlpPolicy,
            HERGoalEnvWrapper(panda_env),
            verbose=True,
            tensorboard_log="tensorboard/",
            gamma=.99,
            param_noise=True,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            learning_rate=0.001,
            prioritized_replay=False, seed=seed)

model.learn(total_timesteps=time_steps,
            callback=[mean_hundred_eps_callback, succ_rate_callback, every_n_steps_callback],
            log_interval=10)
model.save("result")
