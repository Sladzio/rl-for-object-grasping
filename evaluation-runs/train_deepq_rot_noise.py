import os

import numpy as np

import object_data
from CustomMonitor import CustomMonitor
from custom_callbacks import MeanHundredEpsTensorboardCallback, SuccessRateTensorboardCallback
from envs import PandaGraspGymEnv
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines.her import HERGoalEnvWrapper

dqn_tag = "DQN_ROT_NOISE"

log_dir = dqn_tag + "/log/"
eval_log_dir = dqn_tag + "/log/eval/"
trained_models_dir = dqn_tag + "/trainedModels/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(eval_log_dir, exist_ok=True)


def get_environment():
    env = PandaGraspGymEnv(urdf_root=object_data.getDataPath(),
                           is_rendering=False,
                           use_ik=True,
                           is_discrete=True,
                           num_controlled_joints=7,
                           lock_rotation=False,
                           reward_type='dense')
    return env


panda_env = HERGoalEnvWrapper(CustomMonitor(get_environment(), log_dir))
eval_env = HERGoalEnvWrapper(CustomMonitor(get_environment(), eval_log_dir))

every_n_steps_callback = CheckpointCallback(50000, trained_models_dir)
mean_hundred_eps_callback = MeanHundredEpsTensorboardCallback(log_dir)
succ_rate_callback = SuccessRateTensorboardCallback(log_dir)
eval_callback = EvalCallback(eval_env,
                             best_model_save_path=trained_models_dir,
                             log_path=log_dir,
                             eval_freq=100000,
                             deterministic=True,
                             render=False,
                             n_eval_episodes=10)

time_steps = 3000000

model = DQN(LnMlpPolicy,
            panda_env,
            verbose=1,
            tensorboard_log="tensorboard/",
            gamma=.99,
            param_noise=True,
            exploration_fraction=0.35,
            exploration_final_eps=0.02,
            learning_rate=0.001,
            prioritized_replay=False,
            target_network_update_freq=1000, batch_size=32)

model.learn(total_timesteps=time_steps,
            callback=[mean_hundred_eps_callback, succ_rate_callback, every_n_steps_callback, eval_callback],
            log_interval=100,
            tb_log_name=dqn_tag)

model.save(dqn_tag + "_final")
