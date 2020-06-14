import os
import numpy as np
import object_data
from utils.custom_callbacks import MeanHundredEpsTensorboardCallback
from envs import PandaGraspGymEnv
from stable_baselines import DQN, HER
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.deepq.policies import LnMlpPolicy
from stable_baselines.her import GoalSelectionStrategy

best_mean_reward, n_steps = -np.inf, 0
log_dir = "tmp/"

os.makedirs(log_dir, exist_ok=True)


def get_environment():
    env = PandaGraspGymEnv(urdf_root=object_data.getDataPath(), is_rendering=False, use_ik=True, is_discrete=True,
                           num_controlled_joints=7, max_step_count=1000,
                           reward_type='sparse')
    return env


panda_env = Monitor(get_environment(), log_dir)

every_n_steps_callback = CheckpointCallback(50000, "./logs/")

mean_hundred_eps_callback = MeanHundredEpsTensorboardCallback(log_dir)

time_steps = 10000000

model = HER(LnMlpPolicy, panda_env, DQN, n_sampled_goal=4,
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            verbose=True, tensorboard_log="tensorboard/", gamma=.99, param_noise=True, exploration_fraction=0.1,
            exploration_final_eps=0.02, learning_rate=0.001, prioritized_replay=False,
            target_network_update_freq=1000)

model.learn(total_timesteps=time_steps, callback=[mean_hundred_eps_callback, every_n_steps_callback],
            log_interval=10)

model.save("result")
