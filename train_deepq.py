import os
import robot_data
import numpy as np

from CustomMonitor import CustomMonitor
from stable_baselines import DQN, HER
from stable_baselines.bench import Monitor
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.her import GoalSelectionStrategy
from stable_baselines.results_plotter import load_results, ts2xy
from envs import PandaGraspGymEnv
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnSuccessThreshold, CheckpointCallback
from stable_baselines.common.vec_env import DummyVecEnv
from custom_callbacks import MeanHundredEpsCallback,SuccessRateTensorboardCallback
from stable_baselines.her import HERGoalEnvWrapper

best_mean_reward, n_steps = -np.inf, 0
log_dir = "tmp/"

os.makedirs(log_dir, exist_ok=True)


def get_environment():
    env = PandaGraspGymEnv(urdf_root=robot_data.getDataPath(), is_rendering=False, use_ik=True, is_discrete=True,
                           num_controlled_joints=7, is_target_position_fixed=False, max_step_count=1000,
                           reward_type='dense')
    return env


panda_env = CustomMonitor(get_environment(), log_dir)
eval_env = get_environment()

# eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
#                              log_path='./logs/', eval_freq=100000,
#                              deterministic=True, render=False,
#                              n_eval_episodes=10)

every_n_steps_callback = CheckpointCallback(50000, "./logs/")

mean_hundred_eps_callback = MeanHundredEpsCallback(log_dir)
succ_rate_callback = SuccessRateTensorboardCallback(log_dir)

time_steps = 10000000
model_class = DQN
# model = HER(LnMlpPolicy, panda_env, model_class, n_sampled_goal=4,
#             goal_selection_strategy=GoalSelectionStrategy.FUTURE,
#             verbose=True, tensorboard_log="tensorboard/", gamma=.99, param_noise=True, exploration_fraction=0.1,
#             exploration_final_eps=0.02, learning_rate=0.001, prioritized_replay=False, seed=seed)

model = DQN(LnMlpPolicy,
            HERGoalEnvWrapper(panda_env),
            verbose=True,
            tensorboard_log="tensorboard/",
            gamma=.99,
            param_noise=True,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            learning_rate=0.0005,
            prioritized_replay=False)

model.learn(total_timesteps=time_steps, callback=[mean_hundred_eps_callback, succ_rate_callback, every_n_steps_callback],
            log_interval=10)
model.save("result")
