import os
import robot_data
import numpy as np
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.results_plotter import load_results, ts2xy
from envs import PandaGraspGymEnv
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnSuccessThreshold
from stable_baselines.common.vec_env import DummyVecEnv
from custom_callbacks import TensorboardCallback

best_mean_reward, n_steps = -np.inf, 0
log_dir = "tmp/"

os.makedirs(log_dir, exist_ok=True)


def get_environment():
    env = PandaGraspGymEnv(urdf_root=robot_data.getDataPath(), is_rendering=False, use_ik=True, is_discrete=True,
                           num_controlled_joints=7, is_target_position_fixed=False, max_step_count=1000)
    return env


panda_env = Monitor(get_environment(), log_dir)
eval_env = get_environment()

callback_on_best = StopTrainingOnSuccessThreshold(success_rate_goal=0.8, verbose=1)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=100000,
                             deterministic=True, render=False, callback_on_new_best=callback_on_best,
                             n_eval_episodes=10)

tensorboard_callback = TensorboardCallback(log_dir)

time_steps = 2000000

model = DQN(MlpPolicy,
            panda_env,
            verbose=True,
            tensorboard_log="tensorboard/",
            gamma=.99,
            param_noise=False,
            exploration_fraction=0.2,
            exploration_final_eps=0.02,
            learning_rate=0.001,
            prioritized_replay=False)

model.learn(total_timesteps=time_steps, callback=[eval_callback, tensorboard_callback], log_interval=10)
model.save("result")
