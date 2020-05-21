from stable_baselines.results_plotter import load_results, ts2xy

import object_data
from envs import PandaGraspGymEnv
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
import numpy as np
import os
from CustomMonitor import CustomMonitor
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from custom_callbacks import MeanHundredEpsTensorboardCallback, SuccessRateTensorboardCallback
from stable_baselines.her import HERGoalEnvWrapper

best_mean_reward, n_steps = -np.inf, 0
log_dir = "log/"
eval_log_dir = "log/eval/"
trained_models_dir = "trainedModels/"


def get_environment():
    env = PandaGraspGymEnv(urdf_root=object_data.getDataPath(),
                           is_rendering=False,
                           use_ik=True,
                           is_discrete=False,
                           num_controlled_joints=7,
                           lock_rotation=True,
                           max_step_count=1000,
                           reward_type='dense')
    return env


def main():
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(trained_models_dir, exist_ok=True)

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

    time_steps = 1000000

    n_actions = panda_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG(MlpPolicy,
                 panda_env,
                 verbose=1,
                 param_noise=None,
                 action_noise=action_noise,
                 tensorboard_log="tensorboard/",
                 gamma=0.99,
                 n_cpu_tf_sess=1,
                 random_exploration=0.0)

    model.learn(total_timesteps=time_steps,
                callback=[mean_hundred_eps_callback, succ_rate_callback, every_n_steps_callback, eval_callback],
                log_interval=10)

    model.save("DDPG_model")

    # results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG")
    # plt.show()


if __name__ == '__main__':
    main()
