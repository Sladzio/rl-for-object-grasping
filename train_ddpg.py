from stable_baselines.results_plotter import load_results, ts2xy

import object_data
from envs import PandaGraspGymEnv
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
import numpy as np
import os
from CustomMonitor import CustomMonitor
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from custom_callbacks import MeanHundredEpsTensorboardCallback, SuccessRateTensorboardCallback
from stable_baselines.her import HERGoalEnvWrapper

best_mean_reward, n_steps = -np.inf, 0
ddpg_tag = "DDPG"

log_dir = ddpg_tag + "/log/"
eval_log_dir = ddpg_tag + "/log/eval/"
trained_models_dir = ddpg_tag + "/trainedModels/"


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

    time_steps = 2000000

    n_actions = panda_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.005) * np.ones(n_actions))
    # param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)

    model = DDPG(LnMlpPolicy,
                 panda_env,
                 eval_env=eval_env,
                 verbose=1,
                 param_noise=None,
                 action_noise=action_noise,
                 tensorboard_log="tensorboard/",
                 gamma=0.99,
                 n_cpu_tf_sess=None)

    model.learn(total_timesteps=time_steps,
                callback=[mean_hundred_eps_callback, succ_rate_callback, every_n_steps_callback],
                tb_log_name=ddpg_tag,
                log_interval=10)

    model.save("DDPG_model")

    # results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG")
    # plt.show()


if __name__ == '__main__':
    main()
