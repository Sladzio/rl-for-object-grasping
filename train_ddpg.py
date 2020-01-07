from stable_baselines.results_plotter import load_results, ts2xy

import robot_data
from envs import PandaGraspGymEnv
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
import numpy as np
import os
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter
import matplotlib.pyplot as plt

best_mean_reward, n_steps = -np.inf, 0
log_dir = "tmp/"


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True


def main():
    os.makedirs(log_dir, exist_ok=True)
    panda_env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), isRendering=False, useIK=True, isDiscrete=False,
                                 numControlledJoints=7)
    panda_env = Monitor(panda_env, log_dir, allow_early_resets=True)

    param_noise = None
    n_actions = panda_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = DDPG(MlpPolicy, panda_env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log="tensorboard/",gamma=0.9)
    time_steps = 1000000
    model.learn(total_timesteps=time_steps, callback=callback)
    results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG LunarLander")
    plt.show()


if __name__ == '__main__':
    main()
