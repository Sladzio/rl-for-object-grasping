import robot_data
from envs import PandaGraspGymEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.bench import Monitor
import numpy as np
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import results_plotter
import os

log_dir = "models/"
best_mean_reward, n_steps = -np.inf, 0
import matplotlib.pyplot as plt


def callback(_locals, _globals):
    global n_steps, best_mean_reward, log_dir
    # Print stats every 1000 calls
    if (n_steps) % 1000 == 0:
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
                _locals['self'].save('best_model.pkl')
    n_steps += 1
    return True


def main():
    global log_dir
    os.makedirs(log_dir, exist_ok=True)
    batch_size = 16
    panda_env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), renders=False, useIK=True, isDiscrete=False,
                                 numControlledJoints=7)

    # panda_env = Monitor(panda_env, filename=log_dir, allow_early_resets=True)
    panda_env = DummyVecEnv([lambda: panda_env])
    normalize_observations = False
    memory_limit = 1000000
    param_noise = None
    normalize_returns = True
    gamma = 0.9
    n_actions = panda_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = DDPG(LnMlpPolicy, panda_env, normalize_observations=normalize_observations, gamma=gamma,
                 batch_size=batch_size,
                 memory_limit=memory_limit, normalize_returns=normalize_returns, verbose=1, param_noise=param_noise,
                 action_noise=action_noise, tensorboard_log="tensorboard/", reward_scale=1)
    model.learn(total_timesteps=10000000)
    plt.show()


if __name__ == '__main__':
    main()
