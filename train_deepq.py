import os
import robot_data
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.results_plotter import load_results, ts2xy
from envs import PandaGraspGymEnv

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
    panda_env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), isRendering=False, useIK=True, isDiscrete=True,
                                 numControlledJoints=7, isTargetPositionFixed=True)
    panda_env = Monitor(panda_env, log_dir, allow_early_resets=True)

    time_steps = 500000

    model = DQN(MlpPolicy, panda_env,
                verbose=True,
                tensorboard_log="tensorboard/",
                gamma=.99,
                param_noise=False,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                learning_rate=1e-3,
                buffer_size=50000)

    model.learn(total_timesteps=time_steps, callback=callback, log_interval=1000)
    model.save("model.pkl")


if __name__ == '__main__':
    main()
