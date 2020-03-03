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

best_mean_reward, n_steps = -np.inf, 0
log_dir = "tmp/"


def status_callback(_locals, _globals):
    """
    Callback called at each step
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        successful_grasps = panda_env.successful_grasp_count
        grasp_attempts = panda_env.grasp_attempts_count
        episode = panda_env.episode_number
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(
                "TimeSteps: {} Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} Succesful grasps: {} Attempted grasps: {} Episode: {}".format(
                    x[-1], best_mean_reward, mean_reward, successful_grasps, grasp_attempts, episode))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
    n_steps += 1
    return True


os.makedirs(log_dir, exist_ok=True)


def get_environment():
    env = PandaGraspGymEnv(urdf_root=robot_data.getDataPath(), is_rendering=False, use_ik=True, is_discrete=True,
                           num_controlled_joints=7, is_target_position_fixed=True)
    env = DummyVecEnv([lambda: env])
    return env


panda_env = get_environment()
eval_env = get_environment()

callback_on_best = StopTrainingOnSuccessThreshold(success_rate_goal=0.9, verbose=1)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=10000,
                             deterministic=True, render=False, callback_on_new_best=callback_on_best, n_eval_episodes=1)

time_steps = 1000000

model = DQN(MlpPolicy,
            panda_env,
            verbose=True,
            tensorboard_log="tensorboard/",
            gamma=.99,
            param_noise=False,
            exploration_fraction=0.2,
            exploration_final_eps=0.02,
            buffer_size=50000,
            learning_rate=0.001)

model.learn(total_timesteps=time_steps, callback=eval_callback, log_interval=10)
model.save("result")
