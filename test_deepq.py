import inspect
import os

import numpy as np
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv

import robot_data
from envs.panda_grasp_env import PandaGraspGymEnv

from stable_baselines import DQN

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    panda_env = PandaGraspGymEnv(urdf_root=robot_data.getDataPath(), is_rendering=True, use_ik=True, is_discrete=True,
                                 num_controlled_joints=7, is_target_position_fixed=True)
    env = DummyVecEnv([lambda: panda_env])

    model = DQN.load("result.zip")

    episode_rewards, episode_lengths, episode_success = evaluate_policy(model, env,
                                                                        n_eval_episodes=1,
                                                                        render=True,
                                                                        deterministic=True,
                                                                        return_episode_rewards=True)
    print(
        "Final Reward {}, Episode Length{}, Success Rate {}".format(np.mean(episode_rewards), np.mean(episode_lengths),
                                                                    np.mean(episode_success)))


if __name__ == '__main__':
    main()
