import inspect
import os

import numpy as np

import object_data
from envs.panda_grasp_env import PandaGraspGymEnv
from stable_baselines import HER
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.her import HERGoalEnvWrapper

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    panda_env = PandaGraspGymEnv(urdf_root=object_data.getDataPath(), is_rendering=True, use_ik=True, is_discrete=True,
                                 num_controlled_joints=7, reward_type="sparse")

    env = HERGoalEnvWrapper(panda_env)

    model = HER.load("logs/rl_model_1000000_steps.zip")

    episode_rewards, episode_lengths, episode_success = evaluate_policy(model, env,
                                                                        n_eval_episodes=50,
                                                                        render=False,
                                                                        deterministic=True,
                                                                        return_episode_rewards=True)
    print(
        "Final Reward {}, Episode Length{}, Success Rate {}".format(np.mean(episode_rewards), np.mean(episode_lengths),
                                                                    np.mean(episode_success)))


if __name__ == '__main__':
    main()
