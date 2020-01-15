import inspect
import os

import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv

import robot_data
from envs.panda_grasp_env import PandaGraspGymEnv

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    panda_env = PandaGraspGymEnv(urdf_root=robot_data.getDataPath(), is_rendering=True, use_ik=True, is_discrete=True,
                                 num_controlled_joints=7, is_target_position_fixed=True)
    env = DummyVecEnv([lambda: panda_env])

    model = DQN.load("fixed_pos_target_2_deepq.pkl")
    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        print("Step: {} Action: {}".format(i, action))
        obs, rewards, done, info = env.step(action)
        env.render(mode='human')


if __name__ == '__main__':
    main()
