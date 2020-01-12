import inspect
import os

import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv

import robot_data
from envs.panda_grasp_env import PandaGraspGymEnv

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy,LnMlpPolicy

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    panda_env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), isRendering=True, useIK=True, isDiscrete=True,
                                 numControlledJoints=7,isTargetPositionFixed=True)
    env = DummyVecEnv([lambda: panda_env])

    model = DQN.load("tmp/best_model.pkl")
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs,deterministic=False)
        print("Step: {} Action: {}".format(i,action))
        obs, rewards, done, info = env.step(action)
        env.render(mode='human')


if __name__ == '__main__':
    main()