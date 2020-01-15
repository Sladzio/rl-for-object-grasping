import inspect
import os

import robot_data
from envs.panda_grasp_env import PandaGraspGymEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    panda_env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), isRendering=True, useIK=True, isDiscrete=False,
                                 numControlledJoints=7)

    env = DummyVecEnv([lambda: panda_env])

    model = SAC.load("tmp/best_model.pkl")
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='human')


if __name__ == '__main__':
    main()
