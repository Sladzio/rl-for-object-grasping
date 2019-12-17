import inspect
import os
import robot_data
from envs.panda_grasp_env import PandaGraspGymEnv

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


def main():
    panda_env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, useIK=True, isDiscrete=True,
                                 numControlledJoints=7)

    env = DummyVecEnv([lambda: panda_env])
    model = PPO2(MlpPolicy, env, verbose=1)
    model.load("model.pkl")
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='human')


if __name__ == '__main__':
    main()
