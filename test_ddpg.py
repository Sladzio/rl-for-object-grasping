import inspect
import os

import numpy as np

import robot_data
from envs.panda_grasp_env import PandaGraspGymEnv
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    panda_env = PandaGraspGymEnv(urdf_root=robot_data.getDataPath(), is_rendering=True, use_ik=True, is_discrete=False,
                                 num_controlled_joints=7)

    env = DummyVecEnv([lambda: panda_env])
    param_noise = None
    n_actions = panda_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = DDPG(MlpPolicy, panda_env, verbose=1, param_noise=param_noise, action_noise=action_noise,
                 tensorboard_log="tensorboard/")
    model.load("tmp/best_model.pkl")
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='human')


if __name__ == '__main__':
    main()
