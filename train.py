import robot_data
from envs import PandaGraspGymEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
import numpy as np


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    # print("totalt")
    # print(totalt)
    is_solved = totalt > 2000 and total >= 10
    return is_solved


def main():
    batch_size = 16
    panda_env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), renders=False, useIK=False, isDiscrete=False,
                                 numControlledJoints=7)
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
                 action_noise=action_noise, tensorboard_log="../pybullet_logs/pandareach_ddpg/", reward_scale=1)
    model.learn(total_timesteps=10000000)
    model.save("model.pkl")


if __name__ == '__main__':
    main()
