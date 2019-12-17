import robot_data
from envs import PandaGraspGymEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    # print("totalt")
    # print(totalt)
    is_solved = totalt > 2000 and total >= 10
    return is_solved


def main():
    panda_env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), renders=False, useIK=True, isDiscrete=True,
                                 numControlledJoints=7)
    # env = DummyVecEnv([lambda: panda_env])
    # model = PPO2(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=10000)

    # obs = env.reset()
    for i in range(1000):
        panda_env.render()


if __name__ == '__main__':
    main()
