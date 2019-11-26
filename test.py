import inspect
import os
import robot_data
from envs.panda_grasp_env import PandaGraspGymEnv

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), renders=True, useIK=True, isDiscrete=False)

    while True:
        obs, done = env.reset(), False
        print("===================================")
        print("obs")
        print(obs)
        episode_rew = 0
        while not done:
            env.render(mode='human')
            obs, rew, done, _ = env.step([0.5, 0.5, 0.5,
                                          0, 0, 0,
                                          1])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
