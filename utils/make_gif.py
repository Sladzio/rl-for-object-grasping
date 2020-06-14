import inspect
import os

import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv

import object_data
from envs.panda_grasp_env import PandaGraspGymEnv

from stable_baselines import DQN, DDPG
import imageio
import argparse

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)

gif_dir = "gif/"


def main(model_name, algo, testRange, isTargetPositionFixed, isDiscrete):
    panda_env = PandaGraspGymEnv(urdfRoot=object_data.getDataPath(), isRendering=True, useIK=True, isDiscrete=isDiscrete,
                                 numControlledJoints=7, isTargetPositionFixed=isTargetPositionFixed)
    env = DummyVecEnv([lambda: panda_env])

    if algo == "DDPG":
        model = DDPG.load(model_name)
    else:
        model = DQN.load(model_name)
    obs = env.reset()

    images = []
    img = env.get_images()

    for i in range(testRange):
        images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        print("Step: {} Action: {}".format(i, action))
        obs, rewards, done, info = env.step(action)
        env.render(mode='human')
        img = env.get_images()

    os.makedirs(gif_dir, exist_ok=True)
    imageio.mimsave(gif_dir + model_name + '.gif', [np.array(img[0]) for i, img in enumerate(images) if i % 2 == 0], fps=29)


if __name__ == '__main__':
    """ Makes GIFs, example usage make_gif.py -a DDPG -p True -d False"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Name for model to load',         default='best_model.pkl', type=str)
    parser.add_argument('-a', '--algo', help='RL Algorithm',                    default='DQN', type=str, required=False, choices=['DQN', 'DDPG'])
    parser.add_argument('-l', '--len', help='Gif length (more = longer | defaut = 300)',               default=300, type=int, required=False)
    parser.add_argument('-p', '--pos', help='Bool for isTargetPositionFixed',   default=False, type=bool, required=False)
    parser.add_argument('-d', '--discrete', help='Bool for isDiscrete',         default=True, type=bool, required=False)
    args = parser.parse_args()

    main(args.model, args.algo, args.len, args.pos, args.discrete)
