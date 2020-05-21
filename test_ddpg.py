import inspect
import os

import numpy as np

import object_data
from envs.panda_grasp_env import PandaGraspGymEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.her import HERGoalEnvWrapper
import argparse

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main(_model_name, _trained_models_dir, _should_eval, _eval_num_episodes, _should_render):

    panda_env = PandaGraspGymEnv(urdf_root=object_data.getDataPath(),
                                 is_rendering=(not _should_eval) or _should_render,
                                 use_ik=True,
                                 is_discrete=False,
                                 num_controlled_joints=7,
                                 reward_type="dense",
                                 draw_workspace=True)

    model = DDPG.load(_trained_models_dir + "/" + _model_name)

    env = HERGoalEnvWrapper(panda_env)

    if _should_eval:
        episode_rewards, episode_lengths, episode_success = evaluate_policy(model, env,
                                                                            n_eval_episodes=_eval_num_episodes,
                                                                            render=(not _should_eval) or _should_render,
                                                                            deterministic=True,
                                                                            return_episode_rewards=True)
        print("Final Reward DDPG: {}, Episodes: {}, Episode Length: {}, Success Rate: {}"
              .format(
                np.mean(episode_rewards),
                _eval_num_episodes,
                np.mean(episode_lengths),
                np.mean(episode_success)))

    else:
        obs = env.reset()
        for i in range(2000):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render(mode='human')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Name for a model to test',
                        default='best_model.zip', type=str, required=False)
    parser.add_argument('-d', '--dir', help='Directory in which model is located',
                        default='trainedModels', type=str, required=False)
    parser.add_argument('-e', '--eval', help='Should perform evaluation on model [True/False]', required=False,
                        default=True, type=lambda x: ((str(x).lower() == 'true') or (str(x).lower() == 'True')))
    parser.add_argument('-n', '--num', help='Number of episodes for evaluation (if eval set to True)',
                        default=100, type=int, required=False)
    parser.add_argument('-r', '--render', help='Should render images in human mode, only if eval=True', required=False,
                        default=True, type=lambda x: ((str(x).lower() == 'true') or (str(x).lower() == 'True')))
    args = parser.parse_args()

    main(args.model, args.dir, args.eval, args.num, args.render)
