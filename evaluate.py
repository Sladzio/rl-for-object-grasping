import os
import inspect
import argparse
import numpy as np

from stable_baselines.her import HERGoalEnvWrapper
from stable_baselines.common.evaluation import evaluate_policy

import object_data
from envs.panda_grasp_env import PandaGraspGymEnv
from utils import ALGOS

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def get_environment(_max_step_count=500, _additional_reward=9500, _min_horizontal_distance_reward=0.015,
                    _lock_rotation=True, _is_discrete=False, _should_eval=True, _should_render=False):
    env = PandaGraspGymEnv(urdf_root=object_data.getDataPath(),
                           is_rendering=(not _should_eval) or _should_render,
                           use_ik=True,
                           is_discrete=_is_discrete,
                           num_controlled_joints=7,
                           lock_rotation=_lock_rotation,
                           max_step_count=_max_step_count,
                           additional_reward=_additional_reward,
                           min_horizontal_distance_reward=_min_horizontal_distance_reward,
                           reward_type='dense',
                           draw_workspace=True)
    return env


def main(_algo_name, _trained_models_dir, _trained_model_name, _lock_rotation,
         _should_eval, _eval_num_episodes, _should_render):

    is_discrete = True if _algo_name == 'DQN' else False

    eval_env = HERGoalEnvWrapper(get_environment(_lock_rotation=_lock_rotation, _is_discrete=is_discrete,
                                                  _should_eval=_should_eval, _should_render=_should_render))

    _trained_models_dir = _trained_models_dir if _trained_models_dir.endswith('/') else _trained_models_dir + '/'

    model = ALGOS[_algo_name].load(_trained_models_dir + _trained_model_name)

    if _should_eval:
        episode_rewards, episode_lengths, episode_success = evaluate_policy(model=model,
                                                                            env=eval_env,
                                                                            n_eval_episodes=_eval_num_episodes,
                                                                            render=(not _should_eval) or _should_render,
                                                                            deterministic=True,
                                                                            return_episode_rewards=True)

        print("Final evaluation for DDPG algorithm on {} episodes: "
              "\nReward: \n \tMEAN: {}, \tSTD: {}, \nEpisode Length: \n \tMEAN: {}, \tSTD: {}, \nSuccess Rate: {}"
              .format(_eval_num_episodes,
                      np.mean(episode_rewards),
                      np.std(episode_rewards),
                      np.mean(episode_lengths),
                      np.std(episode_lengths),
                      np.mean(episode_success)))

    else:
        obs = eval_env.reset()
        for i in range(2000):
            action, _states = model.predict(obs)
            obs, rewards, done, info = eval_env.step(action)
            eval_env.render(mode='human')
            if done:
                obs = eval_env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', help='RL Algorithm', default='DQN',
                        type=lambda x: str(x).upper(), required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-m', '--model', help='Name for a model to test and evaluate',
                        default='best_model.zip', type=str, required=False)
    parser.add_argument('-d', '--dir', help='Directory in which model is located - Default: BestModels | Example: '
                                            'TD3/TD3_LOCKET_ROT_TUNED/models/',
                        default='BestModels/', type=str, required=False)
    parser.add_argument('-l', '--lockRot', help='Should lock rotation of targeted object in evaluated environment '
                                                '- Default: True',
                        required=False, default=True, type=lambda x: (str(x).lower() == 'true'), choices=[True, False])

    parser.add_argument('-e', '--eval', help='Should perform evaluation on model - Default: True', required=False,
                        default=True, type=lambda x: (str(x).lower() == 'true'), choices=[True, False])
    parser.add_argument('-n', '--evalNum', help='Number of episodes to use for evaluation (if eval set to True)',
                        default=100, type=int, required=False)
    parser.add_argument('-r', '--render', help='Should render images in human mode, only if eval=True', required=False,
                        default=True, type=lambda x: (str(x).lower() == 'true'), choices=[True, False])
    args = parser.parse_args()

    main(_algo_name=str(args.algo).upper(), _trained_models_dir=args.dir, _trained_model_name=args.model,
         _lock_rotation=args.lockRot, _should_eval=args.eval,
         _eval_num_episodes=args.evalNum, _should_render=args.render)

