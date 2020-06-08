import object_data
from envs import PandaGraspGymEnv
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
import numpy as np
import os
from CustomMonitor import CustomMonitor
from stable_baselines.common.callbacks import CheckpointCallback
from custom_callbacks import MeanHundredEpsTensorboardCallback, SuccessRateTensorboardCallback
from stable_baselines.her import HERGoalEnvWrapper
import argparse
import yaml

algorithm_name = "DDPG"
best_mean_reward, n_steps = -np.inf, 0


def get_environment(_max_step_count=500, _additional_reward=9500,
                    _min_horizontal_distance_reward=0.015, _lock_rotation=True):
    env = PandaGraspGymEnv(urdf_root=object_data.getDataPath(),
                           is_rendering=False,
                           use_ik=True,
                           is_discrete=False,
                           num_controlled_joints=7,
                           lock_rotation=_lock_rotation,
                           max_step_count=_max_step_count,
                           additional_reward=_additional_reward,
                           min_horizontal_distance_reward=_min_horizontal_distance_reward,
                           reward_type='dense')
    return env


def main(_ddpg_tag, _tagSuffix, _saveFreq, _lock_rotation, hyperparams):
    full_tag = algorithm_name + "_" + _ddpg_tag + _tagSuffix
    current_dir = algorithm_name + "/" + full_tag
    log_dir = current_dir + "/log/"
    eval_log_dir = current_dir + "/log/eval/"
    trained_models_dir = current_dir + "/trainedModels/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(trained_models_dir, exist_ok=True)

    panda_env = HERGoalEnvWrapper(CustomMonitor(get_environment(_lock_rotation=_lock_rotation), log_dir))
    eval_env = HERGoalEnvWrapper(CustomMonitor(get_environment(_lock_rotation=_lock_rotation), eval_log_dir))

    callbacks = []
    callbacks.append(CheckpointCallback(_saveFreq, trained_models_dir)) if _saveFreq > 0 else None
    callbacks.append(MeanHundredEpsTensorboardCallback(log_dir))
    callbacks.append(SuccessRateTensorboardCallback(log_dir))

    time_steps = hyperparams.pop('n_timesteps') if hyperparams.get('n_timesteps') is not None else None

    param_noise = None
    action_noise = None
    if hyperparams.get('noise_type') is not None:
        noise_type = hyperparams.pop('noise_type').strip()
        if 'ornstein-uhlenbeck' in noise_type:
            n_actions = panda_env.action_space.shape[-1]
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                        sigma=float(0.005) * np.ones(n_actions))
        elif 'param_noise' in noise_type:
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)

    model = DDPG(env=panda_env,
                 eval_env=eval_env,
                 param_noise=param_noise,
                 action_noise=action_noise,
                 tensorboard_log="tensorboard/",
                 n_cpu_tf_sess=None,
                 **hyperparams)

    model.learn(total_timesteps=time_steps,
                callback=callbacks,
                tb_log_name=full_tag,
                log_interval=10)

    model.save(current_dir + "/" + full_tag + "_final")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', help='Name of configuration tag used for algorithm parameters '
                                            '[CLEAN, TUNED, ACTION_NOISE] \n Default: ACTION_NOISE',
                        default='ACTION_NOISE', type=str, required=False)
    parser.add_argument('-s', '--suf', help='Suffix added for nametag of trained model',
                        default='', type=str, required=False)
    parser.add_argument('-l', '--lockRot', help='Should lock rotation of targeted object '
                                               '\n Default: True', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--saveFreq', help='Save the model every n steps (if negative, no checkpoint)',
                        default=25000, type=int)
    args = parser.parse_args()

    # Load hyperparameters from yaml file
    with open('hyperparams/{}.yml'.format(algorithm_name), 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if args.tag in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[args.tag]
        else:
            raise ValueError("{} Hyperparameters not found for {}".format(args.tag, algorithm_name))

    main(_ddpg_tag=str(args.tag).upper(), _tagSuffix=args.suf, _saveFreq=args.saveFreq, _lock_rotation=args.lockRot, hyperparams=hyperparams)
