from stable_baselines.common.callbacks import BaseCallback

import tensorflow as tf
import numpy as np
import os
from stable_baselines.results_plotter import load_results, ts2xy


class MeanHundredEpsTensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        self.is_tb_set = False
        self.log_dir = log_dir
        super(MeanHundredEpsTensorboardCallback, self).__init__(verbose)

    def _init_callback(self) -> None:
        # Create folder if needed
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
            # Mean training reward over the last 100 episodes
            value = np.mean(y[-100:])
            summary = tf.Summary(value=[tf.Summary.Value(tag='mean_100_episode_reward', simple_value=value)])
            self.locals['writer'].add_summary(summary, len(x))
        return True

class StdHundredEpsTensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        self.is_tb_set = False
        self.log_dir = log_dir
        super(StdHundredEpsTensorboardCallback, self).__init__(verbose)

    def _init_callback(self) -> None:
        # Create folder if needed
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
            # Mean training reward over the last 100 episodes
            value = np.nanstd(y[-100:])
            summary = tf.Summary(value=[tf.Summary.Value(tag='std_100_episode_reward', simple_value=value)])
            self.locals['writer'].add_summary(summary, len(x))
        return True

class SuccessRateTensorboardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        self.is_tb_set = False
        self.log_dir = log_dir
        super(SuccessRateTensorboardCallback, self).__init__(verbose)

    def _init_callback(self) -> None:
        # Create folder if needed
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        results = load_results(self.log_dir)
        success_list = results["s"]
        success_rate = np.sum(success_list[-100:]) / 100.0

        summary = tf.Summary(value=[tf.Summary.Value(tag='success_rate', simple_value=success_rate)])
        self.locals['writer'].add_summary(summary, len(success_list))
        return True

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model:
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, eval_episodes_num=100,  verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.eval_episodes_num = eval_episodes_num
        self.save_path = os.path.join(log_dir, 'best_model_from_callback')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.nanmean(y[-self.eval_episodes_num:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward >= self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True