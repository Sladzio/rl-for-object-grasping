from stable_baselines.common.callbacks import BaseCallback

import tensorflow as tf
import numpy as np
import os
from stable_baselines.results_plotter import load_results, ts2xy


class MeanHundredEpsCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        self.is_tb_set = False
        self.log_dir = log_dir
        super(MeanHundredEpsCallback, self).__init__(verbose)

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
