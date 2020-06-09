from plotter import plot_rewards

dqn_rot_clean = ("DQN_ROT_CLEAN", "/Users/sladek-patryk/rl-for-object-grasping/evaluation-runs/DQN_ROT_CLEAN/log")
dqn_rot_tuned = ("DQN_ROT_TUNED", "/Users/sladek-patryk/rl-for-object-grasping/evaluation-runs/DQN_ROT_TUNED/log")
dqn_rot_noise = ("DQN_ROT_NOISE", "/Users/sladek-patryk/rl-for-object-grasping/evaluation-runs/DQN_ROT_NOISE/log")

plot_rewards([dqn_rot_clean, dqn_rot_tuned, dqn_rot_noise], title="Rotation Enabled DQN Learning Curve")


dqn_rot_clean = ("DQN_LOCKED_ROT_CLEAN", "/Users/sladek-patryk/rl-for-object-grasping/evaluation-runs/DQN_LOCKED_ROT_CLEAN/log")
dqn_rot_tuned = ("DQN_LOCKED_ROT_TUNED", "/Users/sladek-patryk/rl-for-object-grasping/evaluation-runs/DQN_LOCKED_ROT_TUNED/log")
dqn_rot_noise = ("DQN_LOCKED_ROT_NOISE", "/Users/sladek-patryk/rl-for-object-grasping/evaluation-runs/DQN_LOCKED_ROT_NOISE/log")

plot_rewards([dqn_rot_clean, dqn_rot_tuned, dqn_rot_noise], title="Locked Rotation DQN Learning Curve")