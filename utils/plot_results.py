from utils import plot_rewards

dqn_rot_clean = ("DQN_ROT_CLEAN", "../BestModels/DQN/DQN_ROT_CLEAN/log")
dqn_rot_tuned = ("DQN_ROT_TUNED", "../BestModels/DQN/DQN_ROT_CLEAN/log")
dqn_rot_noise = ("DQN_ROT_NOISE", "../BestModels/DQN/DQN_ROT_CLEAN/log")

plot_rewards([dqn_rot_clean, dqn_rot_tuned, dqn_rot_noise], title="Rotation Enabled DQN Learning Curve")


dqn_rot_clean = ("DQN_LOCKED_ROT_CLEAN", "../BestModels/DQN/DQN_LOCKED_ROT_CLEAN/log")
dqn_rot_tuned = ("DQN_LOCKED_ROT_TUNED", "../BestModels/DQN/DQN_ROT_CLEAN/log")
dqn_rot_noise = ("DQN_LOCKED_ROT_NOISE", "../BestModels/DQN/DQN_ROT_CLEAN/log")

plot_rewards([dqn_rot_clean, dqn_rot_tuned, dqn_rot_noise], title="Locked Rotation DQN Learning Curve")
