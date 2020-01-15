import inspect
import os
import robot_data
from envs.panda_grasp_env import PandaGraspGymEnv

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    panda_env = PandaGraspGymEnv(urdf_root=robot_data.getDataPath(), is_rendering=True, use_ik=True, is_discrete=True,
                                 num_controlled_joints=7, is_target_position_fixed=True)
    panda_env.render(mode='human')
    obs = panda_env.get_extended_observation()
    panda_env._panda.apply_action([.2, .27, -0.1999, 0, 0.1, 0, 1], False)

    panda_env._p.stepSimulation()
    panda_env._compute_reward()
    panda_env.perform_grasp()
    while (True):
        panda_env._p.stepSimulation()


if __name__ == '__main__':
    main()
