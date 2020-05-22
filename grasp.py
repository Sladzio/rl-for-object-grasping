import inspect
import os
import object_data
from envs.panda_grasp_env import PandaGraspGymEnv
import numpy as np
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    panda_env = PandaGraspGymEnv(urdf_root=object_data.getDataPath(), is_rendering=True, use_ik=True, is_discrete=True,
                                 num_controlled_joints=7, draw_workspace=True)
    panda_env.render(mode='human')
    obs = panda_env.get_observation()
    xMin = panda_env._table_workspace_shape[0][0]
    xMax = panda_env._table_workspace_shape[0][1]
    yMin = panda_env._table_workspace_shape[1][0]
    yMax = panda_env._table_workspace_shape[1][1]
    panda_env._panda.apply_action([0, 0, -0.30, 0, 0, np.pi/2, 1], False)
    panda_env.step(0)
    while (True):
        panda_env._p.stepSimulation()


if __name__ == '__main__':
    main()
