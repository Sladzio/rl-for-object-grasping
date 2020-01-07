import inspect
import os
import robot_data
from envs.panda_grasp_env import PandaGraspGymEnv

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(current_dir)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)


def main():
    panda_env = PandaGraspGymEnv(urdfRoot=robot_data.getDataPath(), isRendering=True, useIK=True, isDiscrete=True,
                                 numControlledJoints=7, isTargetPositionFixed=True)
    panda_env.render(mode='human')
    obs = panda_env.getExtendedObservation()
    panda_env._panda.apply_action([.03, 0, -0.22, 0, 0, 0, 1], False)

    panda_env._p.stepSimulation()
    panda_env.perform_grasp()
    while (True):
        panda_env._p.stepSimulation()


if __name__ == '__main__':
    main()
