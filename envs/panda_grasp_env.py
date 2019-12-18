import inspect
import os
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

import robot_data
from . import PandaEnv

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentDir)
os.sys.path.insert(0, currentDir)

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PandaGraspGymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self, urdfRoot=robot_data.getDataPath(),
                 useIK=0,
                 isDiscrete=0,
                 actionRepeat=1,
                 renders=False,
                 maxSteps=1000,
                 dist_delta=0.03,
                 fixedPositionObj=False,
                 includeVelObs=True,
                 numControlledJoints=7):

        self.numControlledJoints = numControlledJoints
        self._isDiscrete = isDiscrete
        if self._isDiscrete:
            self.action_space = 13
        else:
            self.action_space = self.numControlledJoints
        self._timeStep = 1. / 240.
        self._useIK = useIK
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = False
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._h_table = []
        self._target_dist_max = 0.3
        self._target_dist_min = 0.2
        self._p = p
        self.fixedPositionObj = fixedPositionObj
        self.includeVelObs = includeVelObs

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2.5, 90, -60, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        self.reset()
        observationDim = len(self.getExtendedObservation())
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype='float32')

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(self._panda.getActionDimension())

        else:
            # self.action_dim = 2 #self._panda.getActionDimension()
            self._action_bound = 1
            action_high = np.array([self._action_bound] * self.action_space)
            self.action_space = spaces.Box(-action_high, action_high, dtype='float32')

        self.viewer = None

    def reset(self):
        self.terminated = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        self._envStepCounter = 0

        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), useFixedBase=True)
        # Load robot
        self._panda = PandaEnv(self._urdfRoot, timeStep=self._timeStep, basePosition=[0, 0, 0.625],
                               useInverseKinematics=self._useIK, numControlledJoints=self.numControlledJoints,
                               includeVelObs=self.includeVelObs)

        # Load table and object for simulation
        tableId = p.loadURDF(os.path.join(self._urdfRoot, "franka/table.urdf"), useFixedBase=True)

        table_info = p.getVisualShapeData(tableId, -1)[0]
        self._h_table = table_info[5][2] + table_info[3][2]

        # limit panda workspace to table plane
        self._panda.workspace_lim[2][0] = self._h_table
        # Randomize start position of object and target.

        if (self.fixedPositionObj):
            self._objID = p.loadURDF(os.path.join(self._urdfRoot, "franka/cube_small.urdf"),
                                     basePosition=[0.7, 0.0, 0.64], useFixedBase=True)
        else:
            self.target_pose = self._sample_pose()[0]
            self._objID = p.loadURDF(os.path.join(self._urdfRoot, "franka/cube_small.urdf"),
                                     basePosition=self.target_pose, useFixedBase=False)

        self._debugGUI()
        p.setGravity(0, 0, -9.8)
        # Let the world run for a bit
        p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def getExtendedObservation(self):

        # get robot observations
        self._observation = self._panda.getObservation()
        target_obj_pos, target_obj_orn = p.getBasePositionAndOrientation(self._objID)

        self._observation.extend(list(target_obj_pos))
        self._observation.extend(list(target_obj_orn))
        return self._observation

    def step(self, action):
        if self._isDiscrete:
            dv = 0.01
            dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0][action]
            dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
            droll = [0, 0, 0, 0, 0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
            dpitch = [0, 0, 0, 0, 0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
            dyaw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dv, dv][action]
            f = 1
            real_action = [dx, dy, dz, droll, dpitch, dyaw, f]
            return self.real_step(real_action)
        else:
            dv = 1
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[2] * dv
            droll = action[3] * dv
            dpitch = action[4] * dv
            dyaw = action[5] * dv
            f = 1
            real_action = [dx, dy, dz, droll, dpitch, dyaw, f]
            return self.real_step(real_action)

    def real_step(self, action):

        for i in range(self._actionRepeat):
            self._panda.apply_action(action)
            p.stepSimulation()

            if self._termination():
                break

            self._envStepCounter += 1

        if self._renders:
            time.sleep(self._timeStep)

        self._observation = self.getExtendedObservation()

        reward = self._compute_reward()

        done = self._termination()

        return np.array(self._observation), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._panda.pandaId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):

        end_eff_pos = self._panda.getObservation()[0:3]
        target_obj_pos, target_obj_orn = p.getBasePositionAndOrientation(self._objID)
        d = goal_distance(np.array(target_obj_pos), np.array(end_eff_pos))

        if d <= self._target_dist_min or self._envStepCounter > self._maxSteps:
            self.terminated = True
            self.perform_grasp()
            self._observation = self.getExtendedObservation()
            return True

        return False

    def perform_grasp(self):
        finger_angle = 0.3
        for i in range(100):
            grasp_action = [0, 0, 0.0001, 0, 0, 0, finger_angle]
            self._panda.apply_action(grasp_action)
            p.stepSimulation()
            finger_angle = finger_angle - (0.3 / 100.)
            if finger_angle < 0:
                finger_angle = 0
        for i in range(1000):
            grasp_action = [0, 0, 0.001, 0, 0, 0, finger_angle]
            self._panda.apply_action(grasp_action)
            p.stepSimulation()
            block_pos, block_orn = p.getBasePositionAndOrientation(self._objID)
            if block_pos[2] > 0.23:
                break
            state = p.getLinkState(self._panda.pandaId, self._panda.endEffLink)
            actual_end_effector_pos = state[0]
            if actual_end_effector_pos[2] > 0.5:
                break

    def _compute_reward(self):

        blockPos, blockOrn = p.getBasePositionAndOrientation(self._objID)
        closestPoints = p.getClosestPoints(self._objID, self._panda.pandaId, 1000, -1,
                                           self._panda.endEffLink)

        reward = -1000

        numPt = len(closestPoints)
        # print(numPt)
        if (numPt > 0):
            # print("reward:")
            reward = -closestPoints[0][8] * 10
        if (blockPos[2] > 0.8):
            reward = reward + 10000
            print("successfully grasped a block!!!")
            # print("self._envStepCounter")
            # print(self._envStepCounter)
            # print("self._envStepCounter")
            # print(self._envStepCounter)
            # print("reward")
            # print(reward)
        # print("reward")
        # print(reward)
        return reward

        # target_obj_pos, target_obj_orn = p.getBasePositionAndOrientation(self._objID)
        # end_effector_pos = self._panda.getObservation()[0:3]
        # d = goal_distance(np.array(end_effector_pos), np.array(target_obj_pos))
        # reward = -d
        # if d <= self._target_dist_min:
        #     reward = np.float32(1000.0) + (100 - d * 80)
        # return reward

    def _sample_pose(self):
        ws_lim = self._panda.workspace_lim
        px, tx = np.random.uniform(low=ws_lim[0][0], high=ws_lim[0][1], size=(2))
        py, ty = np.random.uniform(low=ws_lim[1][0], high=ws_lim[1][1], size=(2))
        pz, tz = self._h_table, self._h_table
        obj_pose = [px, py, pz]
        tg_pose = [tx, ty, tz]

        return obj_pose, tg_pose

    def _debugGUI(self):
        # TO DO
        return 0
