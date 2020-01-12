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
                 useIK=True,
                 isDiscrete=0,
                 actionRepeatAmount=1,
                 isRendering=False,
                 maxSteps=1000,
                 isTargetPositionFixed=False,
                 numControlledJoints=7,
                 isContinuousDownwardEnabled=False):

        self._isContinuousDownwardEnabled = isContinuousDownwardEnabled
        self._attemptedGrasp = False
        self._numControlledJoints = numControlledJoints
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._useIK = useIK
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeatAmount
        self._observation = []
        self._envStepCounter = 0
        self._renders = isRendering
        self._maxSteps = maxSteps
        self.terminated = False
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._h_table = []
        self._target_dist_min = 0.075
        self._p = p
        self._fixedPositionObj = isTargetPositionFixed

        if self._isDiscrete:
            self.action_space = 13
        else:
            self.action_space = self._numControlledJoints

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.5, 125, -30, [0.52, -0.2, 0.33])
        else:
            p.connect(p.DIRECT)
        self.reset()
        observationDim = len(self.getExtendedObservation())
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype='float32')

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(self.action_space)

        else:
            # self.action_dim = 2 #self._panda.getActionDimension()
            self._action_bound = 1
            action_high = np.array([self._action_bound] * self.action_space)
            self.action_space = spaces.Box(-action_high, action_high, dtype='float32')

        self.viewer = None

    def reset(self):
        self.terminated = False
        self._attemptedGrasp = False
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        self._envStepCounter = 0

        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), useFixedBase=True)
        # Load robot
        self._panda = PandaEnv(self._urdfRoot, timeStep=self._timeStep, basePosition=[0, 0, 0.625],
                               useInverseKinematics=self._useIK, numControlledJoints=self._numControlledJoints)

        # Load table and object for simulation
        tableId = p.loadURDF(os.path.join(self._urdfRoot, "franka/table.urdf"), useFixedBase=True)

        table_info = p.getVisualShapeData(tableId, -1)[0]
        self._h_table = table_info[5][2] + table_info[3][2] + 0.01

        # limit panda workspace to table plane
        self._panda.workspace_lim[2][0] = self._h_table
        # Randomize start position of object and target.

        if self._fixedPositionObj:
            self.targetObjectId = p.loadURDF(os.path.join(self._urdfRoot, "franka/cube_small.urdf"),
                                             basePosition=[0.7, 0.25, self._h_table], useFixedBase=False,
                                             globalScaling=.5)
        else:
            self.target_pose = self._sample_pose()[0]
            self.targetObjectId = p.loadURDF(os.path.join(self._urdfRoot, "franka/cube_small.urdf"),
                                             basePosition=self.target_pose, useFixedBase=False, globalScaling=.5)

        p.setGravity(0, 0, -9.8)
        # Let the world run for a bit
        p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def getExtendedObservation(self):

        # get robot observations
        self._observation = self._panda.getObservation()
        gripperState = p.getLinkState(self._panda.pandaId, self._panda.gripperIndex)
        gripperPos = gripperState[0]
        gripperOrn = gripperState[1]
        invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)

        targetObjPos, targetObjOrn = p.getBasePositionAndOrientation(self.targetObjectId)
        targetObjPosInGripperSpace, targetObjOrnInGripperSpace = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                                      targetObjPos, targetObjOrn)
        targetObjEulerInGripperSpace = p.getEulerFromQuaternion(targetObjOrnInGripperSpace)

        # we return the relative x,y position and euler angle of block in gripper space
        targetObjRelativeToGripper = [targetObjPosInGripperSpace[0], targetObjPosInGripperSpace[1],
                                      targetObjEulerInGripperSpace[2]]

        self._observation.extend(targetObjRelativeToGripper)
        # self._observation.extend(list(target_obj_pos))
        # self._observation.extend(list(target_obj_orn))

        return self._observation

    def step(self, action):
        if self._isDiscrete:
            dv = 0.001
            dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0, 0, 0][action]
            if self._isContinuousDownwardEnabled:
                dz = -dv
            else:
                dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
            droll = [0, 0, 0, 0, 0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
            dpitch = [0, 0, 0, 0, 0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
            dyaw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dv, dv][action]
            f = 1
            real_action = [dx, dy, dz, droll, dpitch, dyaw, f]
            return self.real_step(real_action)
        else:
            dv = 1.5
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

        closestPoints = p.getClosestPoints(self.targetObjectId, self._panda.pandaId, 0.035, -1,
                                           self._panda.gripperIndex)

        if len(closestPoints) > 0:
            self.perform_grasp()

        reward = self._compute_reward()

        done = self._termination()

        return np.array(self._observation), reward, done, {}

    def _termination(self):
        return self._attemptedGrasp or self._envStepCounter >= self._maxSteps

    def perform_grasp(self):
        anim_length = 100
        self._panda.updateGripPos()
        self.close_fingers(anim_length)
        self.lift_gripper(anim_length)
        self._attemptedGrasp = True

    def close_fingers(self, anim_length):
        finger_angle = 1
        for i in range(anim_length):
            grasp_action = [0, 0, 0, 0, 0, 0, finger_angle]
            self._panda.apply_action(grasp_action)
            p.stepSimulation()
            finger_angle -= 1 / anim_length;
            if finger_angle < 0:
                finger_angle = 0

    def lift_gripper(self, anim_length):
        for i in range(anim_length):
            grasp_action = [0, 0, 0.005, 0, 0, 0, 0]
            self._panda.apply_action(grasp_action)
            p.stepSimulation()
            block_pos, block_orn = p.getBasePositionAndOrientation(self.targetObjectId)
            if block_pos[2] > 0.8:
                break
            state = p.getLinkState(self._panda.pandaId, self._panda.gripperIndex)
            actual_end_effector_pos = state[0]
            if actual_end_effector_pos[2] > 2:
                break

    def _compute_reward(self):
        target_obj_pos, target_obj_orn = p.getBasePositionAndOrientation(self.targetObjectId)

        closestPoints = p.getClosestPoints(self.targetObjectId, self._panda.pandaId, 1000, -1,
                                           self._panda.gripperIndex)

        reward = -1000

        numPt = len(closestPoints)

        if numPt > 0:

            pointA = np.array([closestPoints[0][5][0], closestPoints[0][5][1], 0])
            pointB = np.array([closestPoints[0][6][0], closestPoints[0][6][1], 0])

            reward = 0
            horizontal_distance = np.linalg.norm(pointA - pointB)
            reward -= horizontal_distance * 10
            if horizontal_distance < 0.05:
                reward -= closestPoints[0][8] * 10
            else:
                reward -= 10
        if target_obj_pos[2] > 0.7:
            reward = 10000
            print("successfully grasped a block!!!")
        return reward

    def _sample_pose(self):
        ws_lim = self._panda.workspace_lim
        px, tx = np.random.uniform(low=ws_lim[0][0], high=ws_lim[0][1], size=2)
        py, ty = np.random.uniform(low=ws_lim[1][0], high=ws_lim[1][1], size=2)
        pz, tz = self._h_table, self._h_table
        obj_pose = [px, py, pz]
        tg_pose = [tx, ty, tz]

        return obj_pose, tg_pose

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
