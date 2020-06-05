import os
import time
from collections import OrderedDict

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
import object_data
from . import PandaEnv


class PandaGraspGymEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self,
                 urdf_root=object_data.getDataPath(),
                 use_ik=True,
                 is_discrete=0,
                 action_repeat_amount=1,
                 is_rendering=False,
                 max_step_count=500,
                 additional_reward=9000,
                 num_controlled_joints=7,
                 is_continuous_downward_enabled=False,
                 reward_type='dense', draw_workspace=False, lock_rotation=True):

        self._episode_number = 1
        self._grasp_attempts_count = 0
        self._largest_observation_value = 1
        self._largest_action_value = 1
        self._img_height = 720
        self._img_width = 960
        self._distance_to_start_grasp = 0.11
        self._is_continuous_downward_enabled = is_continuous_downward_enabled
        self._attempted_grasp = False
        self._num_controlled_joints = num_controlled_joints
        self._is_discrete = is_discrete
        self._time_step = 1. / 240.
        self._use_ik = use_ik
        self._urdf_root = urdf_root
        self._action_repeat_amount = action_repeat_amount
        self._observation = {}
        self._env_step_counter = 0
        self._is_rendering = is_rendering
        self._max_step_count = max_step_count
        self._additional_reward = additional_reward
        self._terminated = False
        self._cam_dist = 1.8
        self._cam_yaw = 90
        self._cam_pitch = -20
        self._p = p
        self._successful_grasp_count = 0
        self._panda = None
        self._reward_type = reward_type
        self._target_object_id = None
        self._is_successful_grasp = False
        self.lift_distance = 0.04
        self._distance_threshold = 0.01
        self._table_workspace_shape = []
        self._draw_workspace = draw_workspace
        self._table_pos = [0, 0, 0]
        self._table_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        self._robot_start_x_offset = -0.35
        self._lock_rotation = lock_rotation

        if self._is_discrete:
            if self._lock_rotation:
                self.action_space = 7
            else:
                self.action_space = 13
        else:
            if self._lock_rotation:
                self.action_space = 3
            else:
                self.action_space = 4


        if self._is_rendering:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.reset()

        obs = self.get_observation()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        if self._is_discrete:
            self.action_space = spaces.Discrete(self.action_space)

        else:
            # self.action_space = spaces.Box(-1., 1., shape=(self.action_space,), dtype='float32')
            action_high = np.array([self._largest_action_value] * self.action_space)
            self.action_space = spaces.Box(-action_high, action_high, dtype='float32')

        if self._is_rendering:
            base_pos, orn = self._p.getBasePositionAndOrientation(self._panda.panda_id)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, base_pos)

    def reset(self):
        self._is_successful_grasp = False
        self._terminated = False
        self._attempted_grasp = False
        self._env_step_counter = 0
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, lightPosition=[0, 0, 10])

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)

        table_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), useFixedBase=True,
                              basePosition=self._table_pos, baseOrientation=self._table_orn)

        self._table_workspace_shape = self._compute_workspace(table_id)

        robot_base_pos = [self._robot_start_x_offset, 0, self._get_table_height()]

        if self._draw_workspace:
            self._draw_workspace_limits()

        self._panda = PandaEnv(robot_base_pos, self._urdf_root, time_step=self._time_step,
                               use_ik=self._use_ik, num_controlled_joints=self._num_controlled_joints)

        pos = self._get_random_position_on_table()

        if self._lock_rotation:
            orn = p.getQuaternionFromEuler([0, 0, 0])
        else:
            orn = self._get_random_z_orientation()

        self._target_object_id = p.loadURDF(os.path.join(self._urdf_root, "cube/cube.urdf"),
                                            basePosition=pos, baseOrientation=orn, useFixedBase=False,
                                            globalScaling=.5)

        self._goal_position = np.add(pos, [0, 0, self.lift_distance])
        p.setGravity(0, 0, -9.8)

        p.stepSimulation()

        self._observation = self.get_observation()

        return self._observation

    def _compute_workspace(self, table_id):
        table_shape = p.getVisualShapeData(table_id, -1)
        table_top_shape = table_shape[0]
        table_height = table_top_shape[5][2] + table_top_shape[3][2] / 2
        table_leg_pos = table_shape[1][5]
        # Rotation of the vector is needed
        # getVisualShapeData returns position of local visual frame, relative to link/joint frame
        table_leg_pos = p.rotateVector(self._table_orn, table_leg_pos)
        max_workspace_x = abs(table_leg_pos[0])
        max_workspace_y = abs(table_leg_pos[1])
        return [
            [-max_workspace_x + 0.3, max_workspace_x - 0.05],  # X
            [-max_workspace_y + 0.3, max_workspace_y - 0.3],  # Y
            [table_height, 1]  # Z
        ]

    def _get_table_height(self):
        return self._table_workspace_shape[2][0]

    def _get_random_position_on_table(self):
        ws_lim = self._table_workspace_shape
        x = np.random.uniform(low=ws_lim[0][0], high=ws_lim[0][1])
        y = np.random.uniform(low=ws_lim[1][0], high=ws_lim[1][1])
        z = self._get_table_height()
        obj_pose = [x, y, z]

        return obj_pose

    def _get_random_z_orientation(self):
        random_angle = np.random.uniform(low=0, high=2 * np.pi)
        euler_orn = [0, 0, random_angle]
        orn = p.getQuaternionFromEuler(euler_orn)
        return orn

    def get_observation(self):
        observation = self._panda.get_observation()
        target_obj_pos, target_obj_orn = p.getBasePositionAndOrientation(self._target_object_id)
        if not self._lock_rotation:
            observation.extend(list(target_obj_orn))
        achieved_goal = list(target_obj_pos)
        desired_goal = self._goal_position
        return {
            'observation': np.asarray(observation.copy()),
            'achieved_goal': np.asarray(achieved_goal.copy()),
            'desired_goal': np.asarray(desired_goal.copy())
        }

    def generate_action_array(self, action):
        if self._is_discrete:
            delta_pos = 0.01
            delta_angle = 0.01

            # Position
            dx = [0, -delta_pos, delta_pos, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -delta_pos, delta_pos, 0, 0, 0, 0, 0, 0, 0, 0][action]
            if self._is_continuous_downward_enabled:
                dz = -delta_pos
            else:
                dz = [0, 0, 0, 0, 0, -delta_pos, delta_pos, 0, 0, 0, 0, 0, 0][action]

            # Orientation
            droll = [0, 0, 0, 0, 0, 0, 0, -delta_angle, delta_angle, 0, 0, 0, 0][action]
            dpitch = [0, 0, 0, 0, 0, 0, 0, 0, 0, -delta_angle, delta_angle, 0, 0][action]
            dyaw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -delta_angle, delta_angle][action]

            # Gripper
            gripper_angle = 1

            return [dx, dy, dz, droll, dpitch, dyaw, gripper_angle]

        else:
            delta_pos = 0.01
            delta_angle = 0.075

            # Position
            dx = action[0] * delta_pos
            dy = action[1] * delta_pos
            dz = action[2] * delta_pos

            # Orientation
            # droll = action[3] * delta_angle
            # dpitch = action[4] * delta_angle

            if self._lock_rotation:
                dyaw = 0
            else:
                dyaw = action[3] * delta_angle

            # Gripper
            gripper_angle = 1

            return [dx, dy, dz, 0, 0, dyaw, gripper_angle]

    def step(self, action):
        action = self.generate_action_array(action)
        self._panda.apply_action(action)
        p.stepSimulation()
        self._env_step_counter += 1

        if self._is_rendering:
            time.sleep(self._time_step)

        self._observation = self.get_observation()

        vertical_distance = self.get_vertical_distance_to_target()

        if vertical_distance <= self._distance_to_start_grasp:
            self.perform_grasp()
            self._observation = self.get_observation()

        achieved_goal = self._observation['achieved_goal']
        desired_goal = self._observation['desired_goal']
        is_success = self._is_success(achieved_goal, desired_goal)
        if not is_success and self._attempted_grasp:
            self.open_fingers()
            self._attempted_grasp = False

        info = {'is_success': is_success}

        if is_success:
            self._is_successful_grasp = True
            self._successful_grasp_count += 1
            print(
                "Successfully grasped a block!!! Timestep: {} Episode: {}, Grasp Count: {} Attempted Grasps: {} ".format(
                    self._env_step_counter, self._episode_number, self._successful_grasp_count,
                    self._grasp_attempts_count))

        reward = self.compute_reward(achieved_goal, desired_goal, info)
        done = self._termination()
        if done:
            self._episode_number += 1

        return self._observation, reward, done, info

    def _termination(self):
        return self._is_successful_grasp or self._env_step_counter >= self._max_step_count

    def perform_grasp(self):
        anim_length = 100
        self._panda.update_gripper_pos()
        self.close_fingers(anim_length)
        self.lift_gripper(anim_length)
        self._attempted_grasp = True
        self._grasp_attempts_count += 1

    def open_fingers(self, anim_length=100):
        finger_angle = 0
        for i in range(anim_length):
            grasp_action = [0, 0, 0, 0, 0, 0, finger_angle]
            self._panda.apply_action(grasp_action)
            p.stepSimulation()
            finger_angle += 1 / anim_length
            if finger_angle > 1:
                finger_angle = 1

    def close_fingers(self, anim_length):
        finger_angle = 1
        for i in range(anim_length):
            grasp_action = [0, 0, 0, 0, 0, 0, finger_angle]
            self._panda.apply_action(grasp_action)
            p.stepSimulation()
            finger_angle -= 1 / anim_length
            if finger_angle < 0:
                finger_angle = 0

    def lift_gripper(self, anim_length):
        for i in range(anim_length):
            grasp_action = [0, 0, 0.005, 0, 0, 0, 0]
            self._panda.apply_action(grasp_action)
            p.stepSimulation()
            block_pos, block_orn = p.getBasePositionAndOrientation(self._target_object_id)
            if block_pos[2] > 0.8:
                break
            state = p.getLinkState(self._panda.panda_id, self._panda.gripper_index)
            actual_end_effector_pos = state[0]
            if actual_end_effector_pos[2] > 2.5:
                break

    def get_distance(self, point_a, point_b, axis):
        point_a = np.array([axis[0] * point_a[0], axis[1] * point_a[1], axis[2] * point_a[2]])
        point_b = np.array([axis[0] * point_b[0], axis[1] * point_b[1], axis[2] * point_b[2]])
        return np.linalg.norm(point_a - point_b)

    def get_gripper_pos(self):
        return self._observation["observation"][0:3]

    def get_target_pos(self):
        return self._observation["achieved_goal"]

    def get_horizontal_distance_to_target(self):
        target_obj_pos = self.get_target_pos()
        gripper_pos = self.get_gripper_pos()
        return self.get_distance(gripper_pos, target_obj_pos, [1, 1, 0])

    def get_distance_to_target(self):
        target_obj_pos = self.get_target_pos()
        gripper_pos = self.get_gripper_pos()
        return self.get_distance(gripper_pos, target_obj_pos, [1, 1, 1])

    def get_vertical_distance_to_target(self):
        target_obj_pos = self.get_target_pos()
        gripper_pos = self.get_gripper_pos()
        return self.get_distance(gripper_pos, target_obj_pos, [0, 0, 1])

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self._reward_type == 'sparse':
            failed_grasp_penalty = self.compute_failed_grasp_penalty()
            reward = self.compute_sparse_reward(achieved_goal, desired_goal) - failed_grasp_penalty
        else:
            reward = self.compute_dense_reward(achieved_goal, desired_goal)

        return reward

    def compute_failed_grasp_penalty(self):
        if self._attempted_grasp and not self._is_successful_grasp:
            return self._max_step_count - self._env_step_counter
        else:
            return 0

    def compute_dense_reward(self, achieved_goal, desired_goal):
        horizontal_distance = self.get_horizontal_distance_to_target()
        reward = -horizontal_distance
        if horizontal_distance <= 0.015:
            reward = -(self.get_distance_to_target())
        else:
            reward -= 1

        if self._is_success(achieved_goal, desired_goal):
            reward += self._additional_reward + self._max_step_count

        return reward * 10

    def compute_sparse_reward(self, achieved_goal, desired_goal):
        if self._is_success(achieved_goal, desired_goal):
            return 0
        else:
            return -1

    def _is_success(self, object_position, goal_position):
        if self._attempted_grasp:
            if object_position[2] >= goal_position[2]:
                return True
        return False

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._panda.panda_id)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(self._img_width) / self._img_height,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=self._img_width,
                                                  height=self._img_height,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self._img_height, self._img_width, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _draw_workspace_limits(self):
        xMin = self._table_workspace_shape[0][0]
        xMax = self._table_workspace_shape[0][1]
        yMin = self._table_workspace_shape[1][0]
        yMax = self._table_workspace_shape[1][1]
        zMin = self._table_workspace_shape[2][0] + 0.01

        p.addUserDebugLine([xMin, yMin, zMin], [xMax, yMin, zMin])
        p.addUserDebugLine([xMax, yMin, zMin], [xMax, yMax, zMin])
        p.addUserDebugLine([xMax, yMax, zMin], [xMin, yMax, zMin])
        p.addUserDebugLine([xMin, yMax, zMin], [xMin, yMin, zMin])
