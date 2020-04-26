import math as m
import os
import pybullet as p
import object_data
import numpy as np


class PandaEnv:

    def __init__(self, base_position, urdf_root_path=object_data.getDataPath(), time_step=0.01, use_ik=False,
                 num_controlled_joints=7):
        self.finger_force = 10
        self.urdf_root_path = os.path.join(urdf_root_path, "robot/panda.urdf")
        self.time_step = time_step
        self.use_ik = use_ik
        self.base_position = base_position
        self.gripper_index = 8
        self.num_controlled_joints = num_controlled_joints
        self.max_force = 5 * 240.
        self.max_velocity = .35
        self.start_joint_positions = [0.006, 0.4, -0.01, -1.6, 0.005, 2, -2.4, 0, 0, 0.05, 0.05]
        self.motor_count = len(self.start_joint_positions)
        self.panda_id = None
        self.gripper_pos = []  # x,y,z
        self.gripper_orn = []  # roll,pitch,yaw
        self.reset()

    def reset(self):
        self.panda_id = p.loadURDF(self.urdf_root_path, basePosition=self.base_position, useFixedBase=True)

        for i in range(self.motor_count):
            p.resetJointState(self.panda_id, i, self.start_joint_positions[i])
            p.setJointMotorControl2(self.panda_id, i, p.POSITION_CONTROL, targetPosition=self.start_joint_positions[i],
                                    force=self.max_force)

        state = p.getLinkState(self.panda_id, self.gripper_index)

        for j in range(p.getNumJoints(self.panda_id)):
            p.changeDynamics(self.panda_id, j, linearDamping=0, angularDamping=0)

        c = p.createConstraint(self.panda_id,
                               9,
                               self.panda_id,
                               10,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
        self.gripper_pos = list(state[0])
        self.gripper_orn = list(p.getEulerFromQuaternion(list(state[1])))

    def get_observation(self):
        observation = []
        state = p.getLinkState(self.panda_id, self.gripper_index, computeLinkVelocity=True)
        pos = state[0]
        orn = state[1]
        velL = state[6]
        velA = state[7]
        observation.extend(list(pos))
        observation.extend(list(orn))
        observation.extend(list(velL))
        observation.extend(list(velA))

        return observation

    def update_gripper_pos(self):
        state = p.getLinkState(self.panda_id, self.gripper_index)
        self.gripper_pos = list(state[0])
        euler = p.getEulerFromQuaternion(state[1])
        self.gripper_orn = list(euler)

    def apply_action(self, action, use_simulation=True):
        if self.use_ik:

            dx = action[0]
            dy = action[1]
            dz = action[2]

            droll = action[3]
            dpitch = action[4]
            dyaw = action[5]

            finger_angle = action[6]

            self.gripper_pos[0] = self.gripper_pos[0] + dx
            self.gripper_pos[1] = self.gripper_pos[1] + dy
            self.gripper_pos[2] = self.gripper_pos[2] + dz

            self.gripper_orn[0] = self.gripper_orn[0] + droll
            self.gripper_orn[1] = self.gripper_orn[1] + dpitch
            self.gripper_orn[2] = self.gripper_orn[2] + dyaw

            quat_orn = p.getQuaternionFromEuler(self.gripper_orn)

            joint_poses = p.calculateInverseKinematics(self.panda_id, self.gripper_index, self.gripper_pos, quat_orn)

            if use_simulation:
                for i in range(self.num_controlled_joints):
                    joint_info = p.getJointInfo(self.panda_id, i)
                    if joint_info[3] > -1:
                        p.setJointMotorControl2(bodyUniqueId=self.panda_id,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=joint_poses[i],
                                                force=self.max_force)
            else:
                for i in range(self.num_controlled_joints):
                    p.resetJointState(self.panda_id, i, joint_poses[i])

            # fingers

            p.setJointMotorControl2(self.panda_id,
                                    9,
                                    p.POSITION_CONTROL,
                                    targetPosition=finger_angle * 0.03 + 0.01,
                                    force=self.finger_force)
            p.setJointMotorControl2(self.panda_id,
                                    10,
                                    p.POSITION_CONTROL,
                                    targetPosition=finger_angle * 0.03 + 0.01,
                                    force=self.finger_force)

        else:
            for a in range(len(action)):
                curr_motor_pos = p.getJointState(self.panda_id, a)[0]
                new_motor_pos = curr_motor_pos + action[a]
                p.setJointMotorControl2(self.panda_id,
                                        a,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        positionGain=0.25,
                                        velocityGain=0.75,
                                        force=self.max_force)
