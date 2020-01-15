import math as m
import os
import pybullet as p
import robot_data


class PandaEnv:

    def __init__(self, base_position, urdf_root_path=robot_data.getDataPath(), time_step=0.01, use_ik=False,
                 num_controlled_joints=7):
        self.finger_force = 10
        self.urdf_root_path = os.path.join(urdf_root_path, "franka/robot/panda.urdf")
        self.time_step = time_step
        self.use_ik = use_ik
        self.base_position = base_position
        self.workspace_lim = [[0.2, 1],  # X
                              [-0.3, 0.3],  # Y
                              [0, 1]]  # Z
        self.workspace_lim_gripper = [[0.1, 0.70],  # X
                                      [-0.4, 0.4],  # Y
                                      [0.65, 5]]  # Z
        self.gripper_index = 8
        self.num_controlled_joints = num_controlled_joints
        self.max_force = 200
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

        self.gripper_pos = list(state[0])
        self.gripper_orn = list(p.getEulerFromQuaternion(list(state[1])))

    def get_observation(self):
        observation = []
        state = p.getLinkState(self.panda_id, self.gripper_index)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(list(pos))
        observation.extend(list(euler))
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

            self.gripper_pos[0] = min(self.workspace_lim_gripper[0][1],
                                      max(self.workspace_lim_gripper[0][0], self.gripper_pos[0] + dx))
            self.gripper_pos[1] = min(self.workspace_lim_gripper[1][1],
                                      max(self.workspace_lim_gripper[1][0], self.gripper_pos[1] + dy))
            self.gripper_pos[2] = min(self.workspace_lim_gripper[2][1],
                                      max(self.workspace_lim_gripper[2][0], self.gripper_pos[2] + dz))

            self.gripper_orn[0] = min(m.pi, max(-m.pi, self.gripper_orn[0] + droll))
            self.gripper_orn[1] = min(m.pi, max(-m.pi, self.gripper_orn[1] + dpitch))
            self.gripper_orn[2] = min(m.pi, max(-m.pi, self.gripper_orn[2] + dyaw))

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
                                                force=self.max_force,
                                                maxVelocity=self.max_velocity,
                                                positionGain=0.3,
                                                velocityGain=1)
            else:
                for i in range(self.num_controlled_joints):
                    p.resetJointState(self.panda_id, i, joint_poses[i])

            # fingers

            p.setJointMotorControl2(self.panda_id,
                                    9,
                                    p.POSITION_CONTROL,
                                    targetPosition=finger_angle * 0.05,
                                    force=self.finger_force)
            p.setJointMotorControl2(self.panda_id,
                                    10,
                                    p.POSITION_CONTROL,
                                    targetPosition=finger_angle * 0.05,
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
