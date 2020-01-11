import os, inspect
import pybullet as p
import robot_data
import math as m

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class PandaEnv:

    def __init__(self, urdfRootPath=robot_data.getDataPath(), timeStep=0.01, useInverseKinematics=0,
                 basePosition=[-0.6, -0.4, 0.625], numControlledJoints=7):
        self.fingerAForce = 10
        self.fingerBForce = 10
        self.urdfRootPath = os.path.join(urdfRootPath, "franka/robot/panda.urdf")
        self.timeStep = timeStep
        self.useInverseKinematics = useInverseKinematics
        self.useNullSpace = 0
        self.useSimulation = 1
        self.basePosition = basePosition
        self.workspace_lim = [[0.2, 1], [-0.3, 0.3], [0, 1]]
        self.workspace_lim_endEff = [[0.1, 0.70], [-0.4, 0.4], [0.65, 5]]
        self.gripperIndex = 8
        self.numControlledJoints = numControlledJoints
        self.max_force = 200
        self.max_velocity = .35
        self.jointPositions = [
            0.006, 0.4, -0.01, -1.6, 0.005, 2, -2.4, 0, 0, 0.05, 0.05
        ]
        self.reset()

    def reset(self):
        # load model and position it's base on base position
        self.pandaId = p.loadURDF(self.urdfRootPath, basePosition=self.basePosition, useFixedBase=True)
        for i in range(11):
            p.resetJointState(self.pandaId, i, self.jointPositions[i])
            p.setJointMotorControl2(self.pandaId, i, p.POSITION_CONTROL, targetPosition=self.jointPositions[i],
                                    force=self.max_force)

        state = p.getLinkState(self.pandaId, self.gripperIndex)

        self.endEffPos = list(state[0])  # x,y,z
        self.endEffOrn = list(p.getEulerFromQuaternion(list(state[1])))  # roll,pitch,yaw

    def getJointsRanges(self):
        # to-be-defined
        return 0

    def getActionDimension(self):
        return self.numControlledJoints

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.pandaId, self.gripperIndex)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(list(pos))
        observation.extend(list(euler))
        return observation

    def updateGripPos(self):
        state = p.getLinkState(self.pandaId, self.gripperIndex)
        self.endEffPos = list(state[0])
        euler = p.getEulerFromQuaternion(state[1])
        self.endEffOrn = list(euler)

    def apply_action(self, action, useSimulation=True):
        if self.useInverseKinematics:

            dx = action[0]
            dy = action[1]
            dz = action[2]

            droll = action[3]
            dpitch = action[4]
            dyaw = action[5]

            finger_angle = action[6]

            self.endEffPos[0] = min(self.workspace_lim_endEff[0][1],
                                    max(self.workspace_lim_endEff[0][0], self.endEffPos[0] + dx))
            self.endEffPos[1] = min(self.workspace_lim_endEff[1][1],
                                    max(self.workspace_lim_endEff[1][0], self.endEffPos[1] + dy))
            self.endEffPos[2] = min(self.workspace_lim_endEff[2][1],
                                    max(self.workspace_lim_endEff[2][0], self.endEffPos[2] + dz))

            self.endEffOrn[0] = min(m.pi, max(-m.pi, self.endEffOrn[0] + droll))
            self.endEffOrn[1] = min(m.pi, max(-m.pi, self.endEffOrn[1] + dpitch))
            self.endEffOrn[2] = min(m.pi, max(-m.pi, self.endEffOrn[2] + dyaw))

            quat_orn = p.getQuaternionFromEuler(self.endEffOrn)

            joint_poses = p.calculateInverseKinematics(self.pandaId, self.gripperIndex, self.endEffPos, quat_orn)

            if useSimulation:
                for i in range(self.numControlledJoints):
                    joint_info = p.getJointInfo(self.pandaId, i)
                    if joint_info[3] > -1:
                        p.setJointMotorControl2(bodyUniqueId=self.pandaId,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=joint_poses[i],
                                                force=self.max_force,
                                                maxVelocity=self.max_velocity,
                                                positionGain=0.3,
                                                velocityGain=1)
            else:
                for i in range(self.numControlledJoints):
                    p.resetJointState(self.pandaId, i, joint_poses[i])

            # fingers

            p.setJointMotorControl2(self.pandaId,
                                    9,
                                    p.POSITION_CONTROL,
                                    targetPosition=finger_angle * 0.05,
                                    force=self.fingerAForce)
            p.setJointMotorControl2(self.pandaId,
                                    10,
                                    p.POSITION_CONTROL,
                                    targetPosition=finger_angle * 0.05,
                                    force=self.fingerBForce)

        else:
            for a in range(len(action)):
                curr_motor_pos = p.getJointState(self.pandaId, a)[0]
                new_motor_pos = curr_motor_pos + action[a]  # supposed to be a delta
                p.setJointMotorControl2(self.pandaId,
                                        a,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        positionGain=0.25,
                                        velocityGain=0.75,
                                        force=self.max_force)
