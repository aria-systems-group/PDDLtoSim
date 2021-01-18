"""
A file that calls pybullet API to simulate a given pddl file
"""

# import pybullet
# import time
# import pybullet_data
#
# PATH_TO_PANDA_URDF = "/home/karan-m/Documents/Research/franka_simulation/PDDLtoSim/models/robots/Panda" + "/panda.urdf"
#
#
# def _run_simulator():
#     physics_client = pybullet.connect(pybullet.GUI)
#     pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
#     pybullet.setGravity(0, 0, -10)
#
#     plane_id = pybullet.loadURDF("plane.urdf")
#     start_pos = [0, 0, 10]
#     start_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])
#
#     box_id = pybullet.loadURDF(PATH_TO_PANDA_URDF, start_pos, start_orientation)
#
#     # # set the center of mass frame (loadURDF sets base link frame)
#     # start_orientation.resetBasePositionAndOrientation(box_id, start_pos, start_orientation)
#     #
#     # for i in range(10000):
#     #     pybullet.stepSimulation()
#     #     time.sleep(1. / 240.)
#
#     cubePos, cubeOrn = pybullet.getBasePositionAndOrientation(box_id)
#     print(cubePos, cubeOrn)
#     pybullet.disconnect()
import time
import numpy as np
import math

import pybullet as p
import pybullet_data as pd

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11  # 8
pandaNumDofs = 7

ll = [-7] * pandaNumDofs
# upper limits for null space (todo: set them to proper range)
ul = [7] * pandaNumDofs
# joint ranges for null space (todo: set them to proper range)
jr = [7] * pandaNumDofs
# restposes for null space
jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions


class PandaSim(object):
    def __init__(self, bullet_client, offset):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)

        # print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.legos = []

        self.bullet_client.loadURDF("tray/tray.urdf", [0 + offset[0], 0 + offset[1], -0.6 + offset[2]],
                                    [-0.5, -0.5, -0.5, 0.5], flags=flags)
        self.legos.append(
            self.bullet_client.loadURDF("lego/lego.urdf", np.array([0.1, 0.3, -0.5]) + self.offset, flags=flags))
        self.bullet_client.changeVisualShape(self.legos[0], -1, rgbaColor=[1, 0, 0, 1])
        self.legos.append(
            self.bullet_client.loadURDF("lego/lego.urdf", np.array([-0.1, 0.3, -0.5]) + self.offset, flags=flags))
        self.legos.append(
            self.bullet_client.loadURDF("lego/lego.urdf", np.array([0.1, 0.3, -0.7]) + self.offset, flags=flags))
        self.sphereId = self.bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.6]) + self.offset,
                                                    flags=flags)
        self.bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.5]) + self.offset, flags=flags)
        self.bullet_client.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.7]) + self.offset, flags=flags)
        orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]) + self.offset, orn,
                                                 useFixedBase=True, flags=flags)
        index = 0
        self.state = 0
        self.control_dt = 1. / 240.
        self.finger_target = 0
        self.gripper_height = 0.2

        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                                                9,
                                                self.panda,
                                                10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            # print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
        self.t = 0.

    def reset(self):
        pass

    def update_state(self):
        keys = self.bullet_client.getKeyboardEvents()
        if len(keys) > 0:
            for k, v in keys.items():
                if v & self.bullet_client.KEY_WAS_TRIGGERED:
                    if (k == ord('1')):
                        self.state = 1
                    if (k == ord('2')):
                        self.state = 2
                    if (k == ord('3')):
                        self.state = 3
                    if (k == ord('4')):
                        self.state = 4
                    if (k == ord('5')):
                        self.state = 5
                    if (k == ord('6')):
                        self.state = 6
                if v & self.bullet_client.KEY_WAS_RELEASED:
                    self.state = 0

    def step(self):
        if self.state == 6:
            self.finger_target = 0.01
        if self.state == 5:
            self.finger_target = 0.04
        self.bullet_client.submitProfileTiming("step")
        self.update_state()
        # print("self.state=",self.state)
        # print("self.finger_target=",self.finger_target)
        alpha = 0.9  # 0.99
        if self.state == 1 or self.state == 2 or self.state == 3 or self.state == 4 or self.state == 7:
            # gripper_height = 0.034
            self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.03
            if self.state == 2 or self.state == 3 or self.state == 7:
                self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.2

            t = self.t
            self.t += self.control_dt
            pos = [self.offset[0] + 0.2 * math.sin(1.5 * t), self.offset[1] + self.gripper_height,
                   self.offset[2] + -0.6 + 0.1 * math.cos(1.5 * t)]
            if self.state == 3 or self.state == 4:
                pos, o = self.bullet_client.getBasePositionAndOrientation(self.legos[0])
                pos = [pos[0], self.gripper_height, pos[2]]
                self.prev_pos = pos
            if self.state == 7:
                pos = self.prev_pos
                diffX = pos[0] - self.offset[0]
                diffZ = pos[2] - (self.offset[2] - 0.6)
                self.prev_pos = [self.prev_pos[0] - diffX * 0.1, self.prev_pos[1], self.prev_pos[2] - diffZ * 0.1]

            orn = self.bullet_client.getQuaternionFromEuler([math.pi / 2., 0., 0.])
            self.bullet_client.submitProfileTiming("IK")
            jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll,
                                                                       ul,
                                                                       jr, rp, maxNumIterations=20)
            self.bullet_client.submitProfileTiming()
            for i in range(pandaNumDofs):
                self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                         jointPoses[i], force=5 * 240.)
            # target for fingers
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=10)
        self.bullet_client.submitProfileTiming()


class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 3, 5, 4, 6, 3, 7]
        self.state_durations = [1, 1, 1, 2, 1, 1, 10]

    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state = self.states[self.cur_state]
            # print("self.state=",self.state)

if __name__ == "__main__":
    # video requires ffmpeg available in path
    createVideo = False
    fps = 240.
    timeStep = 1. / fps

    if createVideo:
        p.connect(p.GUI, options="--minGraphicsUpdateTimeMs=0 --mp4=\"pybullet_grasp.mp4\" --mp4fps=" + str(fps))
    else:
        p.connect(p.GUI)

    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22,
                                 cameraTargetPosition=[0.35, -0.13, 0])
    p.setAdditionalSearchPath(pd.getDataPath())

    p.setTimeStep(timeStep)
    p.setGravity(0, -9.8, 0)

    panda = PandaSim(p, [0, 0, 0])
    panda.control_dt = timeStep

    logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")
    panda.bullet_client.submitProfileTiming("start")
    for i in range(100000):
        panda.bullet_client.submitProfileTiming("full_step")
        panda.step()
        p.stepSimulation()
        if createVideo:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        if not createVideo:
            time.sleep(timeStep)
        panda.bullet_client.submitProfileTiming()
    panda.bullet_client.submitProfileTiming()
    panda.bullet_client.stopStateLogging(logId)