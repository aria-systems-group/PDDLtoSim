"""
A file that calls pybullet API to simulate a given pddl file
"""
import warnings
import time
import numpy as np
import math

from typing import Tuple, Optional, List

import pybullet as p
import pybullet_data as pd

PATH_TO_LOCAL_URDF = "/home/karan-m/Documents/Research/franka_simulation/PDDLtoSim/models/"
PATH_TO_BOXES = "boxes/"
PATH_TO_FURNITURE = "furniture/"

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


class Panda:

    def __init__(self, bullet_client, offset: np, mode: str, joint_positions: List, fps: int = 120):
        self.step_size: float = 1/fps
        self.bullet_client = bullet_client
        self.offset = np.array(offset)
        self.mode: Optional[str] = mode

        # set simulation parameters
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in self._get_valid_modes():
            warnings.warn(f"Please make sure that the mode you entered is of type Str and one of the following: "
                          f"{self._get_valid_modes()}")

        self._mode = mode

    # set of valid modes that execute different types of simulation environment
    def _get_valid_modes(self):
        _valid_modes = ["interactive", "simulate"]
        return _valid_modes

    def _build_env(self):

        # build the plane
        self.bullet_client.loadURDF(fileName="plane.urdf")

        _base_positon = np.array([0, 0, 0])
        _base_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # build the robot
        self.robot_obj = self.bullet_client.loadURDF(fileName="franka_panda/panda.urdf",
                                                     basePosition=_base_positon + np.array([-0.65, -0.4, 0.6]),
                                                     baseOrientation=_base_orientation,
                                                     useFixedBase=True,
                                                     flags=self.flags)

        # build the Table
        PATH_TO_TABLE = PATH_TO_LOCAL_URDF + PATH_TO_FURNITURE + "table.sdf"
        PATH_TO_RED_BOX = PATH_TO_LOCAL_URDF + PATH_TO_BOXES + "red_box.urdf"
        PATH_TO_BLACK_BOX = PATH_TO_LOCAL_URDF + PATH_TO_BOXES + "black_box.urdf"
        # self.bullet_client.loadSDF(sdfFileName=PATH_TO_TABLE)

        self.bullet_client.loadURDF(fileName="table/table.urdf")

        # add boxes on the table
        self.box_obj = self.bullet_client.loadURDF(fileName=PATH_TO_RED_BOX,
                                                   basePosition=_base_positon + np.array([-0.1, 0, 0.7 + 0.17/2]),
                                                   baseOrientation=_base_orientation,
                                                   useFixedBase=False,
                                                   flags=self.flags)

        # self.bullet_client.loadURDF(fileName=PATH_TO_BLACK_BOX,
        #                             basePosition=_base_positon + + np.array([-0.3, 0, 0.6 + 0.17/2]),
        #                             baseOrientation=_base_orientation,
        #                             useFixedBase=True,
        #                             flags=self.flags)

        index = 0
        self.state = 0
        self.finger_target = 0
        self.gripper_height = 0.6 + 0.2

        # create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(parentBodyUniqueId=self.robot_obj,
                                                parentLinkIndex=9,
                                                childBodyUniqueId=self.robot_obj,
                                                childLinkIndex=10,
                                                jointType=self.bullet_client.JOINT_GEAR,
                                                jointAxis=[1, 0, 0],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0])

        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        for j in range(self.bullet_client.getNumJoints(self.robot_obj)):
            self.bullet_client.changeDynamics(self.robot_obj, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.robot_obj, j)
            # print("info=",info)
            _joint_name = info[1]
            _joint_type = info[2]

            if _joint_type == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(self.robot_obj, j, jointPositions[index])
                index = index + 1

            if _joint_type == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.robot_obj, j, jointPositions[index])
                index = index + 1

        self.t = 0

    def _interactive_update_state(self):
        keys = self.bullet_client.getKeyboardEvents()
        if len(keys) > 0:
            for k, v in keys.items():
                if v & self.bullet_client.KEY_WAS_TRIGGERED:
                    if k == ord('1'):
                        self.state = 1
                    if k == ord('2'):
                        self.state = 2
                    if k == ord('3'):
                        self.state = 3
                    if k == ord('4'):
                        self.state = 4
                    if k == ord('5'):
                        self.state = 5
                    if k == ord('6'):
                        self.state = 6
                    if k == ord('7'):
                        self.state = 7
                    if k == ord('8'):
                        self.state = 8
                if v & self.bullet_client.KEY_WAS_RELEASED:
                    self.state = 0

    def _interactive_step(self):
        """
        A function that makes the Franka Panda robot respond to keyboard inputs
        """

        if self.state == 7:
            self.finger_target = 0.01

        if self.state == 8:
            self.finger_target = 0.04

        self.bullet_client.submitProfileTiming("step")
        self._interactive_update_state()

        # _robot_base_pos = self.bullet_client.getBasePositionAndOrientation(self.robot_obj)
        _robot_ee_world_pos: list = self.bullet_client.getLinkState(self.robot_obj, pandaEndEffectorIndex)[4]

        self.pos = _robot_ee_world_pos

        alpha = 0.99  # 0.99
        # state 1 - change the x value
        # state 2 - change the y value
        # state 3 - change the z value
        # state 8 - Open the fingers value
        # state 7 - close the fingers value
        if self.state == 1:
            self.pos = [self.pos[0] + alpha * self.step_size, self.pos[1], self.pos[2]]

        elif self.state == 2:
            self.pos = [self.pos[0], self.pos[1] + alpha * self.step_size, self.pos[2]]

        elif self.state == 3:
            self.pos = [self.pos[0], self.pos[1], self.pos[2] + alpha * self.step_size]

        if self.state == 4:
            self.pos = [self.pos[0] - alpha * self.step_size, self.pos[1], self.pos[2]]

        elif self.state == 5:
            self.pos = [self.pos[0], self.pos[1] - alpha * self.step_size, self.pos[2]]

        elif self.state == 6:
            self.pos = [self.pos[0], self.pos[1], self.pos[2] - alpha * self.step_size]

        orn = self.bullet_client.getQuaternionFromEuler([math.pi, 0., 0.])
        self.bullet_client.submitProfileTiming("IK")

        joint_poses = self.bullet_client.calculateInverseKinematics(self.robot_obj,
                                                                    pandaEndEffectorIndex, self.pos, orn, ll,
                                                                    ul, jr, rp, maxNumIterations=20)
        self.bullet_client.submitProfileTiming()

        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(self.robot_obj, i, self.bullet_client.POSITION_CONTROL,
                                                     joint_poses[i], force=5 * 240.)

        # target for fingers
        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.robot_obj, i, self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target, force=10)
        self.bullet_client.submitProfileTiming()

    def _continuous_pick_and_place(self, time_step: int, org_box_pos: list):
        if 0 < time_step <= 200:
            # go to the object

            _grasp_pos = [org_box_pos[0], org_box_pos[1], org_box_pos[2] + 0.3]
            self.perform_action(_grasp_pos)
            self.pre_grasp()
        elif 200 < time_step <= 300:
            # open the end effector
            self.pre_grasp()
        elif 300 < time_step <= 400:
            # go down to the target
            _grasp_pos = [org_box_pos[0], org_box_pos[1], org_box_pos[2]]
            self.perform_action(_grasp_pos)
            self.pre_grasp()

        elif 400 < time_step <= 500:
            # grasp the object
            self.grasp()

        elif 500 < time_step <= 700:
            # place the object somewhere and release
            # go down to the target and open the end effector
            _grasp_pos = [org_box_pos[0], org_box_pos[1], org_box_pos[2] + 0.3]
            self.perform_action(_grasp_pos)
            self.grasp()
        elif 700 < time_step <= 800:
            # go to some rest position
            _grasp_pos = [org_box_pos[0], org_box_pos[1], org_box_pos[2] + 0.03]
            self.perform_action(_grasp_pos)
            self.grasp()
        elif 800 < time_step <= 900:
            # go to some rest position
            self.pre_grasp()

    def _iterative_pick_and_place(self, org_box_pos: list):
        _grasp_pos = [org_box_pos[0], org_box_pos[1], org_box_pos[2] + 0.3]
        self.perform_action(_grasp_pos)
        self.pre_grasp()
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

        _grasp_pos = [org_box_pos[0], org_box_pos[1], org_box_pos[2]]
        self.perform_action(_grasp_pos)
        self.pre_grasp()
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

        self.grasp()
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

        _grasp_pos = [org_box_pos[0], org_box_pos[1], org_box_pos[2] + 0.3]
        self.perform_action(_grasp_pos)
        self.grasp()
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

        _grasp_pos = [org_box_pos[0], org_box_pos[1], org_box_pos[2] + 0.01]
        self.perform_action(_grasp_pos)
        self.grasp()
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

        self.pre_grasp()
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)




    def perform_action(self, pos: list, max_vel: int = 2):
        if len(pos) != 3:
            warnings.warn("Please provide a sequence of Robot End Effector Position.")

        # orientation remains constant
        orn = self.bullet_client.getQuaternionFromEuler([math.pi, 0., 0.])
        self.bullet_client.submitProfileTiming("IK")

        joint_poses = self.bullet_client.calculateInverseKinematics(self.robot_obj,
                                                                    pandaEndEffectorIndex, pos, orn, ll,
                                                                    ul, jr, rp, maxNumIterations=20)

        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(bodyUniqueId=self.robot_obj,
                                                     jointIndex=i,
                                                     controlMode=self.bullet_client.POSITION_CONTROL,
                                                     targetPosition=joint_poses[i],
                                                     maxVelocity=max_vel)

    def pre_grasp(self):
        self.apply_action_fingers([0.04, 0.04], pre_grasp=True)

    def grasp(self):
        self.apply_action_fingers([0.0, 0.0])

    def apply_action_fingers(self, finger_pos, pre_grasp: bool = False):
        assert len(finger_pos) == 2, ('finger joints are 2! The number of positions you passed is ', len(finger_pos))
        force = 20
        if pre_grasp:
            force = 0

        for i in [9, 10]:
            self.bullet_client.setJointMotorControl2(self.robot_obj,
                                                     i,
                                                     self.bullet_client.POSITION_CONTROL,
                                                     self.finger_target,
                                                     force=force,
                                                     maxVelocity=2)
    def simulate(self):
        # build the env
        self._build_env()

        self.bullet_client.submitProfileTiming("start")
        _red_box_pos = self.bullet_client.getBasePositionAndOrientation(self.box_obj)[0]
        _testcase = False

        # start simulating
        if self.mode == "interactive":
            for i in range(10000):
                # self.bullet_client.submitProfileTiming("full_step")
                self._interactive_step()
                self.bullet_client.stepSimulation()
                # self.bullet_client.submitProfileTiming()
                time.sleep(0.01)

        elif self.mode == "simulate":
            if _testcase:
                self._iterative_pick_and_place(org_box_pos=_red_box_pos)
            else:
                for i in range(900):
                    self._continuous_pick_and_place(time_step=i, org_box_pos=_red_box_pos)
                    self.bullet_client.stepSimulation()
                    time.sleep(0.01)

        self.bullet_client.submitProfileTiming()


if __name__ == "__main__":
    # video requires ffmpeg available in path
    createVideo = False
    fps = 240.
    timeStep = 1. / fps

    if createVideo:
        p.connect(p.GUI, options="--minGraphicsUpdateTimeMs=0 --mp4=\"pybullet_grasp.mp4\" --mp4fps=" + str(fps))
    else:
        p.connect(p.GUI)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.setTimeStep(timeStep)
    p.setGravity(0, 0, -9.8)

    panda_sim = Panda(p, [0, 0, 0], joint_positions=jointPositions, mode="simulate")
    panda_sim.simulate()