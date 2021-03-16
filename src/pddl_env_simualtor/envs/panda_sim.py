import os
import math
import time
import pybullet as pb
import pybullet_data as pd
import numpy as np

from typing import Tuple, Optional, List

# import local packages
from src.pddl_env_simualtor.workspace_envs.manipulation_env import ManipulationDomain


class PandaSim:

    initial_positions = {
        'panda_joint1': 0.0, 'panda_joint2': -0.54, 'panda_joint3': 0.0,
        'panda_joint4': -2.6, 'panda_joint5': -0.30, 'panda_joint6': 2.0,
        'panda_joint7': 1.0, 'panda_finger_joint1': 0.02, 'panda_finger_joint2': 0.02,
    }

    def __init__(self,
                 physics_client_id,
                 use_IK: int = 0,
                 base_position: iter =(0.0, -0.4, 0.6),
                 control_orientation: int = 1,
                 joint_action_space: int = 7,
                 record: bool = False):

        self._time_step = 1. / 240.
        self._physics_client_id = physics_client_id
        self._use_IK = use_IK
        self._control_orientation = control_orientation
        self._base_position = base_position
        self.sim_start_default_height = 0.00
        self.joint_action_space = joint_action_space
        self._workspace_lim = [[0.3, 0.65], [-0.3, 0.3], [0.65, 1.5]]
        self.end_eff_idx = 11  # 8
        self._home_hand_pose = []
        self._num_dof = 7
        self._joint_name_to_ids = {}
        self.robot_id = None
        self._world: Optional[ManipulationDomain] = None
        self._record = record
        self.render()

    @property
    def world(self):
        return self._world

    @property
    def time_step(self):
        return self.time_step

    @property
    def base_position(self):
        return self._base_position

    @property
    def workspace_lim(self):
        return self._workspace_lim

    @property
    def num_dof(self):
        return self._num_dof

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, flag: bool):
        self._record = flag

    def render(self):
        # initialize simulation parameters
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=self._physics_client_id)
        pb.setPhysicsEngineParameter(maxNumCmdPer1ms=1000, physicsClientId=self._physics_client_id)
        pb.setTimeStep(self._time_step, physicsClientId=self._physics_client_id)
        pb.setGravity(0, 0, -9.8, physicsClientId=self._physics_client_id)
        pb.setAdditionalSearchPath(pd.getDataPath(), physicsClientId=self._physics_client_id)

        # Load robot model
        flags = pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | pb.URDF_USE_INERTIA_FROM_FILE | pb.URDF_USE_SELF_COLLISION
        self.robot_id = pb.loadURDF(fileName="franka_panda/panda.urdf",
                                    basePosition=self._base_position,
                                    baseOrientation=pb.getQuaternionFromEuler([0, 0, 3.14/2]),
                                    useFixedBase=True,
                                    flags=flags,
                                    physicsClientId=self._physics_client_id)

        assert self.robot_id is not None, "Failed to load the panda model"

        # reset joints to home position
        num_joints = pb.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
        idx = 0
        for i in range(num_joints):
            joint_info = pb.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is pb.JOINT_REVOLUTE or joint_type is pb.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()

                self._joint_name_to_ids[joint_name] = i

                pb.resetJointState(self.robot_id, i, self.initial_positions[joint_name],
                                   physicsClientId=self._physics_client_id)

                pb.setJointMotorControl2(self.robot_id, i, pb.POSITION_CONTROL,
                                         targetPosition=self.initial_positions[joint_name],
                                         positionGain=0.2,
                                         velocityGain=1.0,
                                         physicsClientId=self._physics_client_id)

                idx += 1

        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

        if self._use_IK:
            self._home_hand_pose = [0.0, -0.1, 0.8,
                                    min(math.pi, max(-math.pi, math.pi)),
                                    min(math.pi, max(-math.pi, 0)), math.pi/2]
                                    # min(0, max(-math.pi, 0))]

            self.apply_low_level_action(self._home_hand_pose)
            pb.stepSimulation(physicsClientId=self._physics_client_id)

            # create a constraint to keep the fingers centered
            c = pb.createConstraint(parentBodyUniqueId=self.robot_id,
                                    parentLinkIndex=9,
                                    childBodyUniqueId=self.robot_id,
                                    childLinkIndex=10,
                                    jointType=pb.JOINT_GEAR,
                                    jointAxis=[1, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=[0, 0, 0])

            pb.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        self._world = ManipulationDomain(physics_client_id=self._physics_client_id,
                                         workspace_lim=self._workspace_lim)
        # start the world for few secs
        for _ in range(100):
            pb.stepSimulation(physicsClientId=self._physics_client_id)

    def goal_distance(self, a: np.ndarray, b: np.ndarray):
        if not a.shape == b.shape:
            raise AssertionError("goal_distance(): shape of points mismatch")
        return np.linalg.norm(a - b, axis=-1)

    def get_ee_location(self, debug: bool = False):
        """
        A helper function that returns the pose : including the position and quanternion of the end effector in the
         simulation
        """
        _state = pb.getLinkState(self.robot_id,
                                 self.end_eff_idx,
                                 computeLinkVelocity=1,
                                 computeForwardKinematics=1,
                                 physicsClientId=self._physics_client_id)
        _ee_pos = _state[0]
        _ee_orn = _state[1]

        if debug:
            print(_state)

        return _ee_pos, _ee_orn

    def done_action(self, target_pose: Tuple[list, list], tol_pos=0.01, tol_orn=0.05) -> bool:
        """
        A helper function that determines if the desired pose has been achieved or not
        """

        _ee_pos, _ee_orn = self.get_ee_location()

        d_pos = self.goal_distance(np.array(target_pose[0]), np.array(_ee_pos))
        d_orn = self.goal_distance(np.array(target_pose[1]), np.array(_ee_orn))

        if d_pos < tol_pos and d_orn < tol_orn:
            return True
        return False

    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

        for joint_name in self._joint_name_to_ids.keys():
            jointInfo = pb.getJointInfo(self.robot_id,
                                        self._joint_name_to_ids[joint_name],
                                        physicsClientId=self._physics_client_id)

            ll, ul = jointInfo[8:10]
            jr = ul - ll
            # For simplicity, assume resting state == initial state
            rp = self.initial_positions[joint_name]
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(jr)
            rest_poses.append(rp)

        return lower_limits, upper_limits, joint_ranges, rest_poses

    def apply_high_level_action(self, action_type: str, pose, vel=1):
        """
        A wrapper that call the appropriate action function based on the action type, automatically adjusts the time
        required to reach the desired location state.
        """

        if action_type == "openEE":
            # apply action
            self.pre_grasp(vel)
            for _ in range(500):
                # simulate it
                pb.stepSimulation()
                time.sleep(self._time_step)

        elif action_type == "closeEE":
            # apply action
            self.grasp(max_velocity=vel)
            for _ in range(500):
                # simulate it
                pb.stepSimulation()
                time.sleep(self._time_step)

        elif action_type == "transit":
            _ee_target_quat_orn = pb.getQuaternionFromEuler(pose[3:6])
            _ee_target_pos = pose[:3]

            self.apply_low_level_action(action=pose, max_vel=vel)
            self.pre_grasp(vel)
            _timer = 0
            while not self.done_action((_ee_target_pos, _ee_target_quat_orn)):
                # simulate it
                pb.stepSimulation()
                time.sleep(self._time_step)
                _timer += self._time_step

                if _timer > 10:
                    break

        elif action_type == "transfer":
            _ee_target_quat_orn = pb.getQuaternionFromEuler(pose[3:6])
            _ee_target_pos = pose[:3]

            self.apply_low_level_action(action=pose, max_vel=vel)
            self.grasp(max_velocity=vel)

            _timer = 0
            while not self.done_action((_ee_target_pos, _ee_target_quat_orn)):
                # simulate it
                pb.stepSimulation()
                time.sleep(self._time_step)
                _timer += self._time_step

                if _timer > 10:
                    break

    def apply_low_level_action(self, action, max_vel=-1):

        if self._use_IK:
            # ------------------ #
            # --- IK control --- #
            # ------------------ #
            self.apply_action_ik(action=action, max_vel=max_vel)

        else:
            # --------------------- #
            # --- Joint control --- #
            # --------------------- #
            self.apply_joint_control(action=action)

    def apply_action_ik(self, action, max_vel=-1):
        if not (len(action) == 3 or len(action) == 6 or len(action) == 7):
            raise AssertionError('number of action commands must be \n- 3: (dx,dy,dz)'
                                 '\n- 6: (dx,dy,dz,droll,dpitch,dyaw)'
                                 '\n- 7: (dx,dy,dz,qx,qy,qz,w)'
                                 '\ninstead it is: ', len(action))

        # --- Constraint end-effector pose inside the workspace --- #
        dx, dy, dz = action[:3]
        new_pos = [dx, dy,
                   min(self._workspace_lim[2][1], max(self._workspace_lim[2][0], dz))]

        # if orientation is not under control, keep it fixed
        if not self._control_orientation:
            new_quat_orn = pb.getQuaternionFromEuler(self._home_hand_pose[3:6])

        # otherwise, if it is defined as euler angles
        elif len(action) == 6:
            droll, dpitch, dyaw = action[3:]

            eu_orn = [min(math.pi, max(-math.pi, droll)),
                      min(math.pi, max(-math.pi, dpitch)),
                      min(math.pi, max(-math.pi, dyaw))]

            new_quat_orn = pb.getQuaternionFromEuler(eu_orn)

        # otherwise, if it is define as quaternion
        elif len(action) == 7:
            new_quat_orn = action[3:7]

        # otherwise, use current orientation
        else:
            new_quat_orn = \
                pb.getLinkState(self.robot_id, self.end_eff_idx, physicsClientId=self._physics_client_id)[5]

        # --- compute joint positions with IK --- #
        jointPoses = pb.calculateInverseKinematics(self.robot_id, self.end_eff_idx, new_pos, new_quat_orn,
                                                   maxNumIterations=100,
                                                   residualThreshold=.001,
                                                   physicsClientId=self._physics_client_id)

        # --- set joint control --- #
        if max_vel == -1:
            pb.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                         jointIndices=self._joint_name_to_ids.values(),
                                         controlMode=pb.POSITION_CONTROL,
                                         targetPositions=jointPoses,
                                         positionGains=[0.2] * len(jointPoses),
                                         velocityGains=[1.0] * len(jointPoses),
                                         physicsClientId=self._physics_client_id)

        else:
            for i in range(self._num_dof):
                pb.setJointMotorControl2(bodyUniqueId=self.robot_id,
                                         jointIndex=i,
                                         controlMode=pb.POSITION_CONTROL,
                                         targetPosition=jointPoses[i],
                                         maxVelocity=max_vel,
                                         physicsClientId=self._physics_client_id)

    def apply_joint_control(self, action):

        assert len(action) == self.joint_action_space, (
            'number of motor commands differs from number of motor to control', len(action))

        joint_idxs = tuple(self._joint_name_to_ids.values())
        for i, val in enumerate(action):
            motor = joint_idxs[i]
            new_motor_pos = min(self.ul[i], max(self.ll[i], val))

            pb.setJointMotorControl2(self.robot_id,
                                     motor,
                                     pb.POSITION_CONTROL,
                                     targetPosition=new_motor_pos,
                                     positionGain=0.5, velocityGain=1.0,
                                     physicsClientId=self._physics_client_id)

    def pre_grasp(self, max_velocity: int = 1):
        self.apply_action_fingers([0.04, 0.04], max_velocity)

    def grasp(self, obj_id: int = None, max_velocity: int = 1):
        self.apply_action_fingers([0.0, 0.0], max_velocity, obj_id)

    def apply_action_fingers(self, action, max_velocity, obj_id: int = None):
        # move finger joints in position control
        assert len(action) == 2, ('finger joints are 2! The number of actions you passed is ', len(action))

        idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

        # use object id to check contact force and eventually stop the finger motion
        if obj_id is not None:
            _, forces = self.check_contact_fingertips(obj_id)
            # print("contact forces {}".format(forces))

            if forces[0] >= 20.0:
                action[0] = pb.getJointState(self.robot_id, idx_fingers[0], physicsClientId=self._physics_client_id)[0]

            if forces[1] >= 20.0:
                action[1] = pb.getJointState(self.robot_id, idx_fingers[1], physicsClientId=self._physics_client_id)[0]

        for i, idx in enumerate(idx_fingers):
            pb.setJointMotorControl2(self.robot_id,
                                     idx,
                                     pb.POSITION_CONTROL,
                                     targetPosition=action[i],
                                     force=30,
                                     maxVelocity=max_velocity,
                                     physicsClientId=self._physics_client_id)

    def check_contact_fingertips(self, obj_id):
        # check if there is any contact on the internal part of the fingers, to control if they are correctly touching
        # an object

        idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

        p0 = pb.getContactPoints(obj_id, self.robot_id,
                                 linkIndexB=idx_fingers[0],
                                 physicsClientId=self._physics_client_id)
        p1 = pb.getContactPoints(obj_id, self.robot_id,
                                 linkIndexB=idx_fingers[1],
                                 physicsClientId=self._physics_client_id)

        p0_contact = 0
        p0_f = [0]
        if len(p0) > 0:
            # get cartesian position of the finger link frame in world coordinates
            w_pos_f0 = pb.getLinkState(self.robot_id, idx_fingers[0], physicsClientId=self._physics_client_id)[4:6]
            f0_pos_w = pb.invertTransform(w_pos_f0[0], w_pos_f0[1])

            for pp in p0:
                # compute relative position of the contact point wrt the finger link frame
                f0_pos_pp = pb.multiplyTransforms(f0_pos_w[0], f0_pos_w[1], pp[6], f0_pos_w[1])

                # check if contact in the internal part of finger
                if f0_pos_pp[0][1] <= 0.001 and f0_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p0_contact += 1
                    p0_f.append(pp[9])

        p0_f_mean = np.mean(p0_f)

        p1_contact = 0
        p1_f = [0]
        if len(p1) > 0:
            w_pos_f1 = pb.getLinkState(self.robot_id, idx_fingers[1], physicsClientId=self._physics_client_id)[4:6]
            f1_pos_w = pb.invertTransform(w_pos_f1[0], w_pos_f1[1])

            for pp in p1:
                # compute relative position of the contact point wrt the finger link frame
                f1_pos_pp = pb.multiplyTransforms(f1_pos_w[0], f1_pos_w[1], pp[6], f1_pos_w[1])

                # check if contact in the internal part of finger
                if f1_pos_pp[0][1] >= -0.001 and f1_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p1_contact += 1
                    p1_f.append(pp[9])

        p1_f_mean = np.mean(p0_f)

        return (p0_contact > 0) + (p1_contact > 0), (p0_f_mean, p1_f_mean)


if __name__ == "__main__":

    physics_client = pb.connect(pb.GUI)
    panda = PandaSim(physics_client, use_IK=1)
