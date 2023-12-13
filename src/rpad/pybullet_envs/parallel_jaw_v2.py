import os

try:
    from importlib.resources import as_file, files  # type: ignore
except ImportError:
    from importlib_resources import as_file, files  # type: ignore

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

fn = as_file(files("rpad_pybullet_envs_where2act_data").joinpath("."))
with fn as f:
    PARALLEL_JAW_GRIPPER_URDF = os.path.join(f, "panda_gripper.urdf")

MAX_RANGE = 0.04
MIN_RANGE = 0.0


class FloatingParallelJawGripper:
    def __init__(self, client_id):
        self.client_id = client_id

        # This is just a floating gripper yay. It gets loaded at the origin.
        self.base_id = p.loadURDF(
            PARALLEL_JAW_GRIPPER_URDF, physicsClientId=self.client_id
        )

        # Reset the gripper to be somewhere.
        g_pos, g_ori = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.reset_pose(g_pos, g_ori)

        self.mass = 5

        self.activated = False
        self.contact_const = None
        self.contact_link_index = None

    def reset_pose(self, pos, ori):
        x, y, z = pos
        qx, qy, qz, qw = ori

        # Convert to pybullet's euler angles.
        r = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = r.as_euler("XYZ")

        for joint_id, joint_val in enumerate([x, y, z, roll, pitch, yaw]):
            p.resetJointState(
                self.base_id,
                joint_id,
                targetValue=joint_val,
                targetVelocity=0,
                physicsClientId=self.client_id,
            )

    def set_velocity(self, lin_vel, ang_vel):
        p.resetBaseVelocity(self.base_id, lin_vel, ang_vel, self.client_id)

    def detect_contact(self, obj_id):
        points = p.getContactPoints(bodyA=self.base_id, bodyB=obj_id, linkIndexA=0)
        return len(points) != 0

    def set_gripper_command(self, cmd):
        """Control signal should be on 0-1 range.

        Args:
            cmd (float): The control signal.
        """
        assert 0 <= cmd <= 1

        cmd = MIN_RANGE + cmd * (MAX_RANGE - MIN_RANGE)

        # Left finger
        p.setJointMotorControl2(
            bodyUniqueId=self.base_id,
            jointIndex=6,
            controlMode=p.POSITION_CONTROL,
            targetPosition=cmd,
            force=10,
            physicsClientId=self.client_id,
        )
        # Right finger
        p.setJointMotorControl2(
            bodyUniqueId=self.base_id,
            jointIndex=7,
            controlMode=p.POSITION_CONTROL,
            targetPosition=cmd,
            force=10,
            physicsClientId=self.client_id,
        )

    def open(self, timeout=100, tolerance=0.001):
        while timeout > 0:
            self.set_gripper_command(1.0)
            p.stepSimulation(self.client_id)

            # Check to see if the gripper is open.
            # Make sure the joint state of both fingers are above a certain threshold.
            joint_states = p.getJointStates(
                bodyUniqueId=self.base_id,
                jointIndices=[6, 7],
                physicsClientId=self.client_id,
            )

            errs = [abs(state[0] - MAX_RANGE) for state in joint_states]
            if all([err <= tolerance for err in errs]):
                print(f"Gripper is open after {100 - timeout} steps")
                break

            timeout -= 1

    def set_gripper_pose_cmd(self, pos, ori):
        x, y, z = pos
        qx, qy, qz, qw = ori

        # Convert to pybullet's euler angles.
        r = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = r.as_euler("XYZ")

        for joint_id, joint_val in enumerate([x, y, z, roll, pitch, yaw]):
            p.setJointMotorControl2(
                bodyUniqueId=self.base_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_val,
                force=100,
                physicsClientId=self.client_id,
            )

    def move(self, pos, ori, timeout=50000, threshold=5e-4):
        """Move the gripper to the desired pose.

        Args:
            pos (list): The desired position.
            ori (list): The desired orientation.
            timeout (int, optional): The maximum number of steps to take. Defaults to 5000.
            threshold (float, optional): The threshold for the position error. Defaults to 5e-4.

        Returns:
            bool: Whether the gripper reached the desired pose.
        """

        # Wait for the gripper to reach the desired pose.
        curr_pos, curr_ori = self.get_gripper_pose()

        def goal_met(curr_p, curr_o, goal_p, goal_o):
            pos_err = np.linalg.norm(np.array(curr_p) - np.array(goal_p))
            ori_err = np.linalg.norm(np.array(curr_o) - np.array(goal_o))
            return pos_err <= threshold and ori_err <= threshold

        while not goal_met(curr_pos, curr_ori, pos, ori) and timeout > 0:
            self.set_gripper_pose_cmd(pos, ori)
            p.stepSimulation(self.client_id)
            curr_pos, curr_ori = self.get_gripper_pose()

            timeout -= 1

        print(f"Goal met after {5000 - timeout} steps")

    def get_gripper_pose(self):
        # To validate the position of the gripper.....
        # Get the position of joints 0-5 (the gripper pose controllers).
        # joint_states = p.getJointStates(
        #     bodyUniqueId=self.base_id,
        #     jointIndices=[0, 1, 2, 3, 4, 5],
        #     physicsClientId=self.client_id,
        # )
        # x, y, z, rx, ry, rz = [state[0] for state in joint_states]
        # # Compute orientation quaternion from euler angles.
        # comp_pos = [x, y, z]
        # comp_ori = R.from_euler("XYZ", [rx, ry, rz]).as_quat()

        # Also get the position of the gripper base.
        link_state = p.getLinkState(
            bodyUniqueId=self.base_id,
            linkIndex=5,
            computeLinkVelocity=0,
            physicsClientId=self.client_id,
        )
        link_pos, link_ori = link_state[0], link_state[1]
        return link_pos, link_ori

    def close(self, timeout=100, tolerance=0.001):
        while timeout > 0:
            self.set_gripper_command(0.0)
            p.stepSimulation(self.client_id)

            # Check to see if the gripper is closed.
            # Make sure the joint state of both fingers are below a certain threshold.
            joint_states = p.getJointStates(
                bodyUniqueId=self.base_id,
                jointIndices=[6, 7],
                physicsClientId=self.client_id,
            )

            errs = [abs(state[0] - MIN_RANGE) for state in joint_states]
            if all([err <= tolerance for err in errs]):
                print(f"Gripper is closed after {100 - timeout} steps")
                break

            timeout -= 1
        

    def apply_force(self, force):
        body_link_pos, body_link_ori, _, _, _, _ = p.getLinkState(
            bodyUniqueId=self.base_id,
            linkIndex=0,
            computeLinkVelocity=0,
            physicsClientId=self.client_id,
        )

        p.applyExternalForce(
            self.base_id,
            linkIndex=0,
            forceObj=force,
            posObj=body_link_pos,
            flags=p.WORLD_FRAME,
            physicsClientId=self.client_id,
        )
        
    

    def release(self):
        if self.contact_const:
            p.removeConstraint(self.contact_const)
        self.activated = False
        self.contact_link_index = None
