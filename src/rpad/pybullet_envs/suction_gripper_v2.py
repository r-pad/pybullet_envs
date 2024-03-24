import os
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt

try:
    from importlib.resources import as_file, files  # type: ignore
except ImportError:
    from importlib_resources import as_file, files  # type: ignore

import pybullet as p
from scipy.spatial.transform import Rotation as R

fn = as_file(files("rpad.pybullet_envs").joinpath("assets/suction"))

with fn as f:
    SUCTION_URDF = os.path.join(f, "suction_with_mount_no_collision.urdf")


class FloatingSuctionGripperV2:
    """This is a suction gripper modeled after the UMPNet suction gripper.

    The main difference is that we actually model the gripper as mounted to the
    world frame, and use a 6dof joint (a ball joint for orientation, and 3 sliding joints
    for XYZ) in order to control the thing. The mount in the world frame is WELDED to the
    ground, so can't move. This allows us to nicely do like, position+velocity control
    of the gripper itself rather than the super hacky thing of resetting the state
    of the gripper to having a certain velocity at every time step. Sad Sad Sad.
    """

    def __init__(self, client_id):
        self.client_id = client_id
        # load the suction gripper
        self.mount_id = p.loadURDF(
            SUCTION_URDF,
            useFixedBase=True,
            globalScaling=1.0,
            physicsClientId=client_id,
        )
        self.gripper_pos = None
        self.gripper_ori = None
        # control attributes
        self.gains_pos = np.array([0.02] * 3)
        self.gains_ori = np.array([0.05] * 3)
        self.forces_pos = np.array([1000] * 3)
        self.forces_ori = np.array([10] * 3)
        self.pull_speed = 0.05
        # contact attributes
        self.constraint_force = 10000
        self.activated = False
        self.contact_const = None
        self.tip_link_index = 4
        self.contact_link_index = None

    def set_state(self, pos: npt.NDArray, ori: npt.NDArray):
        """Sets the pose of the gripper in the world frame."""
        for ji, js in enumerate(pos):
            p.resetJointState(self.mount_id, ji, js, 0, self.client_id)
        p.resetJointStateMultiDof(self.mount_id, 3, ori, [0, 0, 0], self.client_id)
        self.gripper_pos = pos
        self.gripper_ori = ori

    def update_state(self):
        """Updates the internal state of the gripper based on simulation environment."""
        self.gripper_pos = np.array(
            [s[0] for s in p.getJointStates(self.mount_id, [0, 1, 2], self.client_id)]
        )
        self.gripper_ori = p.getJointStateMultiDof(self.mount_id, 3, self.client_id)[0]

    def detect_contact(self, obj_id: int, obj_link_index: int):
        """Helper function to detect contact; this will return the first contact point on
        the input articulated part, None otherwise"""
        points = p.getContactPoints(
            bodyA=self.mount_id,
            bodyB=obj_id,
            linkIndexA=self.tip_link_index,
            linkIndexB=obj_link_index,
            physicsClientId=self.client_id,
        )
        if len(points) != 0:
            return points[0]

    def activate(self, obj_id: int, obj_link_index: int):
        """This should activate the suction gripper."""
        point = self.detect_contact(obj_id, obj_link_index)
        if point is not None:
            contact_pos_on_A, contact_pos_on_B = point[5], point[6]
            obj_id, contact_link = point[2], point[4]
            # self.debug_point(contact_pos_on_A, True)
            # self.debug_point(contact_pos_on_B, False)

            # Describe the contact point in the TIP FRAME.
            base_link_pos, base_link_ori, _, _, _, _ = p.getLinkState(
                bodyUniqueId=self.mount_id,
                linkIndex=self.tip_link_index,
                computeLinkVelocity=0,
                physicsClientId=self.client_id,
            )
            T_world_tip = base_link_pos, base_link_ori
            T_tip_world = p.invertTransform(*T_world_tip)
            T_world_contact = contact_pos_on_A, base_link_ori
            T_tip_contact = p.multiplyTransforms(*T_tip_world, *T_world_contact)

            # Describe the contact point in the OBJ FRAME. Note that we use the
            # orientation of the gripper
            objcom_link_pos, objcom_link_ori, _, _, _, _ = p.getLinkState(
                bodyUniqueId=obj_id,
                linkIndex=contact_link,
                computeLinkVelocity=0,
                physicsClientId=self.client_id,
            )
            T_world_objcom = objcom_link_pos, objcom_link_ori
            T_objcom_world = p.invertTransform(*T_world_objcom)
            T_obj_contact = p.multiplyTransforms(*T_objcom_world, *T_world_contact)

            # Short names so we can debug later (parent frame; child frame),
            pfp, pfo = T_tip_contact
            cfp, cfo = T_obj_contact

            self.contact_const = p.createConstraint(
                parentBodyUniqueId=self.mount_id,
                parentLinkIndex=self.tip_link_index,
                childBodyUniqueId=obj_id,
                childLinkIndex=contact_link,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                # parentFramePosition=[0, 0, 0],
                parentFramePosition=pfp,
                parentFrameOrientation=pfo,
                childFramePosition=cfp,
                childFrameOrientation=cfo,
                physicsClientId=self.client_id,
            )
            p.changeConstraint(
                self.contact_const,
                maxForce=self.constraint_force,
                physicsClientId=self.client_id,
            )
            self.activated = True
            self.contact_link_index = contact_link
        # else:
        # print("not in contact - no valid points to attach to.")

    def release(self):
        """This should deactivate the suction gripper."""
        if self.contact_const is not None:
            p.removeConstraint(self.contact_const, self.client_id)
            self.contact_const = None
        self.activated = False
        self.contact_link_index = None

    def get_move_fn(
        self, goal_pos: npt.NDArray, goal_ori: npt.NDArray
    ) -> Callable[[], Tuple[npt.NDArray, bool]]:
        """Returns a function which, when called, gives you the next control to execute. This should happen
        at every time step."""

        def move_fn() -> Tuple[npt.NDArray, bool]:
            """Calcuate whatever error you need here to get the signal. Returns zeros and True if the goal is reached."""
            # TODO: how to determine if goal is reached? is tolerance good enough?
            # for now, just returning the same control signal as v1
            ctrl = np.zeros(8)
            # second flag is 0 for move
            ctrl[2:5] = goal_pos
            ctrl[5:] = goal_ori
            return ctrl, False

        return move_fn

    def set_move_cmds(self, ctrl):
        """Sets the control signal for the move command (aka sets the motors)."""
        goal_pos = ctrl[2:5]
        goal_ori = ctrl[5:]
        p.setJointMotorControlArray(
            bodyUniqueId=self.mount_id,
            jointIndices=[0, 1, 2],
            controlMode=p.POSITION_CONTROL,
            targetPositions=goal_pos,
            positionGains=self.gains_pos,
            forces=self.forces_pos,
            physicsClientId=self.client_id,
        )
        p.setJointMotorControlMultiDof(
            bodyUniqueId=self.mount_id,
            jointIndex=3,
            controlMode=p.POSITION_CONTROL,
            targetPosition=R.align_vectors(
                -goal_ori.reshape((1, 3)), np.array([[0, 0, -1]])
            )[0].as_quat(),
            force=self.forces_ori,
            physicsClientId=self.client_id,
        )

    # TODO: control signal should have one extra parameter to handle activation and deactivation
    def get_pull_fn(self, direction) -> Callable[[], Tuple[npt.NDArray, bool]]:
        """Returns a function which, when called, gives you the next control to execute for pulling.
        This should happen at every time step."""

        def pull_fn() -> Tuple[npt.NDArray, bool]:
            """Calcuate whatever error you need here to get the signal. Returns zeros and True if the goal is reached."""
            # ctrl = np.zeros(8)
            ctrl = np.ones(8)
            # second flag is 1 for pull
            ctrl[2:5] = 0
            ctrl[5:] = direction
            return ctrl, False

        return pull_fn

    def set_pull_cmds(self, ctrl):
        """Sets the control signal for the pull command (aka sets the motors)."""
        # p.resetJointState(
        #     self.mount_id,
        #     0,
        #     targetValue=self.gripper_pos[0],
        #     targetVelocity=pull_speed * direction[0],
        #     physicsClientId=self.client_id,
        # )
        # p.resetJointState(
        #     self.mount_id,
        #     1,
        #     targetValue=self.gripper_pos[1],
        #     targetVelocity=pull_speed * direction[1],
        #     physicsClientId=self.client_id,
        # )
        # p.resetJointState(
        #     self.mount_id,
        #     2,
        #     targetValue=self.gripper_pos[2],
        #     targetVelocity=pull_speed * direction[2],
        #     physicsClientId=self.client_id,
        # )
        if self.activated:
            direction = ctrl[5:]
            goal_pos = self.gripper_pos + self.pull_speed * direction
            p.setJointMotorControlArray(
                bodyUniqueId=self.mount_id,
                jointIndices=[0, 1, 2],
                controlMode=p.POSITION_CONTROL,
                targetPositions=goal_pos,
                positionGains=self.gains_pos,
                forces=self.forces_pos,
                physicsClientId=self.client_id,
            )
        else:
            print("Cannot pull - not attached.")

    def debug_joint_forces(self):
        js = p.getJointStatesMultiDof(self.mount_id, [0, 1, 2, 3], self.client_id)
        print(
            f'joint {0}: {", ".join(f"{f:.2f}" for f in js[0][3])}   joint {1}: {", ".join(f"{f:.2f}" for f in js[1][3])}   joint {2}: {", ".join(f"{f:.2f}" for f in js[2][3])}    joint {3}: {", ".join(f"{f:.2f}" for f in js[3][3])} '
        )

    def debug_point(self, point, switch):
        d = 0.1 if switch else -0.1
        p.addUserDebugLine(
            lineFromXYZ=point,
            lineToXYZ=point + np.array([0, 0, d]),
            lineColorRGB=[1, 0, 0],
            lineWidth=5,
            physicsClientId=self.client_id,
        )
        p.addUserDebugLine(
            lineFromXYZ=point,
            lineToXYZ=point + np.array([0, d, 0]),
            lineColorRGB=[0, 1, 0],
            lineWidth=5,
            physicsClientId=self.client_id,
        )
        p.addUserDebugLine(
            lineFromXYZ=point,
            lineToXYZ=point + np.array([d, 0, 0]),
            lineColorRGB=[0, 0, 1],
            lineWidth=5,
            physicsClientId=self.client_id,
        )
