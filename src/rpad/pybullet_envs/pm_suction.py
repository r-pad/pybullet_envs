import copy
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
import pybullet as p
import rpad.visualize_3d.plots as vpl
from rpad.partnet_mobility_utils.articulate import articulate_joint
from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv, PybulletRenderer
from scipy.spatial.transform import Rotation as R

from rpad.pybullet_envs.suction_gripper import FloatingSuctionGripper


def reindex(seg, segmap, obj_id):
    pc_seg_obj = np.ones_like(seg) * -1
    for k, (body, link) in segmap.items():
        if body == obj_id:
            ixs = seg == k
            pc_seg_obj[ixs] = link
    return pc_seg_obj


class PMSuctionEnv:
    def __init__(
        self,
        obj_id: str,
        pm_dataset_path: str,
        camera_pos: List = [-2, 0, 2],
        gui=False,
    ) -> None:
        # Create a core environment that we'll manipulate.
        self._core_env = PMRenderEnv(obj_id, pm_dataset_path, camera_pos, gui=gui)
        self.gui = gui

        # Add a suction gripper.
        self.gripper = FloatingSuctionGripper(self._core_env.client_id)

        self.gripper.set_pose(
            [-1, 0.6, 0.8], p.getQuaternionFromEuler([0, np.pi / 2, 0])
        )

    def teleport_and_approach(self, point, contact_vector, standoff_d: float = 0.2):
        # Normalize contact vector.
        contact_vector = (
            contact_vector / np.linalg.norm(contact_vector, axis=-1)
        ).astype(float)

        p_teleport = (point + contact_vector * standoff_d).astype(float)

        e_z_init: npt.NDArray[np.float64] = np.asarray([0, 0, 1.0]).astype(float)
        e_y = -contact_vector
        e_x = np.cross(-contact_vector, e_z_init)
        e_x = e_x / np.linalg.norm(e_x, axis=-1)
        e_z = np.cross(e_x, e_y)
        e_z = e_z / np.linalg.norm(e_z, axis=-1)
        R_teleport = np.stack([e_x, e_y, e_z], axis=1)
        R_gripper = np.asarray(
            [
                [1, 0, 0],
                [0, 0, 1.0],
                [0, -1.0, 0],
            ]
        )
        # breakpoint()
        o_teleport = R.from_matrix(R_teleport @ R_gripper).as_quat()

        self.gripper.set_pose(p_teleport, o_teleport)

        contact = self.gripper.detect_contact(self._core_env.obj_id)
        max_steps = 500
        curr_steps = 0
        self.gripper.set_velocity(-contact_vector * 0.4, [0, 0, 0])
        while not contact and curr_steps < max_steps:
            p.stepSimulation(self._core_env.client_id)
            curr_steps += 1
            if self.gui:
                time.sleep(1 / 240.0)
            if curr_steps % 10 == 0:
                contact = self.gripper.detect_contact(self._core_env.obj_id)

        if contact:
            print("contact detected")

        return contact

    def attach(self):
        self.gripper.activate(self._core_env.obj_id)

    def pull(self, direction, n_steps: int = 100):
        direction = np.asarray(direction)
        direction = direction / np.linalg.norm(direction, axis=-1)
        # breakpoint()
        for _ in range(n_steps):
            self.gripper.set_velocity(direction * 0.4, [0, 0, 0])
            p.stepSimulation(self._core_env.client_id)
            if self.gui:
                time.sleep(1 / 240.0)

    def get_joint_value(self, target_link: str):
        link_index = self._core_env.link_name_to_index[target_link]
        state = p.getJointState(
            self._core_env.obj_id, link_index, self._core_env.client_id
        )
        joint_pos = state[0]
        return joint_pos

    def detect_success(self, target_link: str):
        link_index = self._core_env.link_name_to_index[target_link]
        info = p.getJointInfo(
            self._core_env.obj_id, link_index, self._core_env.client_id
        )
        lower, upper = info[8], info[9]
        curr_pos = self.get_joint_value(target_link)

        print(f"lower: {lower}, upper: {upper}, curr: {curr_pos}")

        sign = -1 if upper < 0 else 1
        return sign * (upper - curr_pos) < 0.001

    def reset(self):
        pass

    def set_joint_state(self, state):
        return self._core_env.set_joint_angles(state)

    def render(self, filter_nonobj_pts=False, n_pts=1200):
        return self._core_env.render()


@dataclass
class TrialResult:
    success: bool
    init_angle: float
    final_angle: float

    # UMPNet metric goes here
    metric: float


def run_trial(
    env: PMSuctionEnv,
    raw_data: PMObject,
    target_link: str,
    model,
    n_steps: int = 30,
    n_pts: int = 1200,
    traj_len: int = 1,  # By default, only move one step
) -> TrialResult:
    # First, reset the environment.
    env.reset()

    # Sometimes doors collide with themselves. It's dumb.
    # if raw_data.category == "Door" and raw_data.semantics.by_name(target_link).type == "hinge":
    #     env.set_joint_state(target_link, 0.2)

    if raw_data.semantics.by_name(target_link).type == "hinge":
        target_joint = raw_data.obj.get_joint_by_child(target_link).name
        env.set_joint_state({target_joint: 0.05})

    # Predict the flow on the observation.
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    rgb, depth, seg, P_cam, P_world, pc_seg, segmap = (
        pc_obs["rgb"],
        pc_obs["depth"],
        pc_obs["seg"],
        pc_obs["P_cam"],
        pc_obs["P_world"],
        pc_obs["pc_seg"],
        pc_obs["segmap"],
    )
    pc_seg = reindex(pc_seg, segmap, env._core_env.obj_id)

    pred_flow = model(copy.deepcopy(pc_obs))

    # flow_fig(torch.from_numpy(P_world), pred_flow, sizeref=0.1, use_v2=True).show()
    # breakpoint()

    # Filter down just the points on the target link.
    link_ixs = pc_seg == env._core_env.link_name_to_index[target_link]

    # breakpoint()

    assert link_ixs.any()

    # The attachment point is the point with the highest flow.
    best_flow_ix = np.linalg.norm(pred_flow[link_ixs], axis=-1).argmax()
    best_flow = pred_flow[link_ixs][best_flow_ix]
    best_point = P_world[link_ixs][best_flow_ix]

    # fig = flow_fig(P_world, pred_flow)
    # fig.add_trace(vector(*best_point, *best_flow, scene="scene1", color="yellow"))
    # fig.show()

    # Teleport to an approach pose, approach, the object and grasp.
    contact = env.teleport_and_approach(best_point, best_flow)

    if not contact:
        print("No contact detected")
        return TrialResult(
            success=False,
            init_angle=0,
            final_angle=0,
            metric=0,
        )

    env.attach()

    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    success = False

    for i in range(n_steps):
        # Predict the flow on the observation.
        pred_trajectory = model(pc_obs)
        pred_trajectory = pred_trajectory.reshape(
            pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
        )

        for traj_step in range(traj_len):
            pred_flow = pred_trajectory[:, traj_step, :]

            rgb, depth, seg, P_cam, P_world, pc_seg, segmap = (
                pc_obs["rgb"],
                pc_obs["depth"],
                pc_obs["seg"],
                pc_obs["P_cam"],
                pc_obs["P_world"],
                pc_obs["pc_seg"],
                pc_obs["segmap"],
            )
            pc_seg = reindex(pc_seg, segmap, env._core_env.obj_id)

            # Filter down just the points on the target link.
            link_ixs = pc_seg == env._core_env.link_name_to_index[target_link]
            assert link_ixs.any()

            # Get the best direction.
            best_flow_ix = np.linalg.norm(pred_flow[link_ixs], axis=-1).argmax()
            best_flow = pred_flow[link_ixs][best_flow_ix]

            # Perform the pulling.
            env.pull(best_flow)

            success = env.detect_success(target_link)

            if success:
                break

            pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)

    init_angle = 0.0
    final_angle = 0.0
    upper_bound = 0.0
    metric = 0.0

    return TrialResult(
        success=success,
        init_angle=init_angle,
        final_angle=final_angle,
        metric=metric,
    )


# COPIED FROM FLOWBOT, DON'T WANT TO ADD FLOWBOT AS A DEPENDENCY...
def compute_normalized_flow(
    P_world, T_world_base, current_jas, pc_seg, labelmap, pm_raw_data, linknames
):
    """Compute normalized flow for an object, based on its kinematics.

    Args:
        P_world (npt.NDArray[np.float32]): Point cloud render of the object in the world frame.
        T_world_base (npt.NDArray[np.float32]): The pose of the base link in the world frame.
        current_jas (Dict[str, float]): The current joint angles (easy to acquire from the render that created the points.)
        pc_seg (npt.NDArray[np.uint8]): The segmentation labels of each point.
        labelmap (Dict[str, int]): Map from the link name to segmentation name.
        pm_raw_data (PMObject): The object description, essentially providing the kinematic structure of the object.
        linknames (Union[Literal['all'], Sequence[str]], optional): The names of the links for which to
            compute flow. Defaults to "all", which will articulate all of them.

    Returns:
        npt.NDArray[np.float32]: _description_
    """

    # We actuate all links.
    if linknames == "all":
        joints = pm_raw_data.semantics.by_type("slider")
        joints += pm_raw_data.semantics.by_type("hinge")
        linknames = [joint.name for joint in joints]

    flow = np.zeros_like(P_world)

    for linkname in linknames:
        P_world_new = articulate_joint(
            pm_raw_data,
            current_jas,
            linkname,
            0.01,  # Articulate by only a little bit.
            P_world,
            pc_seg,
            labelmap,
            T_world_base,
        )
        link_flow = P_world_new - P_world
        flow += link_flow

    largest_mag: float = np.linalg.norm(flow, axis=-1).max()

    normalized_flow = flow / (largest_mag + 1e-6)

    return normalized_flow


# TODO: this probably needs to be renamed, but PMSuctionEnv is already taken^
class PMSuctionDemoEnv:
    def __init__(self, obj_id, pm_dataset_path, gripper_path, gui):
        self.obj_id = obj_id
        self.obj = PMObject(pm_dataset_path / obj_id)
        self.renderer = PybulletRenderer()
        # initializing SuctionEnv (TODO: probably don't actually need this - everything is re-implemented)
        self.suction_env = PMSuctionEnv(obj_id, pm_dataset_path, gui=gui)
        # gripper mount pose in world frame
        self._mount_pos = None
        self._mount_ori = None
        # TODO: removing default gripper from suction env - maybe just re-implement different version of PMSuctionEnv?
        p.removeBody(
            self.suction_env.gripper.base_id, self.suction_env._core_env.client_id
        )
        p.removeBody(
            self.suction_env.gripper.body_id, self.suction_env._core_env.client_id
        )
        # loading gripper with mount
        self.mount_id = p.loadURDF(
            gripper_path,
            useFixedBase=True,
            # basePosition=self._mount_base_pos,
            # baseOrientation=p.getQuaternionFromEuler(self._mount_base_ori),
            globalScaling=1,
            physicsClientId=self.suction_env._core_env.client_id,
        )
        # contact attributes
        self.constraint_force = 10000
        self.gripper_contact_link = 4
        self.activated = False
        self.contact_const = None
        self.contact_link_index = None
        # goal/demo attributes
        self.goal = p.getJointInfo(
            self.suction_env._core_env.obj_id, 1, self.suction_env._core_env.client_id
        )[9]
        self.success_threshold = 0.9

    def reset(self, pos_init, ori_init):
        # deactivating gripper - this will handle updates to self.activated and self.contact_link_index
        self.release()
        # re-setting all object joints
        obj_id = self.suction_env._core_env.obj_id
        for ji in range(p.getNumJoints(obj_id)):
            p.resetJointState(obj_id, ji, 0, 0, self.suction_env._core_env.client_id)
        # re-setting gripper with initial pose and orientation
        for ji, js in enumerate(pos_init):
            p.resetJointState(
                self.mount_id, ji, js, 0, self.suction_env._core_env.client_id
            )
        p.resetJointStateMultiDof(
            self.mount_id, 3, ori_init, [0, 0, 0], self.suction_env._core_env.client_id
        )
        self._mount_pos = pos_init
        self._mount_ori = ori_init
        p.stepSimulation(self.suction_env._core_env.client_id)
        return

    def render(self):
        obs = self.suction_env.render()
        pos = obs["P_world"]
        seg = obs["pc_seg"]
        fig = vpl.segmentation_fig(pos[::10], seg[::10])
        # plt.imshow(obs["rgb"])
        fig.show()

    def detect_contact(self):
        """Helper function to detect contact; this will return the first contact point on the articulated part (door), or None otherwise"""
        points = p.getContactPoints(
            bodyA=self.mount_id,
            bodyB=self.suction_env._core_env.obj_id,
            linkIndexA=4,
            physicsClientId=self.suction_env._core_env.client_id,
        )
        # search through each point, return the first valid point
        for point in points:
            if point[self.gripper_contact_link] == 1:
                return point

    def move(self, t_pos, t_ori, force):
        # # target position (from delta)
        # t_pos = self._mount_pos + np.array(d_pos)
        # # target orientation - converting rotation to mount frame, then applying to current orientation transform (from delta)
        # d_ori_R = R.from_euler('xyz', d_ori).as_matrix()
        # gripper_R = R.from_quat(self._mount_ori).as_matrix()
        # t_ori = R.from_matrix(gripper_R @ gripper_R.T @ d_ori_R @ gripper_R).as_quat()
        # self.debug_joint_forces()

        # joint control
        # TODO: is there some mix of velocity/position gains that makes the trajectory smoother?
        p.setJointMotorControlArray(
            self.mount_id,
            [0, 1, 2],
            p.POSITION_CONTROL,
            targetPositions=t_pos,
            # targetVelocities=[0]*3,
            # velocityGains=[10, 10, 10],
            forces=[force] * 3,
            physicsClientId=self.suction_env._core_env.client_id,
        )
        p.setJointMotorControlMultiDof(
            self.mount_id,
            3,
            p.POSITION_CONTROL,
            targetPosition=R.align_vectors(
                -t_ori.reshape((1, 3)), np.array([[0, 0, -1]])
            )[0].as_quat(),
            force=[10, 10, 10],
            physicsClientId=self.suction_env._core_env.client_id,
        )
        # self.debug_joint_forces()
        max_steps = 1000
        curr_steps = 0

        contact = self.detect_contact() is not None
        # contact = False
        while not contact and curr_steps < max_steps:
            curr_steps += 1
            # self.debug_joint_forces()
            p.stepSimulation(self.suction_env._core_env.client_id)
            if self.suction_env.gui:
                time.sleep(1 / 240.0)
            if curr_steps % 1 == 0:
                # contact check
                contact = self.detect_contact() is not None
        if contact:
            print("contact detected!")
        # updating joint states
        self._mount_pos = np.array(
            [
                s[0]
                for s in p.getJointStates(
                    self.mount_id, [0, 1, 2], self.suction_env._core_env.client_id
                )
            ]
        )
        self._mount_ori = p.getJointStateMultiDof(
            self.mount_id, 3, self.suction_env._core_env.client_id
        )[0]
        return

    def attach(self, constraint_force):
        if not self.activated:
            # points = p.getContactPoints(bodyA=self.mount_id, bodyB=self.suction_env._core_env.obj_id,
            #                             linkIndexA=4, physicsClientId=self.suction_env._core_env.client_id)
            contact = self.detect_contact()
            # if len(points) != 0:
            if contact is not None:
                # We'll choose the first point as the contact.
                point = contact
                contact_pos_on_A, contact_pos_on_B = point[5], point[6]
                obj_id, contact_link = point[2], point[4]

                # Describe the contact point in the TIP FRAME.
                base_link_pos, base_link_ori, _, _, _, _ = p.getLinkState(
                    bodyUniqueId=self.mount_id,
                    linkIndex=self.gripper_contact_link,
                    computeLinkVelocity=0,
                    physicsClientId=self.suction_env._core_env.client_id,
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
                    physicsClientId=self.suction_env._core_env.client_id,
                )
                T_world_objcom = objcom_link_pos, objcom_link_ori
                T_objcom_world = p.invertTransform(*T_world_objcom)
                T_obj_contact = p.multiplyTransforms(*T_objcom_world, *T_world_contact)

                # Short names so we can debug later (parent frame; child frame),
                pfp, pfo = T_tip_contact
                cfp, cfo = T_obj_contact

                self.contact_const = p.createConstraint(
                    parentBodyUniqueId=self.mount_id,
                    parentLinkIndex=self.gripper_contact_link,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=contact_link,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    # parentFramePosition=[0, 0, 0],
                    parentFramePosition=pfp,
                    parentFrameOrientation=pfo,
                    childFramePosition=cfp,
                    childFrameOrientation=cfo,
                    physicsClientId=self.suction_env._core_env.client_id,
                )
                p.changeConstraint(
                    self.contact_const,
                    maxForce=constraint_force,
                    physicsClientId=self.suction_env._core_env.client_id,
                )
                self.activated = True
                self.contact_link_index = contact_link
            else:
                print("not in contact - no valid points to attach to.")

    def release(self):
        if self.contact_const:
            p.removeConstraint(self.contact_const, self.suction_env._core_env.client_id)
            self.contact_const = None
        self.activated = False
        self.contact_link_index = None

    def pull(self, direction, speed_factor=0.2, n_steps=100):
        if self.activated:
            direction = np.asarray(direction)
            direction = direction / np.linalg.norm(direction, axis=-1)
            # p.setJointMotorControlArray(
            #         self.mount_id,
            #         [0, 1, 2],
            #         p.VELOCITY_CONTROL,
            #         targetVelocities=direction*0.1,
            #         #velocityGains=[velocityGain] * 3,
            #         physicsClientId=self.suction_env._core_env.client_id,
            #         forces=[force]*3,
            #     )
            # p.setJointMotorControlArray(
            #     self.mount_id,
            #     [0, 1, 2],
            #     p.POSITION_CONTROL,
            #     targetPositions=self._mount_pos + 0.05*direction,
            #     physicsClientId=self.suction_env._core_env.client_id,
            #         forces=[force]*3,
            # )
            # p.setJointMotorControlMultiDof(
            #         self.mount_id,
            #         3,
            #         p.POSITION_CONTROL,
            #         targetPosition=R.align_vectors(-direction.reshape((1, 3)), np.array([[0, 0, -1]]))[0].as_quat(),
            #         physicsClientId=self.suction_env._core_env.client_id,
            #     )

            # get joint info, lower, upper (TODO: define this as a class attribute?)
            # goal = p.getJointInfo(self.suction_env._core_env.obj_id, 1, self.suction_env._core_env.client_id)[9]
            for _ in range(n_steps):
                # set and simulate velocity
                p.resetJointState(
                    self.mount_id,
                    0,
                    self._mount_pos[0],
                    speed_factor * direction[0],
                    self.suction_env._core_env.client_id,
                )
                p.resetJointState(
                    self.mount_id,
                    1,
                    self._mount_pos[1],
                    speed_factor * direction[1],
                    self.suction_env._core_env.client_id,
                )
                p.resetJointState(
                    self.mount_id,
                    2,
                    self._mount_pos[2],
                    speed_factor * direction[2],
                    self.suction_env._core_env.client_id,
                )
                p.stepSimulation(self.suction_env._core_env.client_id)
                # update joint states
                self._mount_pos = np.array(
                    [
                        s[0]
                        for s in p.getJointStates(
                            self.mount_id,
                            [0, 1, 2],
                            self.suction_env._core_env.client_id,
                        )
                    ]
                )
                self._mount_ori = p.getJointStateMultiDof(
                    self.mount_id, 3, self.suction_env._core_env.client_id
                )[0]
                if self.suction_env.gui:
                    time.sleep(1 / 240.0)
                # goal check
                current = p.getJointState(
                    self.suction_env._core_env.obj_id,
                    1,
                    self.suction_env._core_env.client_id,
                )[0]
                if current / self.goal > self.success_threshold:
                    return True
        else:
            print("cannot pull - not attached")
        return False

    def select_point(self):
        joints = self.suction_env._core_env.get_joint_angles()
        # TODO: this is a hard-coded fix (joint is neither prismatic nor revolute)
        del joints["joint_3"]
        # render with random camera for now
        res = self.renderer.render(self.obj, joints=joints, camera_xyz="random")
        # TODO: save time by passing in specific link instead of "all"
        flow = compute_normalized_flow(
            P_world=res["pos"],
            T_world_base=res["T_world_base"],
            current_jas=res["angles"],
            pc_seg=res["seg"],
            labelmap=res["labelmap"],
            pm_raw_data=self.obj,
            linknames="all",
        )
        # find position and orientation of point with max flow
        link_flow = flow[res["seg"] == 1]
        link_pc = res["pos"][res["seg"] == 1]
        idx = np.argmax(np.linalg.norm(link_flow, axis=1, keepdims=True))
        return link_pc[idx], link_flow[idx]

    def generate_demo(self, pull_iters=25):
        # move and attach to point with max flow
        position_start, direction_start = self.select_point()
        self.move(position_start, direction_start, 1000)
        self.attach(self.constraint_force)
        # continuously pull in direction of flow
        success = False
        if self.activated:
            for _ in range(pull_iters):
                _, direction = self.select_point()
                success = self.pull(direction)
                if success:
                    break
        return success

    def debug_joint_forces(self):
        js = p.getJointStatesMultiDof(
            self.mount_id, [0, 1, 2, 3], self.suction_env._core_env.client_id
        )
        print(
            f'joint {0}: {", ".join(f"{f:.2f}" for f in js[0][3])}   joint {1}: {", ".join(f"{f:.2f}" for f in js[1][3])}   joint {2}: {", ".join(f"{f:.2f}" for f in js[2][3])}    joint {3}: {", ".join(f"{f:.2f}" for f in js[3][3])} '
        )
        return
