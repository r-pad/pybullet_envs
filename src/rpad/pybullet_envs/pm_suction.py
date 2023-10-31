import copy
import time
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pybullet as p
from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
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


class PMSuctionGymEnv(gym.Env):
    # TODO(eric): Implement the following functions.
    pass
