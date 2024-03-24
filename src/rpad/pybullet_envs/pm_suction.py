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

from rpad.pybullet_envs.flowbot_utils import compute_normalized_flow
from rpad.pybullet_envs.suction_gripper import FloatingSuctionGripper
from rpad.pybullet_envs.suction_gripper_v2 import FloatingSuctionGripperV2


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


# max simulation steps for control
TIMEOUT = 1000


class PMSuctionDemoEnv:
    def __init__(self, obj_id, pm_dataset_path, gui, use_egl=False):
        self.obj_id_str = obj_id  # this is PM dataset id
        self.obj = PMObject(pm_dataset_path / obj_id)
        self.renderer = PybulletRenderer()
        self.gui = gui
        self._core_env = PMRenderEnv(
            obj_id, pm_dataset_path, [-2, 0, 2], gui=gui, use_egl=use_egl
        )
        self.client_id = self._core_env.client_id
        self.obj_id = self._core_env.obj_id  # this is environment id
        self.gripper = FloatingSuctionGripperV2(self.client_id)
        # goal attributes
        self.obj_link = 1
        self.goal = p.getJointInfo(self.obj_id, 1, self.client_id)[9]
        self.success_threshold = 0.9

    def reset(self, pos_init, ori_init, obj_pos=None, obj_ori=None):
        # printing location of obj_id
        if obj_pos is not None:
            p.resetBasePositionAndOrientation(
                self.obj_id, obj_pos, obj_ori, self.client_id
            )
        self.gripper.release()
        # re-setting all object joints
        obj_id = self.obj_id
        for ji in range(p.getNumJoints(obj_id)):
            p.resetJointState(obj_id, ji, 0, 0, self.client_id)
        # re-setting gripper with initial pose and orientation
        self.gripper.set_state(pos_init, ori_init)
        # TODO: is this necessary?
        p.stepSimulation(self.client_id)
        return self.get_obs()

    def goal_check(self):
        # returns success, and normalized distance
        current = p.getJointState(
            self.obj_id,
            1,
            self.client_id,
        )[0]
        goal_reached = current / self.goal > self.success_threshold
        normalized_dist = 1 - current / (self.goal * self.success_threshold)
        return goal_reached, normalized_dist

    def render(self):
        obs = self._core_env.render()
        pos = obs["P_world"]
        seg = obs["pc_seg"]
        fig = vpl.segmentation_fig(pos[::10], seg[::10])
        # plt.imshow(obs["rgb"])
        fig.show()

    def move(self, goal_pos, goal_ori, demo=None):
        # TODO: take demo as input, update at each timestep - record success?
        move_fn = self.gripper.get_move_fn(goal_pos, goal_ori)
        ctrl, goal_reached = move_fn()

        if demo is not None:
            # TODO: orientation representations are split here - obs is quat, ctrl is euler
            # TODO: this update needs to be generalized to all demo types
            demo.append({"obs": self.get_obs(), "action": ctrl})
        # contact check
        contact = self.gripper.detect_contact(self.obj_id, self.obj_link)
        curr_steps = 0
        while not goal_reached and contact is None and curr_steps < TIMEOUT:
            # control, sim
            self.gripper.set_move_cmds(ctrl)
            p.stepSimulation(self.client_id)
            if self.gui:
                time.sleep(1 / 240.0)
            # check for contact and update gripper state
            contact = self.gripper.detect_contact(self.obj_id, self.obj_link)
            self.gripper.update_state()
            if contact is not None:
                print("Contact detected.")
                break
            # update control signal
            ctrl, goal_reached = move_fn()
            if demo is not None:
                demo.append({"obs": self.get_obs(), "action": ctrl})
            curr_steps += 1
        # should return demo, and contact
        return demo, contact is not None

    def pull(self, direction, demo=None, n_steps=100):
        if self.gripper.activated:
            direction = np.asarray(direction)
            direction = direction / np.linalg.norm(direction, axis=-1)
            pull_fn = self.gripper.get_pull_fn(direction)
            ctrl, goal_reached = pull_fn()
            if demo is not None:
                demo.append({"obs": self.get_obs(), "action": ctrl})

            for _ in range(n_steps):
                # control, sim
                self.gripper.set_pull_cmds(ctrl)
                # self.gripper.debug_joint_forces()
                p.stepSimulation(self.client_id)
                if self.gui:
                    time.sleep(1 / 240.0)
                # update gripper state and check for goal
                self.gripper.update_state()
                if self.goal_check()[0]:
                    return demo, True
                # update control signal
                ctrl, goal_reached = pull_fn()
                if demo is not None:
                    demo.append({"obs": self.get_obs(), "action": ctrl})
            return demo, False
        else:
            print("Cannot pull - not attached.")

    def select_point(self):
        joints = self._core_env.get_joint_angles()
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

    def generate_demo(self, pull_iters=25, contact_point=None):
        demo = []
        # move to point with max flow, update demo with initial state
        if contact_point is None:
            position_start, direction_start = self.select_point()
            # TODO: hard-coded fix to move away from edge
            position_start[2] = position_start[2] - 0.1
        else:
            position_start = contact_point[0:3]
            direction_start = contact_point[3:6]
        print("Demo contact point: ", position_start)
        demo, contact = self.move(position_start, direction_start, demo)
        if not contact:
            print("Demo discarded: no contact detected.")
            return False, None
        # attach to object, update demo
        demo.append(
            {"obs": self.get_obs(), "action": np.array([1, 0, 0, 0, 0, 0, 0, 0])}
        )
        self.gripper.activate(self.obj_id, self.obj_link)
        # continuously pull in direction of flow
        success = False
        if self.gripper.activated:
            for _ in range(pull_iters):
                _, direction = self.select_point()
                # pull object, update demo
                # TODO: might need to add a manual check if the demo is valid or not
                demo, success = self.pull(direction, demo)
                if success:
                    break
        if success:
            print("Demo successful.")
            # return success, np.stack(demo, axis=0)
            return success, demo
        else:
            print("Demo discarded: pull failed.")
            return success, None

    def get_obs(self):
        # camera render
        camera_obs = self._core_env.render()
        rgb = camera_obs["rgb"]
        depth = camera_obs["depth"]
        pc_world = camera_obs["P_world"]
        pc_cam = camera_obs["P_cam"]
        pc_seg = camera_obs["pc_seg"]
        segmap = camera_obs["segmap"]
        # re-indexing point cloud (0: object, 1: object link, 2: gripper)
        pc_seg_scene = np.ones_like(pc_seg) * -1
        for k, (body, link) in segmap.items():
            if body == self.gripper.mount_id:
                # segment gripper
                ixs = pc_seg == k
                pc_seg_scene[ixs] = 2
            elif body == self.obj_id:
                ixs = pc_seg == k
                # link, or otherwise
                if link == self.obj_link:
                    pc_seg_scene[ixs] = 1
                else:
                    pc_seg_scene[ixs] = 0
        # object joint angle
        ja = p.getJointState(self.obj_id, 1, self.client_id)[0]
        # contact flag
        activated = self.gripper.activated
        contact = self.gripper.detect_contact(self.obj_id, self.obj_link) is not None
        contact_flag = int(activated or contact)
        obs = {
            "pc": pc_cam,
            "seg": pc_seg_scene,
            "joint_angle": ja,
            "contact_flag": contact_flag,
            "gripper_pos": self.gripper.gripper_pos,
            "gripper_ori": self.gripper.gripper_ori,
            "activated": self.gripper.activated,
        }
        return obs

    def get_state_obs(self):
        # object joint angle
        ja = p.getJointState(self.obj_id, 1, self.client_id)[0]
        # activated = int(self.gripper.activated)
        # contact = int(self.gripper.detect_contact(self.obj_id, self.obj_link) is not None)
        # TODO: replacing activated flag with contact flag
        activated = self.gripper.activated
        contact = self.gripper.detect_contact(self.obj_id, self.obj_link) is not None
        # contact_flag = int(activated or contact)
        # flag = int(activated)
        obs = np.concatenate(
            (
                np.array([int(activated or contact), ja]),
                self.gripper.gripper_pos,
                self.gripper.gripper_ori,
            )
        )
        return obs

    def step(self, action, relative=False):
        # action is (8,)

        # TODO: whether or not to pull is 1:1 with activated gripper
        # TODO: move/pull flag is probably obsolete now

        if relative:
            # convert relative position to absolute position
            d_pos = action[2:5]
            abs_pos = self.gripper.gripper_pos + d_pos
            # convert relative orientation to absolute orientation
            d_ori = R.from_euler("xyz", action[5:8]).as_matrix()
            curr_ori = R.from_quat(self.gripper.gripper_ori).as_matrix()
            abs_ori = R.from_matrix(d_ori @ curr_ori).as_euler("xyz")
            action[2:5] = abs_pos
            action[5:8] = abs_ori

        if action[0] == 0:
            if self.gripper.activated:
                self.gripper.release()
            else:
                # move
                self.gripper.set_move_cmds(action)
        elif action[0] == 1:
            if not self.gripper.activated:
                self.gripper.activate(self.obj_id, self.obj_link)
            else:
                # pull
                self.gripper.set_pull_cmds(action)
        p.stepSimulation(self.client_id)
        if self.gui:
            time.sleep(1 / 240.0)
        self.gripper.update_state()
        return self.get_obs()

        # if action[0] == 1 and not self.gripper.activated:
        #     self.gripper.activate(self.obj_id, self.obj_link)
        # if action[0] == 0 and self.gripper.activated:
        #     self.gripper.release()
        # elif action[1] == 0:
        #     self.gripper.set_move_cmds(action)
        # else:
        #     self.gripper.set_pull_cmds(action)
        # p.stepSimulation(self.client_id)
        # if self.gui:
        #     time.sleep(1 / 240.0)
        # self.gripper.update_state()
        # return self.get_state_obs()


class PMObjectEnv:
    def __init__(self, obj_id, pm_dataset_path, gui, use_egl=False):
        self.obj_id_str = obj_id
        self.obj = PMObject(pm_dataset_path / obj_id)
        self.renderer = PybulletRenderer()
        self.gui = gui
        self._core_env = PMRenderEnv(
            obj_id, pm_dataset_path, [-2, 0, 2], gui=gui, use_egl=use_egl
        )
        self.client_id = self._core_env.client_id
        self.obj_id = self._core_env.obj_id
        # goal attributes
        self.obj_link = 1
        self.goal = p.getJointInfo(self.obj_id, 1, self.client_id)[9]
        # rotation axis
        self.x = 0
        self.z = 0
        # door joints and links
        self.links = [js.name for js in self.obj.semantics.by_label("door")]
        link_to_joint = {
            p.getJointInfo(self.obj_id, i, self.client_id)[12].decode("utf-8"): i
            for i in range(p.getNumJoints(self.obj_id))
        }
        self.joints = [link_to_joint[link] for link in self.links]

    def render_obs(self, obs):
        pc = obs["pc"]
        seg = obs["seg"]
        fig = vpl.segmentation_fig(pc[::10], seg[::10])
        fig.show()

    def reset(self, goal_angle=0):
        obj_id = self.obj_id
        for ji in range(p.getNumJoints(obj_id)):
            if ji == self.obj_link:
                p.resetJointState(obj_id, ji, goal_angle, 0, self.client_id)
            else:
                p.resetJointState(obj_id, ji, 0, 0, self.client_id)
        p.stepSimulation(self.client_id)

    def get_obs(self, randomize_camera=None):
        # camera render
        camera_obs = self._core_env.render(camera_xyz=randomize_camera)
        rgb = camera_obs["rgb"]
        depth = camera_obs["depth"]
        pc_world = camera_obs["P_world"]
        pc_cam = camera_obs["P_cam"]
        pc_seg = camera_obs["pc_seg"]
        segmap = camera_obs["segmap"]
        # re-indexing point cloud (0: object, 1: object link, 2: gripper)
        pc_seg_scene = np.ones_like(pc_seg) * -1
        for k, (body, link) in segmap.items():
            if body == self.obj_id:
                ixs = pc_seg == k
                # link, or otherwise
                if link == self.obj_link:
                    pc_seg_scene[ixs] = 1
                else:
                    pc_seg_scene[ixs] = 0
        # object joint angle
        ja = p.getJointState(self.obj_id, 1, self.client_id)[0]
        obs = {
            "pc": pc_cam,  # this is returning camera coordinates
            "seg": pc_seg_scene,
            "joint_angle": ja,
        }
        return obs

    def set_axis_of_rotation(self):
        # get current link center of mass position in world frame
        link_index = self.obj_link
        com_pos = p.getLinkState(self.obj_id, link_index, self.client_id)[0]
        com_pos = np.array(com_pos)
        # axis defined by fixed x and z
        self.x = com_pos[0]
        self.z = com_pos[2]

    def rotate_link_pc(self, obs, goal_angle):
        # copy obs
        obs_rot = copy.deepcopy(obs)
        # get rotation matrix
        r = R.from_euler("y", -goal_angle, degrees=False)
        r = r.as_matrix()
        # get link pc
        pc = obs_rot["pc"]
        seg = obs_rot["seg"]
        link_pc = pc[seg == 1]
        # rotate link pc
        link_pc = link_pc - np.array([self.x, 0, self.z])
        link_pc = link_pc @ r.T
        link_pc = link_pc + np.array([self.x, 0, self.z])
        # update obs
        obs_rot["pc"][seg == 1] = link_pc
        obs_rot["joint_angle"] = goal_angle
        return obs_rot

    def random_demo(self, amount_to_actuate, randomize_camera=None):
        # TODO: this assumes there is only one door joint
        assert len(self.links) == 1, "Only one door joint is supported for random demo."
        # base render
        render = self.renderer.render(self.obj, camera_xyz=randomize_camera)
        p_world_init = render["pos"]
        t_wc = render["T_world_cam"]
        t_wb = render["T_world_base"]
        seg = render["seg"]
        angles = render["angles"]
        labelmap = render["labelmap"]
        rgb = render["rgb"]

        p_world_new = articulate_joint(
            obj=self.obj,
            current_jas=angles,
            link_to_actuate=self.links[0],
            amount_to_actuate=amount_to_actuate,
            pos=p_world_init,
            seg=seg,
            labelmap=labelmap,
            T_world_base=t_wb,
        )
        # transform both point clouds to camera frame
        p_cam_init = (
            np.linalg.inv(t_wc)
            @ np.concatenate([p_world_init, np.ones((len(p_world_init), 1))], axis=1).T
        ).T[:, :3]
        p_cam_new = (
            np.linalg.inv(t_wc)
            @ np.concatenate([p_world_new, np.ones((len(p_world_new), 1))], axis=1).T
        ).T[:, :3]
        return p_cam_init, p_cam_new, seg, rgb, t_wc
