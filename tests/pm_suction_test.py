import os
from typing import Dict, Literal, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
import pybullet as p
import pytest
import rpad.partnet_mobility_utils.articulate as pma
from rpad.partnet_mobility_utils.data import PMObject

from rpad.pybullet_envs.pm_suction import PMSuctionEnv, reindex, run_trial


# COPIED FROM FLOWBOT, DON'T WANT TO ADD FLOWBOT AS A DEPENDENCY...
def compute_normalized_flow(
    P_world: npt.NDArray[np.float32],
    T_world_base: npt.NDArray[np.float32],
    current_jas: Dict[str, float],
    pc_seg: npt.NDArray[np.uint8],
    labelmap: Dict[str, int],
    pm_raw_data: PMObject,
    linknames: Union[Literal["all"], Sequence[str]] = "all",
) -> npt.NDArray[np.float32]:
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
        P_world_new = pma.articulate_joint(
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


class GTFlowModel:
    def __init__(self, raw_data, env):
        self.env = env
        self.raw_data = raw_data

    def __call__(self, obs) -> npt.NDArray[np.float32]:
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = (
            obs["rgb"],
            obs["depth"],
            obs["seg"],
            obs["P_cam"],
            obs["P_world"],
            obs["pc_seg"],
            obs["segmap"],
        )
        pc_seg = reindex(pc_seg, segmap, self.env._core_env.obj_id)

        env = self.env
        raw_data = self.raw_data

        links = raw_data.semantics.by_type("slider")
        links += raw_data.semantics.by_type("hinge")
        current_jas: Dict[str, float] = {}
        for link in links:
            linkname = link.name
            chain = raw_data.obj.get_chain(linkname)
            for joint in chain:
                current_jas[joint.name] = 0
        normalized_flow = compute_normalized_flow(
            P_world,
            env._core_env.T_world_base,
            current_jas,
            pc_seg,
            env._core_env.link_name_to_index,
            raw_data,
            "all",
        )

        return normalized_flow


class RandomFlowModel:
    def __init__(self, raw_data, env):
        self.env = env
        self.raw_data = raw_data

    def __call__(self, obs) -> npt.NDArray[np.float64]:
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = (
            obs["rgb"],
            obs["depth"],
            obs["seg"],
            obs["P_cam"],
            obs["P_world"],
            obs["pc_seg"],
            obs["segmap"],
        )

        return cast(npt.NDArray[np.float64], np.random.random(P_cam.shape))


@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/datasets/partnet-mobility/raw")),
    reason="requires partnet-mobility dataset",
)
def test_pm_suction():
    obj_id = "41083"
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    env = PMSuctionEnv(obj_id, pm_dir, gui=False)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
        "slider"
    )

    seed = 123
    np.random.seed(seed)
    joint = available_joints[np.random.randint(0, len(available_joints))]
    # model = RandomFlowModel(raw_data, env)
    model = GTFlowModel(raw_data, env)

    # t0 = time.perf_counter()
    print(f"opening {joint.name}, {joint.label}")
    res = run_trial(env, raw_data, joint.name, model, n_steps=10)

    assert res.success

    p.disconnect(env._core_env.client_id)
