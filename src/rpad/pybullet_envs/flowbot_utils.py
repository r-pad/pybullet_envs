import numpy as np
from rpad.partnet_mobility_utils.articulate import articulate_joint


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
