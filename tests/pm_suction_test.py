import os

import numpy as np
import numpy.typing as npt
import pytest
from rpad.partnet_mobility_utils.data import PMObject

from rpad.pybullet_envs.pm_suction import PMSuctionEnv, run_trial


class RandomFlowModel:
    def __init__(self, raw_data, env):
        self.env = env
        self.raw_data = raw_data

    def __call__(self, obs) -> npt.NDArray[np.float64]:
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = obs

        return np.random.random(P_cam.shape)


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

    joint = available_joints[np.random.randint(0, len(available_joints))]
    model = RandomFlowModel(raw_data, env)

    # t0 = time.perf_counter()
    print(f"opening {joint.name}, {joint.label}")
    return run_trial(env, raw_data, joint.name, model, n_steps=10)
