import os
from pathlib import Path

import pytest

try:
    from importlib.resources import as_file, files  # type: ignore
except ImportError:
    from importlib_resources import as_file, files  # type: ignore

import numpy as np
from scipy.spatial.transform import Rotation as R

from rpad.pybullet_envs.pm_suction import PMSuctionDemoEnv

HAS_PM = "PARTNET_MOBILITY_DIR" in os.environ and os.path.exists(
    os.environ["PARTNET_MOBILITY_DIR"]
)


@pytest.mark.skipif(not HAS_PM, reason="PARTNET_MOBILITY_DIR not set")
def test_pm_suction():
    OBJ_ID = "7273"
    PM_DIR = Path(os.path.expanduser(os.environ["PARTNET_MOBILITY_DIR"]))
    # fn = as_file(
    #     files("rpad_pybullet_envs_data").joinpath("assets/ur5/suction_with_mount")
    # )
    # with fn as f:
    #     GRIPPER_PATH = os.path.join(f, "suction_with_mount.urdf")

    use_gui = True if "TESTING_GUI" in os.environ else False

    env = PMSuctionDemoEnv(OBJ_ID, PM_DIR, use_gui)
    # initializing environment
    pos_init = np.array([-1.0, 0.6, 0.8])
    ori_init = R.from_euler("xyz", [0, -np.pi / 2, 0]).as_quat()
    env.reset(pos_init, ori_init)
    # generating demonstration
    success, demo = env.generate_demo()
    assert success
    # demo_final_state = demo[-1]["obs"]["ja"]

    print("Sucessful demo obtained.")
    # replication demonstration (TODO: need a proper step function here)
    # for each action in demo...

    # copy all actions

    # check that final joint state is the same
    # use self.goal
