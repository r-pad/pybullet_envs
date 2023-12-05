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

class FloatingParallelJawGripper:
    def __init__(self, client_id):
        self.client_id = client_id

        # This is just a floating gripper yay.
        base_pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.base_id = p.loadURDF(
            PARALLEL_JAW_GRIPPER_URDF, *base_pose, physicsClientId=self.client_id
        )