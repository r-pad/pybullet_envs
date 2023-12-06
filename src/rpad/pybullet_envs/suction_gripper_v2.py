import os

try:
    from importlib.resources import as_file, files  # type: ignore
except ImportError:
    from importlib_resources import as_file, files  # type: ignore


fn = as_file(files("rpad.pybullet_envs").joinpath("assets/suction"))

with fn as f:
    SUCTION_URDF = os.path.join(f, "suction_with_mount_no_collision.urdf")
