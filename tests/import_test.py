import os
from importlib.resources import as_file, files  # type: ignore


def test_import():
    fn = as_file(files("rpad_pybullet_envs_data").joinpath("assets/ur5/suction"))
    with fn as f:
        assert "base.obj" in set(os.listdir(f))
