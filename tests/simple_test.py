import importlib.resources


def test_simple():
    assert 1 == 1


def test_import():
    fs = importlib.resources.files("rpad_pybullet_envs_data")
    breakpoint()
