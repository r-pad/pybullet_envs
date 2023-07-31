import importlib.resources


def test_import():
    fs = importlib.resources.files("rpad_pybullet_envs_data")
    files = {f.name for f in fs.iterdir()}
    assert "base.obj" in files
