import pybullet as p

from rpad.pybullet_envs.parallel_jaw_v2 import FloatingParallelJawGripper

def test_parallel_jaw_gripper_creation():
    client_id = p.connect(p.DIRECT)

    gripper = FloatingParallelJawGripper(client_id)

    p.disconnect(client_id)


def test_parallel_jaw_gripper():
    pass
