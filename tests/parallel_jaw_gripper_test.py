import os

import numpy as np
import pybullet as p
import pybullet_data

# from rpad.pybullet_envs.pm_parallel_jaw import PMFloatingEnv
from rpad.pybullet_envs.parallel_jaw_v2 import FloatingParallelJawGripper


def test_parallel_jaw_gripper_creation():
    client_id = p.connect(p.DIRECT)

    gripper = FloatingParallelJawGripper(client_id)

    p.disconnect(client_id)


def test_parallel_jaw_gripper():
    client_id = p.connect(p.DIRECT)

    gripper = FloatingParallelJawGripper(client_id)

    num_joints = p.getNumJoints(gripper.base_id)
    joint_indices = [i for i in range(num_joints)]

    initial_joint_angles = [0.0] * num_joints
    p.setJointMotorControlArray(
        bodyUniqueId=gripper.base_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=initial_joint_angles,
    )

    target_joint_angles = [0.25] * num_joints

    p.setJointMotorControlArray(
        bodyUniqueId=gripper.base_id,
        jointIndices=joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=target_joint_angles,
        forces=[10.0] * num_joints,  # Adjust force as needed
    )

    # Run the simulation for a few steps
    for _ in range(100):
        p.stepSimulation()


def test_parallel_jaw_gripper_grasping():
    is_gui = "TESTING_MODE" in os.environ and os.environ["TESTING_MODE"] == "GUI"
    mode = p.GUI if is_gui else p.DIRECT

    client_id = p.connect(mode)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=30,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0.5],
    )

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    cube_id = p.loadURDF("cube.urdf", globalScaling=0.05, physicsClientId=client_id)

    # placing cube above origin
    p.resetBasePositionAndOrientation(cube_id, [0, 0, 1], [0, 0, 0, 1], client_id)

    # Turn on gravity.
    p.setGravity(0, 0, -9.81, client_id)

    # Step the sim and wait for it to settle
    for _ in range(1000):
        p.stepSimulation(client_id)

    # Get the settled position of the cube.
    start_pos, _ = p.getBasePositionAndOrientation(cube_id, client_id)

    # Create a parallel-jaw gripper.
    gripper = FloatingParallelJawGripper(client_id)

    # Put the parallel-jaw gripper above the cube.
    gripper.reset_pose([0, 0, 0.25], p.getQuaternionFromEuler([np.pi, 0, 0]))

    # Grasp the block.
    gripper.open()
    gripper.move([0, 0, 0.15], p.getQuaternionFromEuler([np.pi, 0, 0]))
    gripper.close()

    # Move the block to the right.
    gripper.move([0, 0, 0.5], p.getQuaternionFromEuler([np.pi, 0, 0]))
    gripper.move([0, 1.0, 0.5], p.getQuaternionFromEuler([np.pi, 0, 0]))
    gripper.move([0, 1.0, 0.15], p.getQuaternionFromEuler([np.pi, 0, 0]))

    # Place the block and move away.
    gripper.open()
    gripper.move([0, 1.0, 0.4], p.getQuaternionFromEuler([np.pi, 0, 0]))

    # Make sure the block is close to where we want it to be.
    end_pos, _ = p.getBasePositionAndOrientation(cube_id, client_id)
    assert (end_pos[1] - 1.0) < 0.05


if __name__ == "__main__":
    test_parallel_jaw_gripper_grasping()
    while True:
        pass
