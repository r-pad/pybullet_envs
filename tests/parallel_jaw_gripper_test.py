from pathlib import Path

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


CUBE_FILE = str(Path(__file__).parent.parent / "models/cube.urdf")


def test_parallel_jaw_gripper_grasping():
    
    client_id = p.connect(p.GUI)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=30,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0.5],
    )

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    cube_id = p.loadURDF(CUBE_FILE, physicsClientId=client_id)

    # placing cube above origin
    p.resetBasePositionAndOrientation(cube_id, [0, 0, 1], [0, 0, 0, 1], client_id)

    # Turn on gravity.
    p.setGravity(0, 0, -9.81, client_id)

    # Step the sim and wait for it to settle
    for _ in range(1000):
        p.stepSimulation(client_id)

    # breakpoint()

    # Turn off gravity.
    p.setGravity(0, 0, 0, client_id)

    # Get the settled position of the cube.
    start_pos, _ = p.getBasePositionAndOrientation(cube_id, client_id)

    # Create a parallel-jaw gripper.
    gripper = FloatingParallelJawGripper(client_id)

    # breakpoint()

    # Put the parallel-jaw gripper above the cube.
    gripper.reset_pose([0, 0, 0.25], p.getQuaternionFromEuler([np.pi, 0, 0]))

    pose = gripper.get_gripper_pose()

    breakpoint()
    gripper.open()
    # for i in range(1000):
    #     gripper.set_gripper_command(1.0)

    #     p.stepSimulation(client_id)

    #     if i % 100 == 0:
    #         breakpoint()

    breakpoint()

    gripper.get_gripper_pose()

    gripper.move([0, 0, -1.0], p.getQuaternionFromEuler([np.pi, 0, 0]))

    breakpoint()
    
    gripper.move([0, 0, 1.0], p.getQuaternionFromEuler([np.pi, 0, 0]))

    # Get the current position of the gripper.

    # Set the velocity of the gripper to be downward.
    # gripper.set_velocity([0, 0, -0.1], [0, 0, 0])
    
    gripper.close()

    # Step the simulation.
    for i in range(20000):
        p.stepSimulation(client_id)
        gripper.open()
        gripper.move([0, 0, -1.0], p.getQuaternionFromEuler([np.pi, 0, 0]))
        gripper.close()
        
        if gripper.detect_contact(cube_id):
            print("Contact detected")
            
    gripper.move([0, 0, 1.0], p.getQuaternionFromEuler([np.pi, 0, 0]))

    if i >= 20000:
        raise ValueError("No contact detected!")

    # # Grasp the cube with the parallel-jaw gripper.
    # gripper.grasp(cube_id)

    # Set the velocity of the gripper to be upward.
    gripper.set_velocity([0, 0, 0.1], [0, 0, 0])

    # Step the simulation.
    for _ in range(20000):
        p.stepSimulation(client_id)

    # Make sure that the cube is above the ground.
    # apex_pos, _ = p.getBasePositionAndOrientation(cube_id, client_id)
    # if apex_pos[2] - start_pos[2] < 0.1:
    #     raise ValueError("Cube is not off the ground!")

    # Enable gravity.
    p.setGravity(0, 0, -9.81, client_id)

    # Release the grip
    gripper.release()

    # Step the simulation.
    for _ in range(5000):
        gripper.apply_force([0, 0, 9.81 * gripper.mass + 3])
        p.stepSimulation(client_id)


# def partnet_parallel_jaw_test():
#     client_id = p.connect(p.GUI)

#     gripper = PMFloatingEnv(client_id)

#     #p.disconnect(client_id)


if __name__ == "__main__":
    test_parallel_jaw_gripper_grasping()
    while True:
        pass
