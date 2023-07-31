import numpy as np
import pybullet as p
import pybullet_data

from rpad.pybullet_envs.suction_gripper import FloatingSuctionGripper


def test_suction_gripper_creation():
    client_id = p.connect(p.DIRECT)

    gripper = FloatingSuctionGripper(client_id)

    p.disconnect(client_id)


def test_simple_grasp_action():
    # Create a client
    client_id = p.connect(p.DIRECT)

    # Put a plane in the world.
    # Put an R2D2 in the world.
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    r2d2_id = p.loadURDF("r2d2.urdf", physicsClientId=client_id)

    # Place the R2D2 1m above the origin.
    p.resetBasePositionAndOrientation(r2d2_id, [0, 0, 1], [0, 0, 0, 1], client_id)

    # Turn on gravity.
    p.setGravity(0, 0, -9.81, client_id)

    # Step the sim and wait for it to settle.
    for i in range(1000):
        p.stepSimulation(client_id)

    # Turn off gravity.
    p.setGravity(0, 0, 0, client_id)

    # Get the settled position of the R2D2.
    start_pos, _ = p.getBasePositionAndOrientation(r2d2_id, client_id)

    # Create a suction gripper.
    gripper = FloatingSuctionGripper(client_id)

    # Put the suction gripper above the R2D2, with the orientation flipped along the x axis
    gripper.set_pose([0, 0, 1.5], p.getQuaternionFromEuler([np.pi, 0, 0]))

    # Set the velocity of the gripper to be downward.
    gripper.set_velocity([0, 0, -0.1], [0, 0, 0])

    # Step the simulation.
    for i in range(5000):
        p.stepSimulation(client_id)

        # Detect contact.
        if gripper.detect_contact(r2d2_id):
            print("Contact detected!")
            break

    # Check if we detected contact.
    if i >= 5000:
        raise ValueError("No contact detected!")

    # Activate the suction.
    gripper.activate(r2d2_id)

    # Set the velocity of the gripper to be upward.
    gripper.set_velocity([0, 0, 0.1], [0, 0, 0])

    # Step the simulation.
    for i in range(5000):
        p.stepSimulation(client_id)

    # Make sure that the r2d2 is above the ground.
    apex_pos, _ = p.getBasePositionAndOrientation(r2d2_id, client_id)
    if apex_pos[2] - start_pos[2] < 0.1:
        raise ValueError("R2D2 is not off the ground!")

    # Enable gravity.
    p.setGravity(0, 0, -9.81, client_id)

    # Release the suction.
    gripper.release()

    # Step the simulation.
    for i in range(5000):
        # apply force to the gripper to counteract gravity.
        gripper.apply_force([0, 0, 9.81 * gripper.mass + 3])
        p.stepSimulation(client_id)

    # Make sure that the r2d2 is on the ground.
    next_pos, _ = p.getBasePositionAndOrientation(r2d2_id, client_id)
    assert next_pos[2] < apex_pos[2]

    p.disconnect(client_id)
