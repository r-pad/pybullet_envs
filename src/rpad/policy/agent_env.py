import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np

from rpad.pybullet_envs.parallel_jaw_v2 import FloatingParallelJawGripper


class SimplePyBulletEnv(gym.Env):
    def __init__(self):
        self.client_id = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-40,
            cameraTargetPosition=[0.55, -0.35, 0.2],
        )
        self.action_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        self.observation_space = spaces.Box(np.array([-1] * 3), np.array([1] * 3))

        self.max_steps = 1000  # Adjust as needed
        self.current_step = 0
        self.distance_threshold = 0.005  # Adjust as needed

    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 0
        )  # we will enable rendering after we loaded everything
        p.setGravity(0, 0, -10)

        p.setAdditionalSearchPath("../../../models")

        # Create gripper and object
        self.gripper = FloatingParallelJawGripper(self.client_id)
        self.gripper.reset_pose(
            [-1.0, 0, 0.25], p.getQuaternionFromEuler([np.pi, 0, 0])
        )

        self.object_id = p.loadURDF(
            "102645/mobility.urdf", useFixedBase=True, globalScaling=0.40
        )
        p.resetBasePositionAndOrientation(
            self.object_id, [0, 0, 0.25], [0, 0, 0, 1], self.client_id
        )

        observation = self.get_observation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return observation  # Placeholder observation

    def step(self, action):
        max_vel_scaling = 0.1
        d_pos, gripper = action[:3], action[3]

        # O
        d_pos = d_pos * max_vel_scaling
        # Given the action and current position of the gripper, compute the desired next position of the gripper.
        goal_pos = self.gripper.get_gripper_pose()[0] + np.array(d_pos)
        goal_ori = self.gripper.get_gripper_pose()[1]
        goal_state = None

        timeout = 100

        # OPTION 1: WRITE THE CONTROL LOOP HERE. THIS WOULD ALLOW YOU TO CONTROL BOTH THE FINGERS AND GRIPPER AT THE SAME
        # TIME.

        # def at_goal(c, g):
        #     pass

        # Set some command on the gripper. This is likely a relative position command, so we'll want to achieve it if possible.

        # Step the environment for some number of steps with the same command until you achieve the desired position or until you hit
        # some timeout.
        # while timeout > 0:

        #     # Get the current state.
        #     curr_state = None

        #     # Check if we're at the goal state.
        #     if at_goal(curr_state, goal_state):
        #         break

        #     p.stepSimulation()

        #     timeout -= 1

        # OPTION 2 (Preferred): USE THE GRIPPER MOVE COMMAND.
        self.gripper.move(goal_pos, goal_ori, timeout)

        # Then write a loop to control the fingers afterward?
        if gripper > 0:
            self.gripper.open(timeout)
        else:
            self.gripper.close(timeout)

        # Compute the next observation.
        observation = self.get_observation()

        # Compute the reward.
        reward = self.calc_reward()
        done = False

        done = (
            np.linalg.norm(goal_pos - self.gripper.get_gripper_pose()[0])
            < self.distance_threshold
        )
        self.current_step += 1

        # # Check if the maximum number of steps is reached
        if self.current_step >= self.max_steps:
            done = True

        return observation, reward, done, {}

    def render(self, mode="human"):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.10, 0, 0.05],
            distance=1.2,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()

    def get_observation(self):
        gripper_pos, _ = self.gripper.get_gripper_pose()
        target_point = np.array([-1.0, 0.0, 0.25])
        delta_pos = gripper_pos - target_point
        return delta_pos
        # gripper_pos, gripper_orient = p.getBasePositionAndOrientation(self.gripperid)
        # target_obj_angle = p.getJointState(self.objectUid, 0)[0]

        # self.target_point = [1.1, -0.10, 0.30]

        # # delta pos and orient
        # delta_pos = np.array(gripper_pos) - np.array(self.target_point)
        # delta_orient = np.array(gripper_orient)

        # # observe
        # final_observation = np.concatenate(
        #     [delta_pos, delta_orient, [target_obj_angle]]
        # )

        # return final_observation

    def calc_reward(self):
        delta_pos = self.get_observation()
        distance = np.linalg.norm(delta_pos)
        return -distance


# if __name__ == "__main__":
#     env = SimplePyBulletEnv()
#     obs = env.reset()
#     for _ in range(1000):
#         env.render()
#         action = env.action_space.sample()
#         env.step(action)

#     env.close()
