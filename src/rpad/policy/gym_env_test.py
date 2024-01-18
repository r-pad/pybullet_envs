import gym
from agent_env import SimplePyBulletEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# # Import your custom environment
env = SimplePyBulletEnv()

# env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# # # Reset the environment to the initial state
observation = env.reset()

# # # Run a few steps in the environment
for _ in range(100):
    action = (
        env.action_space.sample()
    )  # Replace this with your actual action selection logic
    observation, reward, done, _ = env.step(action)
    env.render()

env.close()
