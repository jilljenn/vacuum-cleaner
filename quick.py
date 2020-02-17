"""
A minimal agent controlling a vacuum cleaner
"""
# pylint: disable=import-error,redefined-outer-name
import numpy as np
from vacuum import RoombaEnv
from gym.wrappers import Monitor


def wrap_env(env):
    """
    Utility functions to record environment and display it.
    To enable video, just do:
        env = wrap_env(env)
    """
    env = Monitor(env, './video', force=True)
    return env


env = wrap_env(RoombaEnv(20, 20, 500, 2))  # width, height, battery, radius
observation = env.reset()

all_obs = []

print(env.action_space)

while True:
    env.render()

    # Your agent goes here
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)
    all_obs.append(observation)

    if done:
        break
env.close()

all_ = np.array(all_obs)
print(np.min(all_, axis=0))  # Some stats regarding states
print(np.max(all_, axis=0))
