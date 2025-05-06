import numpy as np
import gymnasium as gym
from envs_all import *


def test_angle_bound():
    env = SwimmerContinuing()
    obs = env.start(0)
    for i in range(10000):
        action = np.random.random(2)
        obs, reward = env.step(action)


def test_render_simple():
    env = HalfCheetahContinuing(render_mode='rgb_array')
    obs = env.start(0)
    # print(obs)
    for i in range(10):
        action = np.random.random(6)
        obs, reward = env.step(action)
        # print(i, obs, reward, '\n')
        print(env.render().shape)
        # time.sleep(0.2)


if __name__ == "__main__":
    # test_render_simple()
    test_angle_bound()
