import numpy as np
import gymnasium as gym
from envs_all import *


def simple_test():
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
    simple_test()