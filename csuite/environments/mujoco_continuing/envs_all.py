# The continuing (non-episodic) version of Mujoco's HalfCheetah.

import time
import numpy as np
import gymnasium as gym


LARGE_TRUNCATION_LIMIT = 10_000_000
RESET_PENALTY = 10


class BaseContinuingEnvBasedOnGym():
    
    def __init__(self, **env_args):
        assert self.env_key is not None, 'env_key must be defined in the subclass'
        env_args['render_mode'] = 'rgb_array'
        # env_args['render_mode'] = 'human' if env_args['render'] else 'rgb_array'
        self.gym_env = gym.make(self.env_key, max_episode_steps=LARGE_TRUNCATION_LIMIT, **env_args)

    def start(self, seed):
        first_obs, _ = self.gym_env.reset(seed=seed)
        return first_obs
    
    def step(self, action):
        obs, reward, _, _, _ = self.gym_env.step(action)
        return obs, reward
    
    def render(self):
        return self.gym_env.render()


class SwimmerContinuing(BaseContinuingEnvBasedOnGym):
    
    def __init__(self, **env_args):
        self.env_key = 'Swimmer-v5'
        super().__init__(**env_args)


class HalfCheetahContinuing(BaseContinuingEnvBasedOnGym):

    def __init__(self, **env_args):
        self.env_key = 'HalfCheetah-v5'
        super().__init__(**env_args)
    
    def step(self, action):
        obs, reward, terminated_flag, truncated_flag, info = self.gym_env.step(action)
        if self.gym_env.unwrapped.data.body('torso').xpos[2] < 0.15:
            # self.render()
            print('maybe flipped')
            # time.sleep(0.2)
        # print(info)
        return obs, reward


if __name__ == "__main__":
    # env = HalfCheetahContinuing(render_mode='human')
    env = HalfCheetahContinuing()
    obs = env.start(0)
    # print(obs)
    for i in range(10):
        action = np.random.random(6)
        obs, reward = env.step(action)
        # print(i, obs, reward, '\n')
        print(env.render().shape)
        # time.sleep(0.2)
