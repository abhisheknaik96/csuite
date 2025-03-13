# The continuing (non-episodic) version of Mujoco's HalfCheetah.

import time
import numpy as np
import gymnasium as gym


LARGE_TRUNCATION_LIMIT = 10_000_000
RESET_PENALTY = -10


class BaseContinuingEnvBasedOnGym():
    
    def __init__(self, **env_args):
        assert self.env_key is not None, 'env_key must be defined in the subclass'
        self.gym_env = gym.make(self.env_key, max_episode_steps=LARGE_TRUNCATION_LIMIT, **env_args)
        self.rng_seed = None

    def start(self, seed):
        self.rng_seed = seed
        first_obs, _ = self.gym_env.reset(seed=self.rng_seed)
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
            # print('maybe flipped')
            obs, _ = self.gym_env.reset(seed=self.rng_seed)
            reward = RESET_PENALTY
        return obs, reward


class AntContinuing(BaseContinuingEnvBasedOnGym):

    def __init__(self, **env_args):
        self.env_key = 'Ant-v5'
        super().__init__(**env_args)

    def step(self, action):
        obs, reward, terminated_flag, truncated_flag, info = self.gym_env.step(action)
        print(obs[0])
        if terminated_flag:     # this happens when the ant is 'unhealthy': https://gymnasium.farama.org/environments/mujoco/ant/
            print('unhealthy, resetting')
            obs, _ = self.gym_env.reset(seed=self.rng_seed)
            reward = RESET_PENALTY
        return obs, reward


class HumanoidContinuing(BaseContinuingEnvBasedOnGym):

    def __init__(self, **env_args):
        self.env_key = 'Humanoid-v5'
        super().__init__(**env_args)


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
