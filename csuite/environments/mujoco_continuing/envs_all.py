# The continuing (non-episodic) versions of various Mujoco environments (from Gymnasium).

import gymnasium as gym


class BaseContinuingEnvBasedOnGym():
    """The class outlines the basic template that all the domains will follow."""
    
    def __init__(self, **env_args):
        assert self.env_key is not None, 'env_key must be defined in the subclass'
        self.reset_penalty = env_args.get('reset_penalty', -10)
        self.truncation_limit = env_args.get('truncation_limit', 100_000_000)
        self.gym_env = gym.make(self.env_key, max_episode_steps=self.truncation_limit, **env_args)
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
    """
    Changes to the Gymnasium's Swimmer-v5 to make this continuing version:
        - removed the truncation limit (increased it from 1000 to 100M)
    No other change required because the Swimmer never gets 'unhealthy'.
    """
    
    def __init__(self, **env_args):
        self.env_key = 'Swimmer-v5'
        super().__init__(**env_args)


class HalfCheetahContinuing(BaseContinuingEnvBasedOnGym):
    """
    Changes to the Gymnasium's HalfCheetah-v5 to make this continuing version:
        - removed the truncation limit (increased it from 1000 to 100M)
        - if the HalfCheetah flips over, it is again 'reset' with a penalthy (as above)
    The last case is not captured as 'unhealthy' by the original Gymnasium version.
    """

    def __init__(self, **env_args):
        self.env_key = 'HalfCheetah-v5'
        super().__init__(**env_args)
    
    def step(self, action):
        obs, reward, _, _, _ = self.gym_env.step(action)
        if self.gym_env.unwrapped.data.body('torso').xpos[2] < 0.15:
            obs, _ = self.gym_env.reset(seed=self.rng_seed)
            reward = self.reset_penalty
        return obs, reward


class AntContinuing(BaseContinuingEnvBasedOnGym):
    """
    Changes to the Gymnasium's Ant-v5 to make this continuing version:
        - removed the truncation limit (increased it from 1000 to 100M)
        - if the Ant is unhealthy, it is 'reset', and the agent gets a reset penalty and the new observation
        - if the Ant flips over, it is again 'reset' with a penalthy (as above). 
    The last case is not captured as 'unhealthy' by the original Gymnasium version.
    """

    def __init__(self, **env_args):
        self.env_key = 'Ant-v5'
        super().__init__(**env_args)

    def step(self, action):
        obs, reward, terminated_flag, _, _ = self.gym_env.step(action)
        if terminated_flag or (obs[0] < 0.3):                   # In gymnasium, the Ant is not 'unhealthy' even if 
            obs, _ = self.gym_env.reset(seed=self.rng_seed)     # it flips over: https://gymnasium.farama.org/environments/mujoco/ant/
            reward = self.reset_penalty
        return obs, reward


class HumanoidContinuing(BaseContinuingEnvBasedOnGym):
    """
    Changes to the Gymnasium's Ant-v5 to make this continuing version:
        - removed the truncation limit (increased it from 1000 to 100M)
        - if the Humanoid is unhealthy, it is 'reset', and the agent gets a reset penalty and the new observation
    """
    
    def __init__(self, **env_args):
        self.env_key = 'Humanoid-v5'
        super().__init__(**env_args)

    def step(self, action):
        obs, reward, terminated_flag, _, _ = self.gym_env.step(action)
        if terminated_flag:
            obs, _ = self.gym_env.reset(seed=self.rng_seed)
            reward = self.reset_penalty
        return obs, reward
