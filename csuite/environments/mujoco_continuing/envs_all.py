# The continuing (non-episodic) versions of various Mujoco environments (from Gymnasium).

import numpy as np
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

    def _bound_angles(self, angles):
        """Bound the angles in [-pi, pi]."""
        angles %= (2 * np.pi)
        for i, _ in enumerate(angles):
            if angles[i] > np.pi:
                angles[i] -= 2 * np.pi
        return angles

    def step(self, action):
        obs, reward, _, _, _ = self.gym_env.step(action)
        obs[:3] = self._bound_angles(obs[:3])
        return obs, reward


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
        - if the Ant flips over, it is again 'reset' with a penalthy (as above) 
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
    Changes to the Gymnasium's Humanoid-v5 to make this continuing version:
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


class ReacherContinuing(BaseContinuingEnvBasedOnGym):
    """
    Changes to the Gymnasium's Reacher-v5 to make this continuing version:
        - after a fixed number of steps (max_steps_per_goal), the goal is reset, but the robot 
        retains its position. The agent gets a new observation and reward based on the new goal.
    Note that there is no truncation in the original Reacher-v5.
    """

    def __init__(self, **env_args):
        self.env_key = 'Reacher-v5'
        super().__init__(**env_args)
        self.steps_per_goal = 0
        self.max_steps_per_goal = env_args.get('max_steps_per_goal', 50)

    def step(self, action):
        obs, reward, _, _, _ = self.gym_env.step(action)

        self.steps_per_goal += 1
        if self.steps_per_goal >= self.max_steps_per_goal:
            # reset the goal
            obs = self.reset_goal_only()
            # recompute the reward for the new goal (the action is used only to compute the cost of taking that action)
            reward, _ = self.gym_env.unwrapped._get_rew(action)
            self.steps_per_goal = 0
        return obs, reward

    def reset_goal_only(self):
        # Parts of the code from the original Gymnasium reset_model() function for Reacher-v5:
        # https://github.com/Farama-Foundation/Gymnasium/blob/d4dcc211705af0c862f6b28f09b32c07e2cf0cc4/gymnasium/envs/mujoco/reacher_v5.py#L217 
        
        while True:
            self.gym_env.unwrapped.goal = self.gym_env.unwrapped.np_random.uniform(low=-0.2, high=0.2, size=2)
            # ensure the goal is 'reachable'
            if np.linalg.norm(self.gym_env.unwrapped.goal) < 0.2:
                break
        self.gym_env.unwrapped.data.qpos[-2:] = self.gym_env.unwrapped.goal
        self.gym_env.unwrapped.set_state(self.gym_env.unwrapped.data.qpos, self.gym_env.unwrapped.data.qvel)
        
        return self.gym_env.unwrapped._get_obs()
    

class PusherContinuing(BaseContinuingEnvBasedOnGym):
    """
    Changes to the Gymnasium's Pusher-v5 to make this continuing version:
        - after a fixed number of steps (max_steps_per_task), the goal and the object are reset, 
        but the robot retains its position. The agent gets a new observation and reward based on the new goal.
    Note that there is no truncation in the original Pusher-v5.
    """

    def __init__(self, **env_args):
        self.env_key = 'Pusher-v5'
        super().__init__(**env_args)
        self.steps_per_task = 0
        self.max_steps_per_task = env_args.get('max_steps_per_task', 100)

    def step(self, action):
        obs, reward, _, _, _ = self.gym_env.step(action)

        self.steps_per_task += 1
        if self.steps_per_task >= self.max_steps_per_task:
            # reset the goal
            obs = self.reset_task()
            # recompute the reward for the new goal (the action is used only to compute the cost of taking that action)
            reward, _ = self.gym_env.unwrapped._get_rew(action)
            self.steps_per_task = 0
        return obs, reward

    def reset_task(self):
        # Parts of the code from the original Gymnasium reset_model() function for Pusher-v5:
        # https://github.com/Farama-Foundation/Gymnasium/blob/d4dcc211705af0c862f6b28f09b32c07e2cf0cc4/gymnasium/envs/mujoco/pusher_v5.py#L246
        
        while True:
            # sample a new goal position on the left or the right of the table (symmetrically about the robot's torso)
            goal_y = 0 if self.gym_env.unwrapped.np_random.uniform() < 0.5 else -0.9
            self.gym_env.unwrapped.goal_pos = np.asarray([0, goal_y])
            
            # sample a new cylinder position on the either side of the goal and slightly below to enable 'pushing' 
            self.gym_env.unwrapped.cylinder_pos = np.concatenate(
                [
                    self.gym_env.unwrapped.np_random.uniform(low=-0.3, high=0, size=1),
                    self.gym_env.unwrapped.np_random.uniform(low=goal_y-0.2, high=goal_y+0.2, size=1),
                ]
            )
            # ensure the cylinder and the goal are not too close
            if np.linalg.norm(self.gym_env.unwrapped.cylinder_pos - self.gym_env.unwrapped.goal_pos) > 0.17:
                break

        self.gym_env.unwrapped.data.qpos[-4:-2] = self.gym_env.unwrapped.cylinder_pos
        self.gym_env.unwrapped.data.qpos[-2:] = self.gym_env.unwrapped.goal_pos
        self.gym_env.unwrapped.set_state(self.gym_env.unwrapped.data.qpos, self.gym_env.unwrapped.data.qvel)
        
        return self.gym_env.unwrapped._get_obs()
