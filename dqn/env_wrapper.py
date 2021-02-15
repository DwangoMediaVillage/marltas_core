"""Wrappers of OpenAI gym environment."""
from collections import deque

import cv2
import gym
import numpy as np


class TimeLimit(gym.Wrapper):
    """Give time limit of episode length."""
    def __init__(self, env: gym.Env, max_steps: int = 30 * 60 * 60):
        super().__init__(env)
        self.max_steps = max_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None,\
            "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self.max_steps <= self._elapsed_steps:
            info['needs_reset'] = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


class NoopReset(gym.Wrapper):
    """Take noop actions at reset."""
    def __init__(self, env: gym.Env, noop_max: int = 30, noop_action: int = 0):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = noop_action

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs = None
        for _ in range(np.random.randint(1, self.noop_max + 1)):
            obs, _, done, info = self.env.step(self.noop_action)
            if done or info.get('needs_reset', False):
                obs = self.env.reset(**kwargs)
        return obs


class FrameSkip(gym.Wrapper):
    """Skipping give number of steps."""
    def __init__(self, env: gym.Env, steps: int):
        gym.Wrapper.__init__(self, env)
        self.steps = steps
        self.obs_buffer = np.zeros((2, ) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        for t in range(self.steps):
            obs, reward, done, info = self.env.step(action)
            if t == self.steps - 2:
                self.obs_buffer[0] = obs
            if t == self.steps - 1:
                self.obs_buffer[1] = obs
            total_reward += reward
            if done or info.get('needs_reset', False): break
        max_frame = self.obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EpisodicLife(gym.Wrapper):
    """Terminates at atari's life end."""
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.needs_reset = True

    def reset(self, **kwargs):
        if self.needs_reset:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.needs_reset = done or info.get('needs_reset', False)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info


class FireReset(gym.Wrapper):
    """Push FIRE button after reset."""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, info = self.env.step(1)
        if done or info.get('needs_reset', False):
            self.env.reset(**kwargs)
        obs, _, done, info = self.env.step(2)
        if done or info.get('needs_reset', False):
            self.env.reset(**kwargs)
        return obs


class WrapFrame(gym.ObservationWrapper):
    """Glayscale and resize observation image."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

    def observation(self, observation):
        o = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        o = cv2.resize(o, (84, 84))
        return np.expand_dims(o, 0)  # [1, w, h]


class ClipReward(gym.RewardWrapper):
    """Clips reward"""
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class LazyFrames:
    """Helper class to reduce memory consumptions."""
    def __init__(self, frames):
        self.frames = frames

    def __array__(self):
        return np.concatenate(self.frames, axis=0)


class FrameStack(gym.Wrapper):
    """Stack observation images in channel dimension."""
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.frames = deque([], maxlen=self.n)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.n
        return LazyFrames(list(self.frames))
