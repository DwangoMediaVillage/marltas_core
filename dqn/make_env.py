"""Function to initialize an environment."""
from dataclasses import dataclass

import gym

from dqn.env_wrapper import (ClipReward, EpisodicLife, FireReset, FrameSkip,
                             FrameStack, NoopReset, TimeLimit, WrapFrame)
from dqn.utils import ConfigBase


@dataclass
class EnvConfig(ConfigBase):
    """Configuration of environment.

    Attributes:
        name: Name of environment.
    """
    name: str = 'PongNoFrameskip-v4'
    wrap_atari: bool = True


def make_env(config: EnvConfig) -> gym.Env:
    """Init a OpenAI-gym environment.

    Args:
        config: Configuration of environment.

    Returns:
        env: Created env object.
    """
    env = gym.make(config.name)
    if config.wrap_atari:
        env = TimeLimit(env, max_steps=30 * 60 * 60)
        env = NoopReset(env)
        env = FrameSkip(env, 4)
        env = EpisodicLife(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = WrapFrame(env)
        env = ClipReward(env)
        env = FrameStack(env, 4)
    return env
