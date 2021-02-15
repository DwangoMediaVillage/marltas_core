import gym

from dqn.make_env import EnvConfig, make_env


def test_make_atari():
    env_config = EnvConfig(name='PongNoFrameskip-v4')
    env = make_env(env_config)
    assert isinstance(env, gym.Env)


def test_make_non_atari():
    env_config = EnvConfig(name='Acrobot-v1', wrap_atari=False)
    env = make_env(env_config)
    assert isinstance(env, gym.Env)
