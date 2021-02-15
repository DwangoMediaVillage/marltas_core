from test.dammy_env import DammyEnv

from dqn.cnn.actor import Policy
from dqn.cnn.config import CNNConfigBase


def test_policy():
    policy = Policy(config=CNNConfigBase())
    env = DammyEnv()
    obs = env.reset()
    prediction, intrinsic_reward = policy.infer([obs])
