from test.dammy_env import DammyEnv

from dqn.episodic_curiosity import EpisodicCuriosityConfig
from dqn.rnn.config import IntrinsicRewardConfig, RNNConfigBase
from dqn.rnn.policy import Policy


def test_rnn_policy():
    # turn on RND and episodic curiosity rewards
    config = RNNConfigBase(
        intrinsic_reward=IntrinsicRewardConfig(enable=True, episodic_curiosity=EpisodicCuriosityConfig(enable=True)))
    policy = Policy(config=config)

    obs = DammyEnv().reset()
    state = [policy.model.get_init_state()]
    for _ in range(config.intrinsic_reward.episodic_curiosity.k + 1):
        prediction, intrinsic_reward, state = policy.infer([obs], state)
