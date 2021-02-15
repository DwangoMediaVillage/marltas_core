from dqn.replay_buffer import ReplayBufferConfig
from dqn.rnn.config import RNNConfigBase
from dqn.rnn.datum import Loss, SampleFromActor
from dqn.rnn.replay_buffer import ReplayBuffer


def test_replay_buffer(config=RNNConfigBase(replay_buffer=ReplayBufferConfig(prioritized_replay=True, capacity=100))):
    replay_buffer = ReplayBuffer(config=config)
    stat = replay_buffer.get_status()
    assert stat.size == 0
    assert stat.beta is not None
    assert stat.append_sample_per_sec is None
    assert stat.get_sample_per_sec is None
    assert stat.update_loss_per_sec is None
    assert stat.priority_mean is None

    replay_buffer.append_sample(SampleFromActor.as_random(size=10, np_defs=config.sample_from_actor_def))
    replay_buffer.get_sample(10)
    replay_buffer.append_sample(SampleFromActor.as_random(size=10, np_defs=config.sample_from_actor_def))
    replay_buffer.get_sample(10)
    stat = replay_buffer.get_status()

    assert stat.size == 20
    assert stat.beta is not None
    assert stat.append_sample_per_sec is not None
    assert stat.get_sample_per_sec is not None
    assert stat.priority_mean is not None

    replay_buffer.get_sample(10)
    replay_buffer.update_loss(Loss.as_random(size=10, np_defs=config.loss_def))
