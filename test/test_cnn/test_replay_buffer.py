from dqn.cnn.config import CNNConfigBase
from dqn.cnn.datum import Loss, SampleFromActor
from dqn.cnn.replay_buffer import (ReplayBuffer, ReplayBufferClient,
                                   ReplayBufferServer)
from dqn.replay_buffer import ReplayBufferConfig


def test_replay_buffer(config=CNNConfigBase(replay_buffer=ReplayBufferConfig(prioritized_replay=True, capacity=100))):
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


def test_replay_buffer_server(config=CNNConfigBase(replay_buffer_url='localhost:4444')):
    server = ReplayBufferServer(config=config)
    try:
        client = ReplayBufferClient(config=config)

        # can append sample
        client.append_sample(SampleFromActor.as_random(size=10, np_defs=config.sample_from_actor_def))
    finally:
        server.finalize()
