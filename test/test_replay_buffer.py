from typing import Union

import numpy as np

from dqn.replay_buffer import (PrioritizedReplayBuffer, RandomReplayBuffer,
                               ReplayBufferConfig)


def do_test_replay_buffer(buffer: Union[RandomReplayBuffer, PrioritizedReplayBuffer]):
    assert len(buffer) == 0, "buffer may be initialized with some samples"
    buffer.append_sample([np.random.random() for _ in range(5)], np.random.random(5))
    assert len(buffer) == 5

    sample_get, weight = buffer.get_sample(3)
    assert len(sample_get) == 3
    assert np.all(np.logical_not(np.isnan(weight)))

    if isinstance(buffer, PrioritizedReplayBuffer): buffer.update_loss(np.random.random(3))



def test_random_replay_buffer():
    do_test_replay_buffer(RandomReplayBuffer(ReplayBufferConfig(prioritized_replay=False, capacity=100)))


def test_prioritized_replay_buffer():
    do_test_replay_buffer(PrioritizedReplayBuffer(ReplayBufferConfig(prioritized_replay=False, capacity=100)))
