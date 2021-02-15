import numpy as np

from dqn.cnn.config import CNNConfigBase
from dqn.cnn.datum import (Batch, Loss, SampleFromActor, SampleFromBuffer,
                           split_sample_from_actor)


def test_sample_from_actor(config=CNNConfigBase()):
    size = 4
    s = SampleFromActor.as_random(size=size, np_defs=config.sample_from_actor_def)
    assert s.size == size

    s_restore = SampleFromActor.from_bytes(bytes_data=s.to_bytes(np_defs=config.sample_from_actor_def),
                                           np_defs=config.sample_from_actor_def)
    assert s.eq(s_restore, np_defs=config.sample_from_actor_def)

    s_split = split_sample_from_actor(sample=s, sample_from_actor_def=config.sample_from_actor_def)
    for i in range(size):
        for name, np_def in config.sample_from_actor_def.items():
            if np_def is not None:
                assert np.array_equal(getattr(s_split[i], name), getattr(s, name)[i])


def test_sample_from_buffer(config=CNNConfigBase()):
    size = 4
    s = SampleFromBuffer.as_random(size=size, np_defs=config.sample_from_buffer_def)
    assert s.eq(other=SampleFromBuffer.from_bytes(bytes_data=s.to_bytes(np_defs=config.sample_from_buffer_def),
                                                  np_defs=config.sample_from_buffer_def),
                np_defs=config.sample_from_buffer_def)

    sample_from_actor = SampleFromActor.as_random(size=size, np_defs=config.sample_from_actor_def)
    s_buffer = SampleFromBuffer.from_buffer_samples(samples=split_sample_from_actor(
        sample=sample_from_actor, sample_from_actor_def=config.sample_from_actor_def),
                                                    sample_from_actor_def=config.sample_from_actor_def)

    # check shape and type
    for k, v in config.sample_from_buffer_def.items():
        if v is not None:
            x = getattr(s_buffer, k)
            if k != 'weight':
                assert tuple(x.shape) == tuple([size] + list(v.shape))
                assert x.dtype == v.dtype
                assert np.array_equal(x, getattr(sample_from_actor, k))


def test_batch(config=CNNConfigBase()):
    sample_from_buffer = SampleFromBuffer.as_random(size=10, np_defs=config.sample_from_buffer_def)
    batch = Batch.from_buffer_sample(sample_from_buffer)


def test_loss(config=CNNConfigBase()):
    loss = Loss.as_random(size=4, np_defs=config.loss_def)
    assert loss.eq(other=Loss.from_bytes(bytes_data=loss.to_bytes(np_defs=config.loss_def), np_defs=config.loss_def),
                   np_defs=config.loss_def)
