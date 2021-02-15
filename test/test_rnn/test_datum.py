import numpy as np

from dqn.rnn.config import RNNConfigBase
from dqn.rnn.datum import (Batch, SampleFromActor, SampleFromBuffer,
                           split_sample_from_actor)


def test_concat_sample_from_actor(config=RNNConfigBase()):
    sample = SampleFromActor.as_random(np_defs=config.sample_from_actor_def, size=2)
    result = SampleFromActor.concat(samples=[sample], np_defs=config.sample_from_actor_def)
    result.validate_type(np_defs=config.sample_from_actor_def)


def test_split_sample_from_actor(config=RNNConfigBase()):
    sample = SampleFromActor.as_random(size=3, np_defs=config.sample_from_actor_def)
    for i, result in enumerate(split_sample_from_actor(sample=sample, np_defs=config.sample_from_actor_def)):
        for name, np_def in config.sample_from_actor_def.items():
            if np_def is not None:
                assert np.array_equiv(getattr(sample, name)[i], getattr(result, name))

    sample = SampleFromActor.as_random(size=1, np_defs=config.sample_from_actor_def)
    result = split_sample_from_actor(sample=sample, np_defs=config.sample_from_actor_def)[0]


def test_from_buffer_samples(config=RNNConfigBase()):
    sample = SampleFromActor.as_random(size=3, np_defs=config.sample_from_actor_def)
    result = SampleFromBuffer.from_buffer_samples(samples=split_sample_from_actor(sample=sample,
                                                                                  np_defs=config.sample_from_actor_def),
                                                  sample_from_actor_def=config.sample_from_actor_def)
    assert len(result) == 3


def test_batch_from_sample(config=RNNConfigBase()):
    batch = Batch.from_buffer_sample(sample=SampleFromBuffer.as_random(size=10, np_defs=config.sample_from_buffer_def))
    assert len(batch) == 10
