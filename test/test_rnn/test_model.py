import numpy as np
import torch

from dqn.rnn.config import RNNConfigBase
from dqn.rnn.datum import Batch, SampleFromBuffer
from dqn.rnn.model import Model, ModelState


def test_vector_obs():
    config = RNNConfigBase(obs_shape=[
        2,
    ])

    model = Model(config=config)
    obs = torch.rand((1, 10, 2), dtype=torch.float32)
    state = ModelState.stack(
        [ModelState.zeros(size=config.model.lstm_hidden_size, use_intrinsic_model=config.intrinsic_reward.enable)])
    model.forward(obs[:, 0, :], state)
    model.forward_sequence(obs, initial_state=state)


def test_forward(config=RNNConfigBase()):
    model = Model(config=config)

    obs = torch.from_numpy(np.random.random((3, 4, 84, 84)).astype(np.float32))
    state = ModelState.stack(
        [ModelState.zeros(size=config.model.lstm_hidden_size, use_intrinsic_model=config.intrinsic_reward.enable)] * 3)

    for _ in range(2):
        x, state = model.forward(obs, state)


def test_forward_sequence(config=RNNConfigBase()):
    """Check `forward_seuquce()` matches to multi step `forward()`"""
    batch_size = 4
    model = Model(config=config)
    batch = Batch.from_buffer_sample(
        sample=SampleFromBuffer.as_random(size=batch_size, np_defs=config.sample_from_buffer_def))

    q_seq = model.forward_sequence(batch.obs, ModelState.from_batch(batch, config.intrinsic_reward.enable)).q_e

    q_stack = []
    state = ModelState.from_batch(batch, config.intrinsic_reward.enable)
    for t in range(batch.obs.shape[1]):
        q, state = model.forward(batch.obs[:, t], state)
        q_stack.append(q.q_e)
    q_stack = torch.stack(q_stack, 1)

    assert torch.allclose(q_seq, q_stack)
