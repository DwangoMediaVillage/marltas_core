"""Implementation of replay buffer for CNN-DQN."""
import logging
from concurrent import futures
from typing import Any, List, Tuple

import grpc
import numpy as np

from dqn.cnn.config import CNNConfigBase
from dqn.cnn.datum import (Loss, SampleFromActor, SampleFromBuffer,
                           split_sample_from_actor)
from dqn.proto_build import dqn_pb2_grpc
from dqn.replay_buffer import (ReplayBufferBase, ReplayBufferClientBase,
                               ReplayBufferServerServicerBase,
                               ReplayBufferStatus)


class ReplayBuffer(ReplayBufferBase):
    """Replay buffer implementation.

    Args:
        config: Configuration of CNN.
    """
    def __init__(self, config: CNNConfigBase):
        super(ReplayBuffer, self).__init__(config.replay_buffer)
        self.sample_from_actor_def = config.sample_from_actor_def

    def split_sample(self, sample: SampleFromActor) -> Tuple[List[SampleFromActor], np.ndarray]:
        return split_sample_from_actor(sample=sample, sample_from_actor_def=self.sample_from_actor_def), sample.loss

    def post_process_sample(self, sample: List[SampleFromActor], weight: np.ndarray) -> Any:
        s = SampleFromBuffer.from_buffer_samples(samples=sample, sample_from_actor_def=self.sample_from_actor_def)
        s.weight = weight
        return s

    def loss_to_ndarray(self, loss: Loss) -> np.ndarray:
        return loss.loss


class ReplayBufferServicer(ReplayBufferServerServicerBase):
    """Implementation of replay buffer server."""
    def __init__(self, replay_buffer: ReplayBuffer):
        super(ReplayBufferServicer, self).__init__(replay_buffer)
        self.sample_from_actor_def = replay_buffer.sample_from_actor_def

    def bytes_to_sample(self, bytes_data: bytes) -> SampleFromActor:
        return SampleFromActor.from_bytes(np_defs=self.sample_from_actor_def, bytes_data=bytes_data)


class ReplayBufferClient(ReplayBufferClientBase):
    """Implementation of replay buffer client."""
    def __init__(self, config: CNNConfigBase):
        super(ReplayBufferClient, self).__init__(url=config.replay_buffer_url)
        self.sample_from_actor_def = config.sample_from_actor_def

    def serialize_sample(self, sample_from_actor: SampleFromActor) -> bytes:
        return sample_from_actor.to_bytes(np_defs=self.sample_from_actor_def)


class ReplayBufferServer:
    """Helper class for running replay buffer on main thread."""
    def __init__(self, config: CNNConfigBase, logger: logging.Logger = logging.getLogger(__name__)):
        self.logger = logger
        self.replay_buffer = ReplayBuffer(config)
        self.servicer = ReplayBufferServicer(self.replay_buffer)
        self.executor = futures.ThreadPoolExecutor(max_workers=1)
        self.server = grpc.server(self.executor)
        dqn_pb2_grpc.add_ReplayBufferServicer_to_server(self.servicer, self.server)
        self.server.add_insecure_port(config.replay_buffer_url)
        self.server.start()
        self.logger.info(f'Replay buffer server started {config.replay_buffer_url}')

    def get_sample(self, size: int) -> SampleFromBuffer:
        return self.replay_buffer.get_sample(size)

    def update_loss(self, loss: Loss) -> None:
        self.replay_buffer.update_loss(loss)

    def finalize(self) -> None:
        self.executor.shutdown()

    def get_status(self) -> ReplayBufferStatus:
        return self.replay_buffer.get_status()
