"""Replay buffer for prioritized experience replay."""
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple

import grpc
import numpy as np

from dqn.proto_build import dqn_pb2, dqn_pb2_grpc
from dqn.utils import ConfigBase, Counter, EventObject, MovingAverage


@dataclass
class ReplayBufferConfig(ConfigBase):
    """Configuration of replay buffer.

    Attributes:
        prioritized_replay: If true, using prioritized experience replay.
        capacity: Maximum size of samples stored in replay buffer.
        error_min: Minimum of TD error clipping.
        error_max: Maximum of TD error clipping.
        eps: Small value to avoid zero division to compute priority.
        alpha: Factor to determine randomness of prioritized sampling.
        initial_beta: Beta factor at initialization.
        max_step_beta: Maximum sampling step (number of `get_sample` call) to schedule `beta` to `1.0`.
    """
    prioritized_replay: bool = False
    capacity: int = 10000

    # config for prioritized replay
    error_min: float = 0.0
    error_max: float = 1.0
    eps: float = 0.0001
    alpha: float = 0.6
    initial_beta: float = 0.4
    max_step_beta: int = 100000


@dataclass
class ReplayBufferStatus(EventObject):
    """Status of replay buffer.

    Attributes:
        size: Size of stored samples.
        beta: Value of `beta`, given at only when prioritized replay is ON.
        append_sample_per_sec: Number of samples appended per second.
        get_sample_per_sec: Number of samples taken per second.
        update_loss_per_sec: Number of samples whose priority is updated per second.
        priority_mean: Mean of priority in prioritized replay buffer.
    """
    size: int
    beta: Optional[float]
    append_sample_per_sec: Optional[float]
    get_sample_per_sec: Optional[float]
    update_loss_per_sec: Optional[float]
    priority_mean: Optional[float]


def rlock(f, timeout=-1):
    """Wrapping method of ReplayBuffer for thread safe"""
    def _wrap(*args, **kwargs):
        self = args[0]
        time.sleep(0)
        self.lock.acquire(timeout=timeout)
        ret = f(*args, **kwargs)
        self.lock.release()
        return ret

    return _wrap


class ReplayBufferBase:
    """Base class of replay buffer.

    Args:
        config: Configuration of replay buffer.
    """
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.buffer = PrioritizedReplayBuffer(self.config) if self.config.prioritized_replay else RandomReplayBuffer(
            self.config)

        # internal buffer for status
        self.append_sample_counter = Counter(buffer_len=100)
        self.get_sample_counter = Counter(buffer_len=100)
        self.update_loss_counter = Counter(buffer_len=100)

        # lock object for thread safe
        self.lock = threading.RLock()

    @rlock
    def append_sample(self, sample: Any) -> None:
        """Append samples to replay buffer

        This method requires Lock for thread safe.

        Args:
            sample: Sample object to appended.
        """
        sample, loss = self.split_sample(sample)
        self.buffer.append_sample(sample, loss if len(sample) > 1 else [loss])
        self.append_sample_counter.step(len(sample))

    def split_sample(self, sample: Any) -> Tuple[List[Any], np.ndarray]:
        """Split sample into sample list and loss array.

        Args:
            sample: Concatenated sample from actor.

        Returns:
            samples: List of splitted samples.
            loss: NumPy array of sample's loss.
        """
        raise NotImplementedError

    def post_process_sample(self, sample: List[Any], weight: np.ndarray) -> Any:
        """Post processing samples.

        Args:
            sample: List of sample data.
            weight: NumPy array of weights (computed by priorities)

        Returns:
            sample_from_buffer: Processes sample.
        """
        raise NotImplementedError

    def loss_to_ndarray(self, loss: Any) -> np.ndarray:
        """Take loss value array from loss object."""
        raise NotImplementedError

    @rlock
    def get_sample(self, size: int) -> Any:
        """Get `size` samples.

        Args:
            size: Sampling size.

        Returns:
            sample_from_buffer: Concatenated sample data.
        """
        sample, weight = self.buffer.get_sample(size)
        self.get_sample_counter.step(size)
        return self.post_process_sample(sample, weight)

    @rlock
    def update_loss(self, loss: Any) -> None:
        """Update priority of samples taken in `get_sample` last time.

        Args:
            loss: Loss object.
        """
        if self.config.prioritized_replay:
            self.buffer.update_loss(self.loss_to_ndarray(loss))
            self.update_loss_counter.step(len(loss))

    def get_status(self) -> ReplayBufferStatus:
        """Returns replay buffer status."""
        return ReplayBufferStatus(
            size=len(self.buffer),
            beta=self.buffer.beta if self.config.prioritized_replay else None,
            append_sample_per_sec=self.append_sample_counter.get_count_per_sec(),
            get_sample_per_sec=self.get_sample_counter.get_count_per_sec(),
            update_loss_per_sec=self.update_loss_counter.get_count_per_sec(),
            priority_mean=self.buffer.priority_mean.average if self.config.prioritized_replay else None)


class RandomReplayBuffer:
    """Replay buffer with random sampling.

    Args:
        config: Configuration of replay buffer.
    """
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.memory = deque([], maxlen=self.config.capacity)

    def append_sample(self, samples: List[Any], loss=np.ndarray) -> None:
        """Append sample from actor to replay buffer.

        `loss` is not used due to random sampling.

        Args:
            samples: List of sample objects.
        """
        self.memory.extend(samples)

    def get_sample(self, size: int) -> Tuple[List[Any], np.ndarray]:
        """Random sampling.

        Args:
            size: Size of sampling.

        Returns:
            samples: List of samples.
            weights: NumPy array of weights.
        """
        index = np.random.randint(low=0, high=len(self.memory), size=size)
        return [self.memory[i] for i in index], np.ones(size, dtype=np.float32)

    def __len__(self) -> int:
        """Returns size of samples."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Prioritized replay buffer.

    Args:
        config: Configuration object of replay buffer.
    """
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.memory = deque([], maxlen=self.config.capacity)
        self.priority_queue = SumTreeQueue(capacity=self.config.capacity)
        self.sampled_index: Optional[np.ndarray] = None

        self.n_get_sample_called: int = 0
        self.compute_beta = \
            lambda n: self.config.initial_beta + (1.0 - self.config.initial_beta) * (min(n, self.config.max_step_beta) / self.config.max_step_beta)
        self.beta = self.compute_beta(0)
        self.priority_mean = MovingAverage(decay=0.99)

    def __len__(self) -> int:
        """Returns size of samples."""
        return len(self.memory)

    def append_sample(self, samples: List[Any], loss: np.ndarray) -> None:
        """Append sample from actor to replay buffer.

        Args:
            samples: List of samples to be appended.
            loss: NumPy array of loss values.
        """
        sample_size = len(samples)
        priority = self.loss_to_priority(loss)
        [self.priority_queue.append(p) for p in priority]
        self.memory.extend(samples)

        # offset sampled indices
        if self.sampled_index is not None:
            self.sampled_index -= sample_size

        self.priority_mean.step(float(priority.mean()))

    def loss_to_priority(self, loss: np.ndarray) -> np.ndarray:
        """Convert loss to priority.

        Args:
            loss: NumPy array of loss values.

        Returns:
            priority: NumPy array of priorities.
        """
        return np.power(
            np.clip(loss, self.config.error_min, self.config.error_max) + self.config.eps, self.config.alpha)

    def get_sample(self, size: int) -> Tuple[List[Any], np.ndarray]:
        """Prioritized sampling.

        Args:
            size: Sampling size.

        Returns:
            Samples: List of samples.
            weight: Weight of samples computed by priority.
        """
        self.n_get_sample_called += 1
        index, priority = [], []
        for _ in range(size):
            i, p = self.priority_queue.sample()
            index.append(i)
            priority.append(p)

        self.sampled_index = np.array(index)  # backup sampled index for update loss
        return [self.memory[i] for i in index], self.priority_to_weight(np.array(priority, dtype=np.float32))

    def priority_to_weight(self, priority: np.ndarray) -> np.ndarray:
        """Convert priority to weight.

        Args:
            priority: NumPy array of priorities.

        Returns:
            weight: NumPy array of weights.
        """
        N = len(self.priority_queue)
        self.beta = self.compute_beta(self.n_get_sample_called)
        weight = np.power(N * priority, -self.beta)
        weight = weight / weight.max()
        return weight

    def update_loss(self, loss: np.ndarray) -> None:
        """Update priority of samples.

        Args:
            loss: Loss value array.

        Raises:
            AssertionError: when this function is called before `get_sample`.
            AssertionError: when given `loss` length differs from stored sample indices.
        """
        assert not self.sampled_index is None
        assert len(self.sampled_index) == len(loss)
        priority = self.loss_to_priority(loss)
        for i, p in zip(self.sampled_index, priority):
            if i >= 0: self.priority_queue.set_priority(i, p)
        self.sampled_index = None


class SumTreeQueue:
    """FIFO SumTree index queue.
    Args:
        capacity: Maximum size of queue.
    """
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.write_index: int = 0
        node_size = 1
        max_depth = 0
        while node_size < capacity:
            node_size *= 2
            max_depth += 1
        self.node_size: int = node_size
        self.max_depth: int = max_depth
        self.node: np.ndarray = np.zeros(node_size + capacity)
        self.data_size: int = 0

    def append(self, p: float) -> None:
        """Append priority `p`"""
        self._update(self.write_index + self.node_size, p)

        # next index to write priority
        self.write_index = (self.write_index + 1) % self.capacity
        if self.data_size < self.capacity: self.data_size += 1

    def _update(self, index, p) -> None:
        """Update priority of `index`-th node and its parents"""
        delta = p - self.node[index]
        self.node[index] = p
        while index != 0:
            index = index // 2
            self.node[index] += delta

    def sample(self) -> Tuple[int, float]:
        """Sample index by priority.
        Returns
            index: index in [0, data_size]
            priority: found priority
        """
        assert self.data_size > 0

        node_index = self.node_size if self.data_size == 1 else self._find(1, np.random.uniform(0, self.node[1]))
        index = node_index - self.node_size

        assert index < self.data_size
        return index, self.node[node_index]

    def _find(self, index, p) -> int:
        """Find a node from `index` """
        depth = 0
        while depth < self.max_depth:
            left_index = 2 * index
            right_index = left_index + 1
            depth += 1

            # check right node has value node at max depth
            right_has_data = right_index * (2**(self.max_depth - depth)) - self.node_size < self.data_size

            if not right_has_data or p <= self.node[left_index]:
                # go to left node
                index = left_index
            else:
                # go to right node
                p = p - self.node[left_index]
                index = right_index

        return index

    def set_priority(self, index: int, p: float) -> None:
        """Update priority of `index`-th priority"""
        assert 0 <= index < self.data_size
        self._update(index + self.node_size, p)

    @property
    def priority_sum(self) -> float:
        """Sum of priority"""
        return self.node[1]

    @property
    def priority_min(self) -> float:
        """Min of priority"""
        return self.priorities.min()

    @property
    def priorities(self) -> np.ndarray:
        """Vector of priorities"""
        return self.node[self.node_size:self.node_size + self.data_size]

    def __len__(self) -> int:
        return self.data_size


class ReplayBufferServerServicerBase(dqn_pb2_grpc.ReplayBufferServicer):
    """Base class of replay buffer server's servicer.

    Args:
        replay_buffer: ReplayBuffer object.
    """
    def __init__(self, replay_buffer: ReplayBufferBase):
        self.replay_buffer = replay_buffer

    def append_sample(self, request_iterator, context) -> dqn_pb2.Void:
        """gRPC method to store samples."""
        bytes_data = b''
        for req in request_iterator:
            bytes_data += req.data
        self.replay_buffer.append_sample(self.bytes_to_sample(bytes_data))
        return dqn_pb2.Void()

    def bytes_to_sample(self, bytes_data: bytes) -> Any:
        """Deserialize sample bytes data.

        Args:
            bytes_data: Bytes expression of samples.
        """
        raise NotImplementedError


class ReplayBufferClientBase:
    """gRPC client of replay buffer server
    Args:
        url: URL of server
        timeout_sec: Maximum second to wait launch of gRPC connection.
    """
    chunk_size = 3000000

    def __init__(self, url: str, timeout_sec: int = 10):
        self.channel = grpc.insecure_channel(url)
        grpc.channel_ready_future(self.channel).result(timeout=timeout_sec)
        self.stub = dqn_pb2_grpc.ReplayBufferStub(self.channel)

    def sample_generator(self, bytes_data: bytes) -> Generator[dqn_pb2.BytesData, None, None]:
        """Returns generator of sample's bytes.

        Yields:
            bytes_data: Chunk of bytes data.
        """
        for i in range(0, len(bytes_data), self.chunk_size):
            yield dqn_pb2.BytesData(data=bytes_data[i:min(len(bytes_data), i + self.chunk_size)])

    def append_sample(self, sample_from_actor: Any) -> None:
        """Add new samples to buffer.

        Args:
            sample_from_actor: Sample data as NpData object.
        """
        self.stub.append_sample(self.sample_generator(self.serialize_sample(sample_from_actor)))

    def serialize_sample(self, sample_from_actor: Any) -> bytes:
        """Serialize sample_from_actor.

        Args:
            sample_from_actor: Sample data.

        Returns:
            bytes_data: Bytes expression of `sample_from_actor`
        """
        raise NotImplementedError
