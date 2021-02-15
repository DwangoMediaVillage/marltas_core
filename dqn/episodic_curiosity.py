from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential

from dqn.model import ModelBase
from dqn.utils import ConfigBase


@dataclass
class EpisodicCuriosityConfig(ConfigBase):
    """Configuration of episodic curiosity module.

    Attributes:
        enable: Enable episodic curiosity module or not.
        L: Threshoild to determine intrinsic reward by RND and episodic curiosity driven reward.
        capacity: Size of episodic feature memory.
        c: Kernel's pseudo-counts constant.
        xi: Kernel's epsilon.
        s_m: Kernel maximum similarity
        k: Number of neighbors used to compute count.
        distance_moving_decay: Decapy of moving average.
        feature_size: Dimensional size of embedding.
    """
    enable: bool = False
    L: float = 5.0
    capacity: int = 1000
    c: float = 0.001
    xi: float = 0.008
    eps: float = 0.0001
    s_m: float = 8.0
    k: int = 10
    distance_moving_decay: float = 0.99
    feature_size: int = 32


class EmbeddingNetwork(ModelBase):
    """Embedding network to convert observation to feature vector.

    Args;
        feature_size: Dimensional size of output.
    """
    def __init__(self, input_shape: List[int], feature_size: int):
        super(EmbeddingNetwork, self).__init__()

        if len(input_shape) == 3:
            # image processing head
            self.head = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            self.tail = Sequential(nn.Linear(3136, feature_size), nn.ReLU())
        elif len(input_shape) == 1:
            # vector processing head
            hidden_size: int = 32
            self.head = nn.Sequential(nn.Linear(input_shape[0], hidden_size), nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size * 2), nn.ReLU())
            self.tail = Sequential(nn.Linear(hidden_size * 2, feature_size), nn.ReLU())
        else:
            raise NotImplementedError(f"Invalid input shape {input_shape}")

        self.param_info = self.init_param_info()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            x: Observation tensor.

        Returns:
            f: Embedded feature vector
        """
        x = self.head(x)
        return self.tail(torch.flatten(x, start_dim=1))


class InverseModel(ModelBase):
    """Network for inverse inference.

    Args:
        feature_size: Dimensional size of feature vector.
        action_size: Dimensional size of action.
    """
    def __init__(self, feature_size: int, action_size: int):
        super(InverseModel, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_size * 2, 128),
            nn.ReLU(),
        )
        self.tail = nn.Linear(128, action_size)
        self.param_info = self.init_param_info()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            x: Feature vector.

        Returns:
            a: Action prob logits.
        """
        x = self.head(x)
        x = self.tail(x)
        return x


class EpisodicCuriosityModule:
    """Module for computation of episodic curiosity, based on [NGU](https://arxiv.org/abs/2002.06038)

    Args:
        config: Configuration of the module.
        action_size: Action size.
        vector_env_size: Vectorized env size.
    """
    embedding_network: EmbeddingNetwork
    inverse_model: InverseModel

    def __init__(self, config: EpisodicCuriosityConfig, obs_shape: List[int], action_size: int, vector_env_size: int):
        self.config = config
        self.vector_env_size = vector_env_size
        self.embedding_network = EmbeddingNetwork(obs_shape, config.feature_size)
        self.inverse_model = InverseModel(config.feature_size, action_size)
        self.memory = [deque([], maxlen=self.config.capacity) for _ in range(self.vector_env_size)]
        self.distance_average = np.array([None for _ in range(self.vector_env_size)], dtype=np.float)

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Take a vector env step.

        Args:
            x: Observation tensor stacked along vectorized env.

        Returns:
            reward: Reward array.
        """
        # compute feature of observation
        features = self.embedding_network.forward(x).detach().cpu().numpy()
        [m.append(f) for m, f in zip(self.memory, features)]

        # may compute episodic reward
        if len(self.memory[0]) >= self.config.k:
            return np.array([self.compute_episodic_reward(i, f) for i, f in enumerate(features)])
        else:
            return np.zeros(self.vector_env_size)

    def partial_reset(self, index: int) -> None:
        """Partially clear episodic memory.

        Args:
            index: Vector env index.
        """
        self.memory[index].clear()

    def compute_episodic_reward(self, index: int, feature: np.ndarray) -> float:
        """Core implementation of episodic reward.

        Args:
            index: Vector env index.
            feature: Embedded feature vector of observations.

        Returns:
            episodic_reward: Episodic reward
        """
        # compute the k-nearest neighbors of memory and store them in N_k
        d_k = np.sort([np.linalg.norm(feature - m, ord=2) for m in self.memory[index]])[:self.config.k]

        # update moving average
        d_k_mean = d_k.mean()
        if self.distance_average[index] == np.nan:
            self.distance_average[index] = d_k_mean
        else:
            self.distance_average[index] += (1 - self.config.distance_moving_decay) * (d_k_mean -
                                                                                       self.distance_average[index])

        # normalize the distances d_k with the updated moving average
        d_n = d_k / max(1e-4, self.distance_average[index])

        # cluster the normalized distances d_n
        d_n = np.maximum(d_n - self.config.xi, 0)

        # compute the Kernel values between embedding f(x) and its neighbors N_k
        K_v = self.config.eps / (d_n + self.config.eps)

        # compute similarity between the embedding f(x) and its neighbors N_k
        s = np.sqrt(np.sum(K_v)) + self.config.c

        # compute episodic intrinsic reward
        return 0 if s > self.config.s_m else 1 / s

    def update_param(self, param: bytes) -> None:
        """Update embedding network and inverse model by given bytes data.

        Args:
            param: Parameters in bytes expression.
        """
        head = 0
        for m in (self.embedding_network, self.inverse_model):
            total_size = m.param_info['total_size']
            m.update_param(param[head:total_size])
            head += total_size
