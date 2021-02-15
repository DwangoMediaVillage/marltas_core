"""Implementation of CNN Q-Network."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dqn.cnn.config import CNNConfigBase
from dqn.model import ModelBase


class ChildFNNModel(nn.Module):
    """DQN Network.

    Args:
        dueling: Using dueling architecture or not.
        action_size: Dimensional output size.
    """
    def __init__(self, input_size: int, dueling: bool, action_size: int):
        super(ChildFNNModel, self).__init__()
        self.input_size = input_size
        self.dueling = dueling
        self.action_size = action_size

        # vector processing head
        hidden_size: int = 32
        self.head = nn.Sequential(nn.Linear(self.input_size, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size * 2), nn.ReLU())

        # dueling tail or just linear layer
        if self.dueling:
            dueling_hidden_size: int = 32
            self.dueling_advantage = nn.Sequential(nn.Linear(hidden_size * 2, dueling_hidden_size), nn.ReLU(),
                                                   nn.Linear(dueling_hidden_size, self.action_size))
            self.dueling_value = nn.Sequential(nn.Linear(hidden_size * 2, dueling_hidden_size), nn.ReLU(),
                                               nn.Linear(dueling_hidden_size, 1))
        else:
            self.out = nn.Linear(hidden_size * 2, self.action_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            obs: Observation tensor.

        Returns:
            q_value: Predicted Q-Value.
        """
        h = self.head(obs)
        return self.forward_tail(h) if self.dueling else self.out(h)

    def forward_tail(self, h: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            h: Hidden feature tensor.
        Returns:
            q_value: Predicted Q-Value.
        """
        advantage = self.dueling_advantage(h)
        mean = advantage.mean(dim=1, keepdim=True)
        value = self.dueling_value(h)
        return value + (advantage - mean)


class ChildCNNModel(nn.Module):
    """DQN Network.

    Args:
        dueling: Using dueling architecture or not.
        action_size: Dimensional output size.
    """
    def __init__(self, dueling: bool, action_size: int):
        super(ChildCNNModel, self).__init__()
        self.dueling = dueling
        self.action_size = action_size

        # image processing head
        self.head = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # hidden layer
        self.hidden = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())

        # dueling tail or just linear layer
        if self.dueling:
            self.dueling_advantage = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, self.action_size))
            self.dueling_value = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
        else:
            self.out = nn.Linear(512, self.action_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            obs: Observation tensor.

        Returns:
            q_value: Predicted Q-Value.
        """
        x = self.head(obs)
        h = self.hidden(torch.flatten(x, start_dim=1))
        return self.forward_tail(h) if self.dueling else self.out(h)

    def forward_tail(self, h: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            h: Hidden feature tensor.
        Returns:
            q_value: Predicted Q-Value.
        """
        advantage = self.dueling_advantage(h)
        mean = advantage.mean(dim=1, keepdim=True)
        value = self.dueling_value(h)
        return value + (advantage - mean)


@dataclass
class Prediction:
    """Prediction object.

    Attributes:
        q_e: Extrinsic Q-Value.
        q_i: Intrinsic Q-Value.
    """
    q_e: torch.Tensor
    q_i: Optional[torch.Tensor]

    def as_numpy_tuple(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns NumPy expression of prediction.

        Returns:
            prediction: Tuple of NumPy arrays.
        """
        q_e = self.q_e.detach().cpu().numpy()
        q_i = self.q_i.detach().cpu().numpy() if self.q_i is not None else None
        return q_e, q_i

    def cpu(self) -> None:
        """"Fetch Q-Value tensors from GPU."""
        self.q_e = self.q_e.detach().cpu()
        if self.q_i is not None: self.q_i = self.q_i.detach().cpu()


class Model(ModelBase):
    """Implementation of CNN Q-Network.

    Args:
        config: Configuration of DQN.
    """
    def __init__(self, config: CNNConfigBase):
        super(Model, self).__init__()

        if len(config.obs_shape) == 3:
            self.extrinsic_model = ChildCNNModel(dueling=config.model.dueling, action_size=config.model.action_size)
            self.use_intrinsic_model = config.intrinsic_reward.enable
            if self.use_intrinsic_model:
                self.intrinsic_model = ChildCNNModel(dueling=config.model.dueling, action_size=config.model.action_size)
        elif len(config.obs_shape) == 1:
            self.extrinsic_model = ChildFNNModel(input_size=config.obs_shape[0],
                                                 dueling=config.model.dueling,
                                                 action_size=config.model.action_size)
            self.use_intrinsic_model = config.intrinsic_reward.enable
            if self.use_intrinsic_model:
                self.intrinsic_model = ChildFNNModel(input_size=config.obs_shape[0],
                                                     dueling=config.model.dueling,
                                                     action_size=config.model.action_size)
        else:
            raise NotImplementedError(f"Invalid obs shape = {config.obs_shape}")
        self.param_info = self.init_param_info()

    def forward(self, obs: torch.Tensor) -> Prediction:
        """Forward computation.

        Args:
            obs: Observation tensor.

        Returns:
            prediction: Prediction result.
        """
        q_e = self.extrinsic_model.forward(obs)
        q_i = self.intrinsic_model.forward(obs) if self.use_intrinsic_model else None
        return Prediction(q_e=q_e, q_i=q_i)
