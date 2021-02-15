"""Implementation of RNN Q-Network."""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import torch.nn as nn

from dqn.model import ModelBase
from dqn.rnn.config import RNNConfigBase

from .datum import Batch

T = TypeVar('T', bound='Parent')


@dataclass
class ChildModelState:
    """Hidden state for single layer LSTM"""
    h: torch.Tensor  # (1, h)
    c: torch.Tensor  # (1, h)

    @classmethod
    def zeros(cls: Type[T], size: int) -> T:
        return cls(h=torch.zeros((size)), c=torch.zeros((size)))

    def as_numpy_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.h.cpu().detach().numpy(), self.c.cpu().detach().numpy()

    @classmethod
    def stack(cls: Type[T], states: List[T]) -> T:
        return cls(h=torch.stack([state.h for state in states]), c=torch.stack([state.c for state in states]))

    def to(self, device: torch.device) -> None:
        self.h = self.h.to(device)
        self.c = self.c.to(device)

    def cpu(self) -> None:
        self.h = self.h.cpu()
        self.c = self.c.cpu()

    def expand(self, size: int) -> None:
        self.h = self.h.expand(size, -1)
        self.c = self.c.expand(size, -1)


@dataclass
class ModelState:
    """Hidden states of LSTM"""
    extrinsic_state: ChildModelState
    use_intrinsic_model: bool
    intrinsic_state: Optional[ChildModelState] = None

    @classmethod
    def zeros(cls: Type[T], size: int, use_intrinsic_model: bool) -> T:
        return cls(extrinsic_state=ChildModelState.zeros(size=size),
                   intrinsic_state=ChildModelState.zeros(size=size),
                   use_intrinsic_model=use_intrinsic_model)

    def as_numpy_tuple(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        h_e, c_e = self.extrinsic_state.as_numpy_tuple()
        if self.use_intrinsic_model:
            h_i, c_i = self.intrinsic_state.as_numpy_tuple()
        else:
            h_i, c_i = None, None
        return h_e, c_e, h_i, c_i

    @classmethod
    def stack(cls: Type[T], states: List[T]) -> T:
        use_intrinsic_model = states[0].use_intrinsic_model
        extrinsic_state = ChildModelState.stack([state.extrinsic_state for state in states])
        if use_intrinsic_model:
            intrinsic_state = ChildModelState.stack([state.intrinsic_state for state in states])
        else:
            intrinsic_state = None

        return cls(extrinsic_state=extrinsic_state,
                   intrinsic_state=intrinsic_state,
                   use_intrinsic_model=use_intrinsic_model)

    @classmethod
    def from_batch(cls: Type[T], batch: Batch, use_intrinsic_model: bool) -> T:
        return cls(extrinsic_state=ChildModelState(h=batch.lstm_hidden_h_e, c=batch.lstm_hidden_c_e),
                   intrinsic_state=ChildModelState(h=batch.lstm_hidden_h_i, c=batch.lstm_hidden_c_i)
                   if use_intrinsic_model else None,
                   use_intrinsic_model=use_intrinsic_model)

    def to(self, device: torch.device) -> None:
        self.extrinsic_state.to(device)
        if self.use_intrinsic_model: self.intrinsic_state.to(device)

    def cpu(self) -> None:
        self.extrinsic_state.cpu()
        if self.use_intrinsic_model: self.intrinsic_state.cpu()

    @classmethod
    def split(cls: Type[T], state: T) -> List[T]:
        use_intrinsic_model = state.use_intrinsic_model
        batch_size = state.extrinsic_state.h.shape[0]
        return [
            cls(extrinsic_state=ChildModelState(h=state.extrinsic_state.h[i], c=state.extrinsic_state.c[i]),
                intrinsic_state=ChildModelState(h=state.intrinsic_state.h[i], c=state.intrinsic_state.c[i])
                if use_intrinsic_model else None,
                use_intrinsic_model=use_intrinsic_model) for i in range(batch_size)
        ]

    def expand(self, size: int) -> None:
        self.extrinsic_state.expand(size)
        if self.use_intrinsic_model: self.intrinsic_state.expand(size)


class ChildRNN(nn.Module):
    """DNN Network.

    Args:
        input_size: Dimensional size of input.
        dueling: Using dueling architecture or not.
        action_size: Dimensional output size.
        lstm_hidden_size: Size of LSTM cells.
    """
    def __init__(self, input_size: int, dueling: bool, action_size: int, lstm_hidden_size: int):
        self.input_size = input_size
        self.dueling = dueling
        self.action_size = action_size
        self.lstm_hidden_size = lstm_hidden_size
        super(ChildRNN, self).__init__()

        hidden_size: int = 32

        self.head = nn.Sequential(nn.Linear(self.input_size, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, self.lstm_hidden_size), nn.ReLU())
        self.lstm = nn.LSTM(input_size=self.lstm_hidden_size, hidden_size=self.lstm_hidden_size)

        # dueling tail or just linear layer
        if self.dueling:
            dueling_hidden_size: int = 32
            self.dueling_advantage = nn.Sequential(nn.Linear(self.lstm_hidden_size, dueling_hidden_size), nn.ReLU(),
                                                   nn.Linear(dueling_hidden_size, self.action_size))
            self.dueling_value = nn.Sequential(nn.Linear(self.lstm_hidden_size, dueling_hidden_size), nn.ReLU(),
                                               nn.Linear(dueling_hidden_size, 1))
        else:
            self.out = nn.Linear(self.lstm_hidden_size, self.action_size)

    def forward_tail(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            h: Hidden feature tensor.
        Returns:
            q_value: Predicted Q-Value.
        """
        if self.dueling:
            advantage = self.dueling_advantage(x)
            mean = advantage.mean(dim=1, keepdim=True)
            value = self.dueling_value(x)
            return value + (advantage - mean)
        else:
            return self.out(x)

    def forward(self, obs: torch.Tensor, state: ChildModelState) -> Tuple[torch.Tensor, ChildModelState]:
        """Forward computation for a step.

        Args:
            obs: Observation tensor.
            obs_vector: Vector observation tensor.
            state: Previous hidden state.

        Returns:
            output: Tuple of predicted Q-Value and hidden state.
        """
        x = self.head(obs)
        x, (h, c) = self.lstm(x.unsqueeze(0), (state.h.unsqueeze(0), state.c.unsqueeze(0)))
        return self.forward_tail(x[0]), ChildModelState(h=h[0], c=c[0])

    def forward_sequence(self, obs: torch.Tensor, initial_state: ChildModelState) -> torch.Tensor:
        """Forward observation sequences at once.

        Args:
            obs: Observation tensor.
            obs_vector: Vector observation tensor.
            state: Previous hidden state.

        Returns:
            output: Predicted Q-Value
        """
        batch_size, seq_len, k = obs.size()
        x = self.head(obs.view(batch_size * seq_len, k))  # [b * t, h]
        x = x.view(batch_size, seq_len, -1).permute(1, 0, 2)  # [t, b, h]
        x, _ = self.lstm(x, (initial_state.h.unsqueeze(0), initial_state.c.unsqueeze(0)))  # [t, b, h]
        x = x.permute(1, 0, 2).reshape(batch_size * seq_len, -1)  # [b * t, h]
        q_value = self.forward_tail(x)  # [b * t, h]
        return q_value.view(batch_size, seq_len, -1)


class ChildCNN(nn.Module):
    """DQN CNN Network.

    Args:
        dueling: Using dueling architecture or not.
        action_size: Dimensional output size.
        lstm_hidden_size: Size of LSTM cells.
    """
    def __init__(self, dueling: bool, action_size: int, lstm_hidden_size: int):
        self.dueling = dueling
        self.action_size = action_size
        self.lstm_hidden_size = lstm_hidden_size
        super(ChildCNN, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=3136, hidden_size=self.lstm_hidden_size)

        # dueling tail or just linear layer
        if self.dueling:
            self.dueling_advantage = nn.Sequential(nn.Linear(self.lstm_hidden_size, 128), nn.ReLU(),
                                                   nn.Linear(128, self.action_size))
            self.dueling_value = nn.Sequential(nn.Linear(self.lstm_hidden_size, 128), nn.ReLU(), nn.Linear(128, 1))
        else:
            self.out = nn.Linear(self.lstm_hidden_size, self.action_size)

    def forward_tail(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            h: Hidden feature tensor.
        Returns:
            q_value: Predicted Q-Value.
        """
        if self.dueling:
            advantage = self.dueling_advantage(x)
            mean = advantage.mean(dim=1, keepdim=True)
            value = self.dueling_value(x)
            return value + (advantage - mean)
        else:
            return self.out(x)

    def forward(self, obs: torch.Tensor, state: ChildModelState) -> Tuple[torch.Tensor, ChildModelState]:
        """Forward computation for a step.

        Args:
            obs: Observation tensor.
            obs_vector: Vector observation tensor.
            state: Previous hidden state.

        Returns:
            output: Tuple of predicted Q-Value and hidden state.
        """
        x = torch.flatten(self.head(obs), start_dim=1)
        x, (h, c) = self.lstm(x.unsqueeze(0), (state.h.unsqueeze(0), state.c.unsqueeze(0)))
        return self.forward_tail(x[0]), ChildModelState(h=h[0], c=c[0])

    def forward_sequence(self, obs: torch.Tensor, initial_state: ChildModelState) -> torch.Tensor:
        """Forward observation sequences at once.

        Args:
            obs: Observation tensor.
            obs_vector: Vector observation tensor.
            state: Previous hidden state.

        Returns:
            output: Predicted Q-Value
        """
        batch_size, seq_len, k, w, h = obs.size()
        x = torch.flatten(self.head(obs.view(batch_size * seq_len, k, w, h)), start_dim=1)  # [b * t, h]
        x = x.view(batch_size, seq_len, -1).permute(1, 0, 2)  # [t, b, h]
        x, _ = self.lstm(x, (initial_state.h.unsqueeze(0), initial_state.c.unsqueeze(0)))  # [t, b, h]
        x = x.permute(1, 0, 2).reshape(batch_size * seq_len, -1)  # [b * t, h]
        q_value = self.forward_tail(x)  # [b * t, h]
        return q_value.view(batch_size, seq_len, -1)


@dataclass
class Prediction:
    """Prediction object.

    Attributes:
        q_e: Extrinsic Q-Value.
        q_i: Intrinsic Q-Value.
    """
    q_e: torch.Tensor
    q_i: Optional[torch.Tensor] = None

    def as_numpy_tuple(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Returns NumPy expression of prediction.

        Returns:
            prediction: Tuple of NumPy arrays.
        """
        q_e = self.q_e.detach().cpu().numpy()
        q_i = self.q_i.detach().cpu().numpy() if self.q_i is not None else None
        return q_e, q_i

    def cpu(self):
        """Fetch Q-Value tensors from GPU."""
        self.q_e = self.q_e.detach().cpu()
        if self.q_i is not None:
            self.q_i = self.q_i.detach().cpu()


class Model(ModelBase):
    """Implementation of CNN Q-Network.

    Args:
        config: Configuration of DQN.
    """
    def __init__(self, config: RNNConfigBase):
        super(Model, self).__init__()
        self.dueling = config.model.dueling
        self.action_size = config.model.action_size
        self.lstm_hidden_size = config.model.lstm_hidden_size
        self.use_intrinsic_model = config.intrinsic_reward.enable

        if len(config.obs_shape) == 3:
            self.extrinsic_model = ChildCNN(dueling=self.dueling,
                                            action_size=self.action_size,
                                            lstm_hidden_size=self.lstm_hidden_size)

            self.intrinsic_model: Optional[ChildCNN] = None
            if self.use_intrinsic_model:
                self.intrinsic_model = ChildCNN(dueling=self.dueling,
                                                action_size=self.action_size,
                                                lstm_hidden_size=self.lstm_hidden_size)
        elif len(config.obs_shape) == 1:
            self.extrinsic_model = ChildRNN(input_size=config.obs_shape[0],
                                            dueling=self.dueling,
                                            action_size=self.action_size,
                                            lstm_hidden_size=self.lstm_hidden_size)
            self.intrinsic_model: Optional[ChildRNN] = None
            if self.use_intrinsic_model:
                self.intrinsic_model = ChildRNN(input_size=config.obs_shape[0],
                                                dueling=self.dueling,
                                                action_size=self.action_size,
                                                lstm_hidden_size=self.lstm_hidden_size)
        else:
            raise NotImplementedError(f"Invalid obs shape {config.obs_shape}")

        # store byte size of parameters
        self.param_info = self.init_param_info()

    def forward(self, obs: torch.Tensor, state: ModelState) -> Tuple[Prediction, ModelState]:
        """Forward computation for a step.

        Args:
            obs: Observation tensor.
            state: Previous hidden state.

        Returns:
            output: Tuple of predicted Q-Value and hidden state.
        """
        q_e, s_e = self.extrinsic_model.forward(obs, state.extrinsic_state)
        if self.use_intrinsic_model:
            q_i, s_i = self.intrinsic_model.forward(obs, state.intrinsic_state)
            return Prediction(q_e=q_e, q_i=q_i), ModelState(extrinsic_state=s_e,
                                                            intrinsic_state=s_i,
                                                            use_intrinsic_model=self.use_intrinsic_model)
        else:
            return Prediction(q_e=q_e), ModelState(extrinsic_state=s_e, use_intrinsic_model=self.use_intrinsic_model)

    def forward_sequence(self, obs: torch.Tensor, initial_state: ModelState) -> Prediction:
        """Forward observation sequences at once.

        Args:
            obs: Omage observation tensor.
            state: Previous hidden state.

        Returns:
            output: Predicted Q-Value
        """
        q_e = self.extrinsic_model.forward_sequence(obs, initial_state.extrinsic_state)
        if self.use_intrinsic_model:
            assert initial_state.intrinsic_state is not None
            q_i = self.intrinsic_model.forward_sequence(obs, initial_state.intrinsic_state)
            return Prediction(q_e=q_e, q_i=q_i)
        else:
            return Prediction(q_e=q_e)

    def get_init_state(self) -> ModelState:
        """Returns initial state of model.

        Returns:
            model_state: Zero state of LSTM layers.
        """
        return ModelState.zeros(size=self.lstm_hidden_size, use_intrinsic_model=self.use_intrinsic_model)
