"""Datum implementation of RNN-DQN."""
from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar

import numpy as np
import torch

from dqn.datum import SerializableNpData

T = TypeVar('T', bound='Parent')


@dataclass
class SampleFromActor(SerializableNpData):
    """Experience sample produced by actor.

    Attributes:
        loss: TD-error.
        obs: Observations.
        action: Action index.
        discounted_reward_sum: Discounted extrinsic reward sum.
        discounted_intrinsic_reward_sum: Discounted intrinsic reward sum.
        gamma: Discount factor.
        is_done: Termination of episode.
        lstm_hidden_h_e: Hidden state of LSTM for extrinsic Q-value prediction.
        lstm_hidden_c_e: Hidden state of LSTM for extrinsic Q-value prediction.
        lstm_hidden_h_i: Hidden state of LSTM for intrinsic Q-value prediction.
        lstm_hidden_c_i: Hidden state of LSTM for intrinsic Q-value prediction.
    """
    loss: np.ndarray
    obs: np.ndarray
    action: np.ndarray
    discounted_reward_sum: np.ndarray

    gamma: np.ndarray
    is_done: np.ndarray
    lstm_hidden_h_e: np.ndarray
    lstm_hidden_c_e: np.ndarray

    discounted_intrinsic_reward_sum: Optional[np.ndarray] = None
    lstm_hidden_h_i: Optional[np.ndarray] = None
    lstm_hidden_c_i: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.loss)

    @classmethod
    def concat(cls: Type[T], samples: List[T], np_defs: dict) -> T:
        return cls(
            **{
                name: np.concatenate([getattr(sample, name) for sample in samples], axis=0)
                for name, np_def in np_defs.items() if np_def is not None
            })


def split_sample_from_actor(sample: SampleFromActor, np_defs: dict) -> List[SampleFromActor]:
    """Split `SampleFromActor`."""
    if len(sample) == 1:
        for name, np_def in np_defs.items():
            if np_def is not None:
                setattr(sample, name, getattr(sample, name).reshape(np_def.shape).astype(np_def.dtype))
        return [sample]
    else:
        res = []
        for i in range(len(sample)):
            data = {
                name: getattr(sample, name)[i].reshape(np_def.shape).astype(np_def.dtype)
                for name, np_def in np_defs.items() if np_def is not None
            }
            res.append(SampleFromActor(**data))
        return res


@dataclass
class SampleFromBuffer(SerializableNpData):
    """Experience sample by replay buffer.

    Attributes:
        weight: Importance weight of experience.
        obs: Observations.
        action: Action index.
        discounted_reward_sum: Discounted extrinsic reward sum.
        discounted_intrinsic_reward_sum: Discounted intrinsic reward sum.
        gamma: Discount factor.
        is_done: Termination of episode.
        lstm_hidden_h_e: Hidden state of LSTM for extrinsic Q-value prediction.
        lstm_hidden_c_e: Hidden state of LSTM for extrinsic Q-value prediction.
        lstm_hidden_h_i: Hidden state of LSTM for intrinsic Q-value prediction.
        lstm_hidden_c_i: Hidden state of LSTM for intrinsic Q-value prediction.
    """
    weight: np.ndarray
    obs: np.ndarray
    action: np.ndarray
    discounted_reward_sum: np.ndarray

    gamma: np.ndarray
    is_done: np.ndarray
    lstm_hidden_h_e: np.ndarray
    lstm_hidden_c_e: np.ndarray

    discounted_intrinsic_reward_sum: Optional[np.ndarray] = None
    lstm_hidden_h_i: Optional[np.ndarray] = None
    lstm_hidden_c_i: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.weight)

    @classmethod
    def from_buffer_samples(cls: Type[T], samples: List[SampleFromActor], sample_from_actor_def: dict) -> T:
        data = {}
        for actor_name, np_def in sample_from_actor_def.items():
            if np_def is not None:
                name = actor_name if actor_name != 'loss' else 'weight'  # name of SampleFromBuffer
                data[name] = np.stack([getattr(sample, actor_name) for sample in samples], axis=0)
        return cls(**data)


@dataclass
class Loss(SerializableNpData):
    """Loss for updating priority in replay buffer."""
    loss: np.ndarray

    def __len__(self):
        return len(self.loss)


@dataclass
class Batch:
    """Mini-batch for learner."

    Attributes:
        weight: Importance weight of experience.
        obs: Observations.
        action: Action index.
        discounted_reward_sum: Discounted extrinsic reward sum.
        discounted_intrinsic_reward_sum: Discounted intrinsic reward sum.
        gamma: Discount factor.
        is_done: Termination of episode.
        loss_mask: Mask for avoid considering states after episode termination.
        lstm_hidden_h_e: Hidden state of LSTM for extrinsic Q-value prediction.
        lstm_hidden_c_e: Hidden state of LSTM for extrinsic Q-value prediction.
        lstm_hidden_h_i: Hidden state of LSTM for intrinsic Q-value prediction.
        lstm_hidden_c_i: Hidden state of LSTM for intrinsic Q-value prediction.
    """
    weight: torch.Tensor
    obs: torch.Tensor
    action: torch.Tensor
    discounted_reward_sum: torch.Tensor
    gamma: torch.Tensor
    is_done: torch.Tensor
    loss_mask: torch.Tensor
    lstm_hidden_h_e: torch.Tensor
    lstm_hidden_c_e: torch.Tensor

    discounted_intrinsic_reward_sum: Optional[torch.Tensor] = None
    lstm_hidden_h_i: Optional[torch.Tensor] = None
    lstm_hidden_c_i: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.weight)

    def to_device(self, device: torch.device) -> None:
        self.weight = self.weight.to(device)
        self.obs = self.obs.to(device)
        self.action = self.action.to(device)
        self.discounted_reward_sum = self.discounted_reward_sum.to(device)

        self.gamma = self.gamma.to(device)
        self.is_done = self.is_done.to(device)
        self.loss_mask = self.loss_mask.to(device)
        self.lstm_hidden_h_e = self.lstm_hidden_h_e.to(device)
        self.lstm_hidden_c_e = self.lstm_hidden_c_e.to(device)

        if self.discounted_intrinsic_reward_sum is not None:
            self.discounted_intrinsic_reward_sum = self.discounted_intrinsic_reward_sum.to(device)
            self.lstm_hidden_h_i = self.lstm_hidden_h_i.to(device)
            self.lstm_hidden_c_i = self.lstm_hidden_c_i.to(device)

    @classmethod
    def from_buffer_sample(cls: Type[T], sample: SampleFromBuffer) -> T:
        """Initialize `Batch` from `SampleFromBuffer`.

        Args:
            sample: Sample datum from replay buffer.
        """
        loss_mask = np.roll(np.logical_not(sample.is_done), 1, axis=1)
        loss_mask[0, :] = True

        discounted_intrinsic_reward_sum = torch.from_numpy(
            sample.discounted_intrinsic_reward_sum) if sample.discounted_intrinsic_reward_sum is not None else None
        lstm_hidden_h_i = torch.from_numpy(sample.lstm_hidden_h_i) if sample.lstm_hidden_h_i is not None else None
        lstm_hidden_c_i = torch.from_numpy(sample.lstm_hidden_c_i) if sample.lstm_hidden_c_i is not None else None

        return cls(weight=torch.from_numpy(sample.weight),
                   obs=torch.from_numpy(sample.obs.astype(np.float32) / 255.0),
                   action=torch.from_numpy(sample.action.astype(np.int64)),
                   discounted_reward_sum=torch.from_numpy(sample.discounted_reward_sum),
                   discounted_intrinsic_reward_sum=discounted_intrinsic_reward_sum,
                   gamma=torch.from_numpy(sample.gamma),
                   is_done=torch.from_numpy(sample.is_done.astype(np.float32)),
                   loss_mask=torch.from_numpy(loss_mask.astype(np.float32)),
                   lstm_hidden_h_e=torch.from_numpy(sample.lstm_hidden_h_e),
                   lstm_hidden_c_e=torch.from_numpy(sample.lstm_hidden_c_e),
                   lstm_hidden_h_i=lstm_hidden_h_i,
                   lstm_hidden_c_i=lstm_hidden_c_i)
