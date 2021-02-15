"""Datum implementation of CNN-DQN."""
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
        obs: Observation on current step.
        obs_next: Observation after n-step.
        action: Action index.
        discounted_reward_sum: Discounted extrinsic reward sum.
        discounted_intrinsic_reward_sum: Discounted intrinsic reward sum.
        gamma: Discount factor.
        is_done: Termination of episode.
    """
    loss: np.ndarray
    obs: np.ndarray
    obs_next: np.ndarray
    action: np.ndarray
    discounted_reward_sum: np.ndarray
    gamma: np.ndarray
    is_done: np.ndarray

    discounted_intrinsic_reward_sum: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.loss)

    @classmethod
    def concat(cls: Type[T], samples: List[T], sample_from_actor_def: dict) -> T:
        return cls(
            **{
                name: np.concatenate([getattr(sample, name) for sample in samples], axis=0)
                for name, np_def in sample_from_actor_def.items() if np_def is not None
            })


def split_sample_from_actor(sample: SampleFromActor, sample_from_actor_def: dict) -> List[SampleFromActor]:
    """Split `SampleFromActor`."""
    if len(sample) == 1:
        for name, np_def in sample_from_actor_def.items():
            if np_def is not None:
                setattr(sample, name, getattr(sample, name).reshape(np_def.shape).astype(np_def.dtype))
        return [sample]
    else:
        res = []
        for i in range(len(sample)):
            data = {
                name: getattr(sample, name)[i].reshape(np_def.shape).astype(np_def.dtype)
                for name, np_def in sample_from_actor_def.items() if np_def is not None
            }
            res.append(SampleFromActor(**data))
        return res


@dataclass
class SampleFromBuffer(SerializableNpData):
    """Experience sample by replay buffer.

    Attributes:
        weight: Importance weight of experience.
        obs: Observation on current step.
        obs_next: Observation after n-step.
        action: Action index.
        discounted_reward_sum: Discounted extrinsic reward sum.
        discounted_intrinsic_reward_sum: Discounted intrinsic reward sum.
        gamma: Discount factor.
        is_done: Termination of episode.
    """
    weight: np.ndarray
    obs: np.ndarray
    obs_next: np.ndarray
    action: np.ndarray
    discounted_reward_sum: np.ndarray
    gamma: np.ndarray
    is_done: np.ndarray

    discounted_intrinsic_reward_sum: Optional[np.ndarray] = None

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
        obs: Observation on current step.
        obs_next: Observation after n-step.
        action: Action index.
        discounted_reward_sum: Discounted extrinsic reward sum.
        discounted_intrinsic_reward_sum: Discounted intrinsic reward sum.
        gamma: Discount factor.
        is_done: Termination of episode.
        torch_mode: If true, attributes are PyTorch tensor.
    """
    weight: torch.Tensor
    obs: torch.Tensor
    obs_next: torch.Tensor
    action: torch.Tensor
    discounted_reward_sum: torch.Tensor
    gamma: torch.Tensor
    is_done: torch.Tensor

    discounted_intrinsic_reward_sum: Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.weight)

    def to_device(self, device: torch.device) -> None:
        self.weight = self.weight.to(device)
        self.obs = self.obs.to(device)
        self.obs_next = self.obs_next.to(device)
        self.action = self.action.to(device)
        self.discounted_reward_sum = self.discounted_reward_sum.to(device)
        self.gamma = self.gamma.to(device)
        self.is_done = self.is_done.to(device)
        if self.discounted_intrinsic_reward_sum is not None:
            self.discounted_intrinsic_reward_sum = self.discounted_intrinsic_reward_sum.to(device)

    @classmethod
    def from_buffer_sample(cls: Type[T], sample: SampleFromBuffer) -> T:
        """Initialize `Batch` from `SampleFromBuffer`.

        Attributes:
            sample: Sample datum from replay buffer.
        """
        return cls(weight=torch.from_numpy(sample.weight),
                   obs=torch.from_numpy(sample.obs.astype(np.float32) / 255.),
                   obs_next=torch.from_numpy(sample.obs_next.astype(np.float32) / 255.),
                   action=torch.from_numpy(sample.action.astype(np.int64)),
                   discounted_reward_sum=torch.from_numpy(sample.discounted_reward_sum),
                   discounted_intrinsic_reward_sum=torch.from_numpy(sample.discounted_intrinsic_reward_sum)
                   if sample.discounted_intrinsic_reward_sum is not None else None,
                   gamma=torch.from_numpy(sample.gamma),
                   is_done=torch.from_numpy(sample.is_done.astype(np.float32)))
