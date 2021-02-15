"""Basic implementation of model, and network for random network distillation (RND)."""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn

from dqn.utils import ConfigBase


class ModelBase(nn.Module):
    """Online Q-network implementation."""
    def __init__(self):
        super(ModelBase, self).__init__()

    def init_param_info(self) -> dict:
        """Returns byte size of model parameter tensors.

        Returns:
            param_info: Key: parameter name, value: byte size of parameter array.
        """
        param_info = {}
        total_size = 0
        for name, param in self.named_parameters():
            p = param.detach().numpy()
            size = len(p.tobytes())
            param_info[name] = (size, p.shape, p.dtype)
            total_size += size
        param_info['total_size'] = total_size
        return param_info

    def get_param(self) -> bytes:
        """Returns bytes expression of parameters."""
        res = b''
        for param in self.parameters():
            res += param.detach().cpu().numpy().tobytes()
        return res

    def update_param(self, data: bytes) -> None:
        """Set new parameters.

        Args:
            data: Bytes expression of parameters.
        """
        head = 0
        for name, param in self.named_parameters():
            size, shape, dtype = self.param_info[name]
            param_array = np.frombuffer(data[head:head + size], dtype=dtype).reshape(shape)
            param.data = torch.from_numpy(param_array)
            head += size


@dataclass
class RNDModelConfig(ConfigBase):
    """Configuration of RND.

    Attributes:
        feature_size: Dimensional size of feature vector.
    """
    feature_size: int = 32


class RNDModel(ModelBase):
    """Network for random network distillation.

    Args:
        output_size: Dimensional size of network's output.
    """
    def __init__(self, input_shape: List[int], output_size: int):
        super(RNDModel, self).__init__()
        self.output_size = output_size

        self.input_norm = nn.LayerNorm(input_shape)

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

            # linear layers
            self.tail = nn.Sequential(
                nn.Linear(3136, self.output_size), \
            )
        elif len(input_shape) == 1:
            # vector processing head
            hidden_size: int = 32
            self.head = nn.Sequential(nn.Linear(input_shape[0], hidden_size), nn.ReLU(),
                                      nn.Linear(hidden_size, hidden_size * 2), nn.ReLU())

            # linear layers
            self.tail = nn.Sequential(
                nn.Linear(hidden_size * 2, self.output_size), \
            )
        else:
            raise NotImplementedError(f"Invalid input shape {input_shape}")

        self.output_norm = nn.LayerNorm((self.output_size, ))

        # store bytes size of parametrs
        self.param_info = self.init_param_info()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            obs: Observation tensor.
        """
        x = self.input_norm(obs)
        x = torch.clamp(obs, min=-5.0, max=5.0)
        x = self.head(x)
        x = self.tail(torch.flatten(x, start_dim=1))
        return self.output_norm(x)
