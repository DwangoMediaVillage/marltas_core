"""Base class of policy."""
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from dqn.utils import EventObject


@dataclass
class PolicyParam(EventObject):
    """Policy's parameters.

    Attributes:
        epsilon: Epsilon-greedy's epsilons.
        gamma: Discount factors.
    """
    epsilon: np.ndarray
    gamma: np.ndarray


class PolicyBase:
    """Epsilon greedy policy for vectorized envs."""
    def infer(self, obs: List[Any]) -> Any:
        """Inference of action and Q-values from given observations.

        Args:
            obs: Observation lists.
        Returns:
            any: Inference result.
        """
        raise NotImplementedError

    def on_partial_reset(self, index: int) -> None:
        """Reset vector env state of policy.

        Args:
            index: Index of vector env.
        """
        raise NotImplementedError

    def update_param(self, param: Any) -> None:
        """Update policy parameter.

        Args:
            param: Parameter to be set.
        """
        raise NotImplementedError

    def update_model_param(self, param: bytes) -> None:
        """Set new parameter of Q-network.

        Args:
            param: Parameter to be set.
        """
        raise NotImplementedError
