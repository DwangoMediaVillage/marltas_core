"""Base class of actor."""
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import gym
import numpy as np

from dqn.make_env import EnvConfig, make_env
from dqn.utils import Counter, EventObject, MovingAverage


@dataclass
class ActorStatus(EventObject):
    """Status of actor.

    Attributes:
        episode: Number of episodes since training start.
        step: Number of environment steps since training start
        reward_sum: Moving average of non-discounted reward sum per episode.
        epsilon: Mean of epsilon-greedy parameters.
        gamma: Mean of discounted factors.
        episode_len: Moving average of episode length
        intrinsic_reward_sum: Moving average of non-discounted intrinsic reward sum per episode.
        ucb_arm_index: Arm indices of UCB mete contoller selected in each actor.
    """
    episode: int
    step: int
    reward_sum: List[Optional[float]]
    epsilon: np.ndarray
    gamma: np.ndarray
    episode_len_mean: List[Optional[float]]
    intrinsic_reward_sum: List[Optional[float]]
    ucb_arm_index: Optional[List[int]]


class ActorBase:
    """Base of actor to collect training samples by policy.

    args:
        vector_env_size: Number of envs in a process.
        process_index: Index of process in the node.
        env_config: Configuration of environment.
        make_env_func: Function to init an gym environment.
    """
    def __init__(self,
                 vector_env_size: int,
                 process_index: int,
                 env_config: EnvConfig,
                 make_env_func: Callable[[EnvConfig], gym.Env] = make_env,
                 logger: logging.Logger = logging.getLogger(__name__)):
        self.logger = logger
        self.logger.info(f"Initialize actor: {process_index}")

        self.vector_env_size = vector_env_size
        self.process_index = process_index

        # init envs
        self.envs = [make_env_func(env_config) for _ in range(self.vector_env_size)]

        self.obs = None

        # internal buffer for status
        self.reward_sum_mean = [MovingAverage(0.9) for _ in range(self.vector_env_size)]
        self.reward_sum = np.zeros(self.vector_env_size)

        self.episode_len_mean = [MovingAverage(0.9) for _ in range(self.vector_env_size)]
        self.episode_len = np.zeros(self.vector_env_size)

        self.step_counter = [Counter() for _ in range(self.vector_env_size)]
        self.episode_counter = [Counter() for _ in range(self.vector_env_size)]

        self.intrinsic_reward_sum_mean = [MovingAverage(0.9) for _ in range(self.vector_env_size)]
        self.intrinsic_reward_sum = np.zeros(self.vector_env_size)

    def step(self) -> List[Any]:
        """Take a step of environments and may return collected samples."""
        raise NotImplementedError

    def update_policy_param(self, policy_param: Any) -> None:
        """Update policy's parameters."""
        raise NotImplementedError

    def get_policy_param(self) -> Any:
        """Get policy's parameters."""
        raise NotImplementedError

    def update_model_param(self, param: bytes) -> None:
        """Set new Q-network's parameters."""
        raise NotImplementedError

    def get_status(self) -> ActorStatus:
        """Get status of actor."""
        policy_param = self.get_policy_param()
        return ActorStatus(episode=sum([c.count for c in self.episode_counter]),
                           step=sum([c.count for c in self.step_counter]),
                           reward_sum=[m.average for m in self.reward_sum_mean],
                           epsilon=policy_param.epsilon,
                           gamma=policy_param.gamma,
                           episode_len_mean=[m.average for m in self.episode_len_mean],
                           intrinsic_reward_sum=[m.average for m in self.intrinsic_reward_sum_mean],
                           ucb_arm_index=[e.ucb.selected_arm_index
                                          for e in self.explorers] if self.explorers[0].use_ucb else None)

    def finalize(self) -> None:
        """Close environments."""
        try:
            [env.close() for env in self.envs]
        except Exception as e:
            self.logger.error(f"Failed to close envs {e}")
