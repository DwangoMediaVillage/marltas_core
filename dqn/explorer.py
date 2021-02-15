"""Meta controller of epsilon-greedy's parameter using a bandit algorithm."""
from typing import Optional

import numpy as np

from dqn.utils import np_inverse_value_scaling, np_value_scaling


def compute_beta(index: int, N: int, beta: float) -> float:
    """Sigmoid based on [Agent57 paper](http://arxiv.org/abs/2003.13350)"""
    assert index >= 0
    if index == 0:
        return 0.0
    elif index == N - 1:
        return 0.3
    else:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        return beta * sigmoid(10 * (2 * index - (N - 2)) / (N - 2))


class SlidingWindowUCB:
    """Sliding window UCB1-Tuned based on [Agent57 paper](http://arxiv.org/abs/2003.13350)"""
    def __init__(self, n_arms: int = 32, window_size: int = 520, eps: float = 0.1):
        assert n_arms < window_size
        self.N = n_arms
        self.window_size = window_size
        self.eps = eps
        self.reward_sum = np.zeros((self.window_size, self.N), dtype=np.float32)
        self.count = np.zeros((self.window_size, self.N), dtype=np.bool)
        self.selected_arm_index: Optional[int] = None
        self.k = 0
        self.reward_sum_max = -float('inf')
        self.reward_sum_min = float('inf')

    def select_arm(self) -> int:
        """Select arm index."""
        # select arm
        if self.k < self.N:
            arm_index = self.k
        else:
            if np.random.random() < self.eps:
                # take random
                arm_index = np.random.randint(low=0, high=self.N)
            else:
                i = min(self.k, self.window_size)
                n = self.count[:i].sum(axis=0) + 1  # [A, ]
                x = (1 / n) * self.reward_sum[:i].sum(axis=0)  # expected reward sum
                V = (1 / n) * np.square(self.reward_sum[:i] - x.reshape(1, -1)).sum(axis=0) + np.sqrt(
                    2 * np.log(i + 1) / n)
                c = abs(self.reward_sum_max - self.reward_sum_min) * np.sqrt(np.log(i + 1) * np.minimum(0.25, V) / n)
                arm_index = np.argmax(x + c)

        self.selected_arm_index = arm_index
        self.count[self.k % self.window_size, self.selected_arm_index] = True

        self.k += 1
        return int(self.selected_arm_index)

    def update(self, reward: float) -> None:
        """Update statistical info of arms."""
        assert self.selected_arm_index is not None
        self.reward_sum[self.k % self.window_size, self.selected_arm_index] = reward
        self.reward_sum_max = max(self.reward_sum_max, reward)
        self.reward_sum_min = min(self.reward_sum_min, reward)


class Explorer:
    """Select actions by epsilon-greedy algorithm.

    Args:
        action_size: Size of discrete action space.
        init_eps: Initial epsilon of epsilon-greedy.
        init_beta: Initial ratio of extrinsic/intrinsic rewards `[0, 1]`. `0` means 100% extrinsic reward.
        use_intrinsic_reward: If false, intrinsic reward will be ignored.
        use_ucb: If true, explorer selects `beta` by UCB meta controller.
        apply_value_scaling: Apply Q-value scaling function or not.
    """
    def __init__(self, action_size: int, init_eps: float, init_beta: float, use_intrinsic_reward: bool, use_ucb: bool,
                 apply_value_scaling: bool):
        self.action_size = action_size
        self.eps = init_eps
        self.beta = init_beta
        self.use_intrinsic_reward = use_intrinsic_reward
        self.use_ucb = self.use_intrinsic_reward and use_ucb
        self.apply_value_scaliong = apply_value_scaling

        if self.use_ucb:
            self.ucb = SlidingWindowUCB()
            self.beta = self.ucb.select_arm()
            self.reward_sum = 0.0
            self.ucb_beta = [compute_beta(n, self.ucb.N, 0.3) for n in range(self.ucb.N)]

    def compute_q(self, q_e: np.ndarray, q_i: Optional[np.ndarray], beta: float) -> np.ndarray:
        if self.use_intrinsic_reward:
            if self.apply_value_scaliong:
                return np_value_scaling(np_inverse_value_scaling(q_e) + beta * np_inverse_value_scaling(q_i))
            else:
                return q_e + beta * q_i
        else:
            return q_e

    def select_action(self, q_extrinsic: np.ndarray, q_intrinsic: Optional[np.ndarray] = None) -> int:
        """Select an action.

        Args:
            q_extrinsic: Extrinsic reward.
            q_intrinsic: Intrinsic reward.
        """
        if np.random.random() < self.eps:
            return np.random.randint(low=0, high=self.action_size)
        else:
            return np.argmax(self.compute_q(q_extrinsic, q_intrinsic, self.beta))

    def on_step(self, reward: float) -> None:
        """Increment episode reward sum.

        Args:
            reward: Extrinsic reward.
        """
        if self.use_ucb: self.reward_sum += reward

    def on_done(self) -> None:
        """Update meta controller's stat."""
        if self.use_ucb:
            self.ucb.update(self.reward_sum)
            self.beta = self.ucb_beta[self.ucb.select_arm()]
            self.reward_sum = 0  # reset stat

    def update_epsilon(self, epsilon: float) -> None:
        """Update epsilon greedy param."""
        self.eps = epsilon
