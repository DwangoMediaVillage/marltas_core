"""Configuration of CNN-DQN training."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from dqn.datum import NpDef
from dqn.episodic_curiosity import EpisodicCuriosityConfig
from dqn.learner import LearnerConfig
from dqn.make_env import EnvConfig
from dqn.replay_buffer import ReplayBufferConfig
from dqn.utils import ConfigBase


@dataclass
class EvaluatorConfig(ConfigBase):
    """Configuration of evaluator. If `custom_metric_keys` is specified, the metrics will be extracted from `info` dict produced by environments.

    Attributes:
        eps: epsilon-greedy parameter.
        n_eval: Number of episodes for each evaluation.
        custom_metric_keys: List of additional metrics for evaluations.
    """
    eps: float = 0.02
    n_eval: int = 3
    custom_metric_keys: List[str] = field(default_factory=lambda: [])
    custom_metric_types: List[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        assert len(self.custom_metric_keys) == len(self.custom_metric_types)


@dataclass
class ActorConfig(ConfigBase):
    """Configuration of actor.

    Attributes:
        vector_env_size: Size of vectorized environments.
    """
    vector_env_size: int = 3


@dataclass
class ModelConfig(ConfigBase):
    """Configuration of model.

    Attributes:
        action_size: Size of output.
        dueling: Using dueling network architecture or not.
    """
    action_size: int = 6
    dueling: bool = True


@dataclass
class IntrinsicRewardConfig(ConfigBase):
    """Configuration of random network distillation (RND).

    Attributes:
        enable: If true, RND will be enabled.
        feature_size: Size of RND network's output.
        episodic_curiosity: Configuration of episodic curiosity module.
        reward_ratio: Ratio of extrinsic/intrinsic rewards.
        use_ucb: If true, explorer will use UCB algorithm to decide reward ratio.
    """
    enable: bool = False
    feature_size: int = 64
    episodic_curiosity: EpisodicCuriosityConfig = EpisodicCuriosityConfig()
    reward_ratio: float = 0.15
    use_ucb: bool = False


@dataclass
class CNNConfigBase(ConfigBase):
    """Configuration of CNN-DQN training.

    Attributes:
        obs_shape: Dimensional shape of observation.
        n_step: Size of n-step Q-learning.
        gamma: Discount factor.
        apply_value_scaling: Apply value (Q-value) scaling function or not.
        env: Environment configuration.
        replay_buffer: Configuration of experience replay.
        evaluator: Evaluation configuration.
        actor: Actor configuration.
        model: Q-Network model configuration.
        learner: Learner configuration.
        intrinsic_reward: Configuration of intrinsic reward computation modules.
        actor_manager_url: Actor manager server's URL.
        evaluator_url: Evaluator server's URL.
        param_distributor_url: Paramerter distribution server's URL.
        replay_buffer_url: Replay buffer server's URL.
        n_actor_process: Size of actor subprocesses.
        local_buffer_size: Sample size of local actor's buffer.
        send_sample_interval: Environment step size interval to send samples from local buffer to replay buffer server.
        update_param_inferval: Environment step size interval to fetch actor's policy network param from param distributor server.
    """
    obs_shape: List[int] = field(default_factory=lambda: [4, 84, 84])

    n_step: int = 3
    gamma: float = 0.997
    apply_value_scaling: bool = False

    env: EnvConfig = EnvConfig()

    replay_buffer: ReplayBufferConfig = ReplayBufferConfig(prioritized_replay=True, capacity=10_000)
    evaluator: EvaluatorConfig = EvaluatorConfig()
    actor: ActorConfig = ActorConfig()
    model: ModelConfig = ModelConfig()
    learner: LearnerConfig = LearnerConfig(batch_size=64, target_sync_interval=1000)

    # intrinsic reward (curiosity)
    intrinsic_reward: IntrinsicRewardConfig = IntrinsicRewardConfig()

    # for async training
    actor_manager_url: str = 'localhost:1111'
    evaluator_url: str = 'localhost:2222'
    param_distributor_url: str = 'localhost:3333'
    replay_buffer_url: str = 'localhost:4444'

    # for actor running
    n_actor_process: int = 3
    local_buffer_size: int = 10
    send_sample_interval: int = 10
    update_param_interval: int = 1

    ### Definition of serializable NumPy datum.
    sample_from_actor_def: dict = field(default_factory=lambda: {})
    sample_from_buffer_def: dict = field(default_factory=lambda: {})
    loss_def: dict = field(default_factory=lambda: {'loss': NpDef(shape=(), dtype=np.float32)})
    batch_def: dict = field(default_factory=lambda: {})

    def save_as_yaml(self, yaml_path: Path) -> None:
        super(CNNConfigBase, self).save_as_yaml(
            yaml_path=yaml_path,
            ignored_top_keys=['sample_from_actor_def', 'sample_from_buffer_def', 'loss_def', 'batch_def'])

    def __post_init__(self):
        # init npdefs
        obs_shape = tuple(self.obs_shape)

        self.sample_from_actor_def = {
            'loss': NpDef(shape=(), dtype=np.float32),
            'obs': NpDef(shape=obs_shape, dtype=np.uint8),
            'obs_next': NpDef(shape=obs_shape, dtype=np.uint8),
            'action': NpDef(shape=(), dtype=np.uint8),
            'discounted_reward_sum': NpDef(shape=(), dtype=np.float32),
            'discounted_intrinsic_reward_sum':
            NpDef(shape=(), dtype=np.float32) if self.intrinsic_reward.enable else None,
            'gamma': NpDef(shape=(), dtype=np.float32),
            'is_done': NpDef(shape=(), dtype=np.bool),
        }

        self.sample_from_buffer_def = {
            'weight': NpDef(shape=(), dtype=np.float32),
            'obs': NpDef(shape=obs_shape, dtype=np.uint8),
            'obs_next': NpDef(shape=obs_shape, dtype=np.uint8),
            'action': NpDef(shape=(), dtype=np.uint8),
            'discounted_reward_sum': NpDef(shape=(), dtype=np.float32),
            'discounted_intrinsic_reward_sum':
            NpDef(shape=(), dtype=np.float32) if self.intrinsic_reward.enable else None,
            'gamma': NpDef(shape=(), dtype=np.float32),
            'is_done': NpDef(shape=(), dtype=np.bool),
        }

        self.loss_def = {'loss': NpDef(shape=(), dtype=np.float32)}
