"""Base learner class"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from dqn.model import ModelBase
from dqn.utils import ConfigBase, Counter, EventObject, MovingAverage


@dataclass
class LearnerConfig(ConfigBase):
    """Configuration of learner.

    Attributes:
        batch_size: Size of mini-batch.
        target_sync_interval: Online update interval to copy online network to target network.
        gpu_id: Index of GPU.
        adam_lr: Learning rate (alpha) of adam optimizer.
        adam_eps: Epsilon of adam optimizer.
        double_dqn: Using double-DQN algorithm to compute TD-error or not.
    """
    batch_size: int
    target_sync_interval: int
    gpu_id: Optional[int] = None
    adam_lr: float = 0.0001
    adam_eps: float = 1e-8
    double_dqn: bool = True


@dataclass
class LearnerStatus(EventObject):
    """Status of learner.

    Args:
        online_update: Number of online network updates.
        target_update: Number of target network updates.
        online_update_per_sec: Online updates per second.
        loss: Moving average of loss.
        td_error_mean: Moving average of td error mean.
        q_value_mean: Moving average of predicted Q-values.
    """
    online_update: int
    target_update: int
    online_update_per_sec: Optional[float]

    extrinsic_loss: Optional[float]
    intrinsic_loss: Optional[float]

    extrinsic_td_error_mean: Optional[float]
    intrinsic_td_error_mean: Optional[float]

    extrinsic_q_value_mean: Optional[float]
    intrinsic_q_value_mean: Optional[float]

    rnd_loss_mean: Optional[float]
    episodic_curiosity_loss_mean: Optional[float]


@dataclass
class UpdateResult:
    extrinsic_td_error_mean: float
    extrinsic_q_value_mean: float
    extrinsic_loss_mean: float
    intrinsic_td_error_mean: float
    intrinsic_q_value_mean: float
    intrinsic_loss_mean: float
    rnd_loss_mean: float
    episodic_curiosity_loss_mean: float


class LearnerBase:
    """Base of learner.

    Args:
        model: Online network model.
        config: Configuration of learner.
    """
    def __init__(self,
                 online_model: ModelBase,
                 target_model: ModelBase,
                 target_sync_interval: int,
                 logger: logging.Logger = logging.getLogger(__name__)):
        self.logger = logger
        self.online_model = online_model
        self.target_model = target_model
        self.target_sync_interval = target_sync_interval

        # internal buffer for status
        self.extrinsic_loss_mean = MovingAverage(0.99)
        self.extrinsic_q_value_mean = MovingAverage(0.99)
        self.extrinsic_td_error_mean = MovingAverage(0.99)

        self.intrinsic_loss_mean = MovingAverage(0.99)
        self.intrinsic_q_value_mean = MovingAverage(0.99)
        self.intrinsic_td_error_mean = MovingAverage(0.99)

        self.online_update_counter = Counter()
        self.target_update_counter = Counter()
        self.rnd_loss_mean = MovingAverage(0.99)

        self.episodic_curiosity_loss_mean = MovingAverage(0.99)

        self.logger.info(f"Learner is initialized")

    # def init_optimizer(self) -> None:
    def update(self, batch: Any) -> Any:
        """Update online network (and target network).

        Args:
            batch: mini batch.

        Returns:
            loss: Loss object for priority update.
        """
        loss, result = self.update_core(batch)

        # update status
        self.extrinsic_loss_mean.step(result.extrinsic_loss_mean)
        self.extrinsic_q_value_mean.step(result.extrinsic_q_value_mean)
        self.extrinsic_td_error_mean.step(result.extrinsic_td_error_mean)

        self.intrinsic_loss_mean.step(result.intrinsic_loss_mean)
        self.intrinsic_q_value_mean.step(result.intrinsic_q_value_mean)
        self.intrinsic_td_error_mean.step(result.intrinsic_td_error_mean)

        self.rnd_loss_mean.step(result.rnd_loss_mean)
        self.online_update_counter.step()

        self.episodic_curiosity_loss_mean.step(result.episodic_curiosity_loss_mean)

        # update target model
        if self.online_update_counter.count % self.target_sync_interval == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())
            self.target_update_counter.step()

        return loss

    def update_core(self, batch: Any) -> Tuple[Any, UpdateResult]:
        """Core implementation of online update.

        Args:
            batch: mini batch.

        Returns:
            loss_object: Loss object for priority update.
            result: UpdateResult object for stats.
        """
        raise NotImplementedError

    def get_status(self) -> LearnerStatus:
        """Returns status of learner.

        Returns:
            learner_states: LearnerStatus object.
        """
        return LearnerStatus(online_update=self.online_update_counter.count,
                             target_update=self.target_update_counter.count,
                             online_update_per_sec=self.online_update_counter.get_count_per_sec(),
                             extrinsic_loss=self.extrinsic_loss_mean.average,
                             extrinsic_td_error_mean=self.extrinsic_td_error_mean.average,
                             extrinsic_q_value_mean=self.extrinsic_q_value_mean.average,
                             intrinsic_loss=self.intrinsic_loss_mean.average,
                             intrinsic_td_error_mean=self.intrinsic_td_error_mean.average,
                             intrinsic_q_value_mean=self.intrinsic_q_value_mean.average,
                             rnd_loss_mean=self.rnd_loss_mean.average,
                             episodic_curiosity_loss_mean=self.episodic_curiosity_loss_mean.average)

    def get_model_param(self) -> bytes:
        """Returns bytes expression of online network parameters."""
        raise NotImplementedError

    def save_model(self, log_dir: Path, global_step: int) -> None:
        """Save online model parameters to disk.

        Args:
            log_dir: Directory to save pickle file(s).
        """
        raise NotImplementedError

    def load_online_model(self, snap_filename: Path) -> None:
        """Load online model parameters from disk.

        Args:
            snap_filename: Save path of pickle file.
        """
        raise NotImplementedError

    def get_batch_size(self) -> int:
        """Returns size of mini-batch."""
        raise NotImplementedError
