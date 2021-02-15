"""Implementation of CNN-DQN policy."""
from typing import List, Optional, Tuple

import numpy as np
import torch

from dqn.cnn.config import CNNConfigBase
from dqn.cnn.model import Model, Prediction
from dqn.episodic_curiosity import EpisodicCuriosityModule
from dqn.model import RNDModel
from dqn.policy import PolicyBase


class Policy(PolicyBase):
    """Implementation of CNN-DQN policy.

    Args:
        config: CNN configuration.
        gpu_id: If specified, model will be placed on a GPU.
    """
    def __init__(self, config: CNNConfigBase, gpu_id: Optional[int] = None):
        self.use_intrinsic_model = config.intrinsic_reward.enable
        self.episodic_curiosity = config.intrinsic_reward.episodic_curiosity.enable if self.use_intrinsic_model else None
        self.model = Model(config)
        self.action_size = config.model.action_size

        if self.use_intrinsic_model:
            self.rnd_target_model = RNDModel(config.obs_shape, config.intrinsic_reward.feature_size)
            self.rnd_predictor_model = RNDModel(config.obs_shape, config.intrinsic_reward.feature_size)
            self.rnd_loss = torch.nn.MSELoss(reduction='none')

            if self.episodic_curiosity:
                # init episodic curiosity modules
                self.env_size = config.actor.vector_env_size
                self.episodic_curiosity_L = config.intrinsic_reward.episodic_curiosity.L
                self.episodic_curiosity_module = EpisodicCuriosityModule(
                    config=config.intrinsic_reward.episodic_curiosity,
                    obs_shape=config.obs_shape,
                    action_size=config.model.action_size,
                    vector_env_size=config.actor.vector_env_size)

        self.device = None

        if gpu_id is not None:
            assert torch.cuda.is_available(), f"GPU {gpu_id} is not available"
            self.device = torch.device(f'cuda:{gpu_id}')
            self.model = self.model.to(self.device)
            if self.use_intrinsic_model:
                self.rnd_target_model = self.rnd_target_model.to(self.device)
                self.rnd_predictor_model = self.rnd_predictor_model.to(self.device)
                if self.episodic_curiosity:
                    self.episodic_curiosity_module.embedding_network.to(self.device)
                    self.episodic_curiosity_module.inverse_model.to(self.device)

    def on_partial_reset(self, index: int) -> None:
        """Reset vectorized environment's episodic memory."""
        if self.episodic_curiosity:
            self.episodic_curiosity_module.partial_reset(index)

    def infer(self, obs: List[np.ndarray]) -> Tuple[Prediction, Optional[np.ndarray]]:
        """Infers predition by observation.

        Attributes:
            obs: List of observations.

        Returns:
            inference_result: Tuple of prediction and intrinsic reward.
        """
        with torch.no_grad():
            x = self.preprocess_obs(np.stack(obs))
            prediction = self.model.forward(x)

            # may gpu -> cpu
            prediction.cpu()

            # compute intrinsic reward (RND loss)
            intrinsic_reward = None
            if self.use_intrinsic_model:
                x_feature = self.rnd_target_model(x)
                rnd_loss = self.rnd_loss(x_feature,
                                         self.rnd_predictor_model(x)).mean(dim=1).detach().cpu().numpy()  # [b, ]

                # compute episodic curiosity based reward
                if self.episodic_curiosity:
                    episodic_rewards = self.episodic_curiosity_module(x)
                    intrinsic_reward = episodic_rewards * np.minimum(np.maximum(rnd_loss, 1), self.episodic_curiosity_L)
                else:
                    intrinsic_reward = rnd_loss

        return prediction, intrinsic_reward

    def preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Uint8 NumPy array -> torch tensor [0, 1].

        Args:
            obs: Observation array.

        Returns:
            obs: PyTorch tensor of normalized observation array.
        """
        return torch.from_numpy(obs.astype(np.float32) / 255.)

    def update_model_param(self, param: bytes, only_online_model: bool) -> None:
        """Update Q-network parameters.

        Attributes:
            param: Bytes expression of new parameters.
            only_online_model: If true, only online model's parameters are updated.
        """
        total_size = self.model.param_info['total_size']
        self.model.update_param(param[:total_size])
        if only_online_model: return

        if self.use_intrinsic_model:
            head = total_size
            modules = [self.rnd_target_model, self.rnd_predictor_model]
            if self.episodic_curiosity:
                modules.extend(
                    [self.episodic_curiosity_module.embedding_network, self.episodic_curiosity_module.inverse_model])
            for m in modules:
                total_size = m.param_info['total_size']
                m.update_param(param[head:head + total_size])
                head += total_size
