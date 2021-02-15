"""Implementation of RNN-DQN learner."""
import copy
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dqn.episodic_curiosity import EpisodicCuriosityModule
from dqn.learner import LearnerBase, UpdateResult
from dqn.model import RNDModel
from dqn.rnn.config import RNNConfigBase
from dqn.rnn.datum import Batch, Loss
from dqn.rnn.model import Model, ModelState
from dqn.utils import inverse_value_scaling, value_scaling


class Learner(LearnerBase):
    """Implementation of RNN-DQN learner.

    Args:
        config: RNN configuration.
    """
    def __init__(self, config: RNNConfigBase, logger: logging.Logger = logging.getLogger(__name__)):
        self.logger = logger

        # basic learner config
        self.n_step = config.n_step
        self.double_dqn = config.learner.double_dqn
        self.seq_len = config.seq_len
        self.obs_shape = config.obs_shape
        self.batch_size = config.learner.batch_size
        self.apply_value_scaling = config.apply_value_scaling

        # intrinsic model config
        self.use_intrinsic_model = config.intrinsic_reward.enable
        self.rnd_predictor_train_window_size = config.intrinsic_reward.predictor_train_window_size
        self.episodic_curiosity = config.intrinsic_reward.episodic_curiosity.enable

        # initialize model
        self.online_model = Model(config)
        self.target_model = copy.deepcopy(self.online_model)
        if self.use_intrinsic_model:
            size = config.intrinsic_reward.feature_size
            self.rnd_target_model = RNDModel(config.obs_shape, size)
            self.rnd_predictor_model = RNDModel(config.obs_shape, size)
            if self.episodic_curiosity:
                self.episodic_curiosity_module = EpisodicCuriosityModule(
                    config=config.intrinsic_reward.episodic_curiosity,
                    obs_shape=config.obs_shape,
                    action_size=config.model.action_size,
                    vector_env_size=0)

        # init gpu
        self.device = None
        if config.learner.gpu_id is not None:
            assert torch.cuda.is_available(), f"{config.learner.gpu_id} cannot be utilized"
            self.logger.info(f"Use gpu = {config.learner.gpu_id}")
            self.device = torch.device(f'cuda:{config.learner.gpu_id}')
            self.online_model = self.online_model.to(self.device)
            self.target_model = self.target_model.to(self.device)
            if self.use_intrinsic_model:
                self.rnd_target_model = self.rnd_target_model.to(self.device)
                self.rnd_predictor_model = self.rnd_predictor_model.to(self.device)
                if self.episodic_curiosity:
                    self.episodic_curiosity_module.embedding_network.to(self.device)
                    self.episodic_curiosity_module.inverse_model.to(self.device)

        # init optimizer
        init_optim = lambda ps: torch.optim.Adam(ps, lr=config.learner.adam_lr, eps=config.learner.adam_eps)

        self.optimizer = init_optim(self.online_model.extrinsic_model.parameters())

        if self.use_intrinsic_model:
            self.intrinsic_optimizer = init_optim(self.online_model.intrinsic_model.parameters())
            self.rnd_optimizer = init_optim(self.rnd_predictor_model.parameters())
            if self.episodic_curiosity:
                self.episodic_embedding_optimizer = init_optim(
                    self.episodic_curiosity_module.embedding_network.parameters())
                self.episodic_inverse_optimizer = init_optim(self.episodic_curiosity_module.inverse_model.parameters())

        super(Learner, self).__init__(online_model=self.online_model,
                                      target_model=self.target_model,
                                      target_sync_interval=config.learner.target_sync_interval)

    def update_core(self, batch: Batch) -> Tuple[Loss, UpdateResult]:
        """Update online network using batch.

        Args:
            batch: Mini-batch.

        Returns:
            result: Tuple of TD-error and update result stats.
        """
        # send batch to gpu
        if self.device: batch.to_device(self.device)

        # compute target signal and prediction
        online_prediction = self.online_model.forward_sequence(
            obs=batch.obs,
            initial_state=ModelState.from_batch(batch, use_intrinsic_model=self.use_intrinsic_model))  # [b, t, a]
        with torch.no_grad():
            target_prediction = self.target_model.forward_sequence(obs=batch.obs,
                                                                   initial_state=ModelState.from_batch(
                                                                       batch,
                                                                       use_intrinsic_model=self.use_intrinsic_model))

        # update model
        extrinsic_loss, extrinsic_td_error = self.compute_loss(online_prediction.q_e, target_prediction.q_e,
                                                               batch.action, batch.is_done, batch.loss_mask,
                                                               batch.gamma, batch.discounted_reward_sum)
        self.optimizer.zero_grad()
        (extrinsic_loss * batch.weight).mean().backward()
        self.optimizer.step()

        # may update intrinsic model
        intrinsic_td_error_mean = 0.0
        intrinsic_q_value_mean = 0.0
        intrinsic_loss_mean = 0.0
        rnd_loss_value = 0.0
        episodic_curiosity_loss_mean = 0.0

        if self.use_intrinsic_model:
            assert online_prediction.q_i is not None
            assert target_prediction.q_i is not None
            intrinsic_loss, intrinsic_td_error = self.compute_loss(online_prediction.q_i, target_prediction.q_i,
                                                                   batch.action, batch.is_done, batch.loss_mask,
                                                                   batch.gamma, batch.discounted_intrinsic_reward_sum)

            self.intrinsic_optimizer.zero_grad()
            (intrinsic_loss * batch.weight).mean().backward()
            self.intrinsic_optimizer.step()

            # update RND model
            w_size = self.rnd_predictor_train_window_size
            B = len(batch)

            rnd_x_shape = [B * w_size] + self.obs_shape
            # K, W, H = self.obs_shape
            rnd_x = batch.obs[:, -w_size:]\
                        .reshape(rnd_x_shape)
            with torch.no_grad():
                rnd_target_f = self.rnd_target_model.forward(rnd_x)
            rnd_loss = F.mse_loss(rnd_target_f.detach(), self.rnd_predictor_model.forward(rnd_x))

            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()

            # update episodic curiosity models
            if self.episodic_curiosity:
                ec_x = batch.obs[:, -(w_size + 1):-1].reshape(rnd_x_shape)
                ec_x_next = batch.obs[:, -w_size:].reshape(rnd_x_shape)
                ec_a = batch.action[:, -w_size:].reshape(B * w_size)

                f_x = self.episodic_curiosity_module.embedding_network.forward(ec_x)
                f_x_next = self.episodic_curiosity_module.embedding_network.forward(ec_x_next)
                ec_a_pred = self.episodic_curiosity_module.inverse_model(torch.cat([f_x, f_x_next], dim=1))
                ec_loss = F.cross_entropy(ec_a_pred, ec_a)

                self.episodic_embedding_optimizer.zero_grad()
                self.episodic_inverse_optimizer.zero_grad()
                ec_loss.backward()
                self.episodic_embedding_optimizer.step()
                self.episodic_inverse_optimizer.step()
                episodic_curiosity_loss_mean = float(ec_loss.detach().cpu().numpy())

            # copy normalization layer's parameter to target network
            self.rnd_target_model.input_norm.load_state_dict(self.rnd_predictor_model.input_norm.state_dict())
            self.rnd_target_model.output_norm.load_state_dict(self.rnd_predictor_model.output_norm.state_dict())

            intrinsic_td_error_mean = float(intrinsic_td_error.mean())
            intrinsic_q_value_mean = float(online_prediction.q_i.mean().detach().cpu().numpy())
            intrinsic_loss_mean = float(intrinsic_loss.mean().detach().cpu().numpy())
            rnd_loss_value = float(rnd_loss.detach().cpu().numpy())

        loss = Loss(loss=extrinsic_td_error)
        result = UpdateResult(extrinsic_td_error_mean=float(extrinsic_td_error.mean()),
                              extrinsic_q_value_mean=float(online_prediction.q_e.mean().detach().cpu().numpy()),
                              extrinsic_loss_mean=float(extrinsic_loss.mean().detach().cpu().numpy()),
                              intrinsic_td_error_mean=intrinsic_td_error_mean,
                              intrinsic_q_value_mean=intrinsic_q_value_mean,
                              intrinsic_loss_mean=intrinsic_loss_mean,
                              rnd_loss_mean=rnd_loss_value,
                              episodic_curiosity_loss_mean=episodic_curiosity_loss_mean)
        return loss, result

    def compute_loss(self, q_online: torch.Tensor, q_target: torch.Tensor, action: torch.Tensor, is_done: torch.Tensor,
                     loss_mask: torch.Tensor, gamma: torch.Tensor,
                     reward_sum: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """Compute loss and TD-error for Q-network update.

        Args:
            q_online: Online Q-network prediction
            q_target: Target Q-network prediction
            action: Action sequence.
            is_done: Boolean sequence to tell episode terminations.
            loss_mask: Mask for avoid considering states after episode termination.
            gamma: Discount factor.
            reward_sum: Not discounted reward sequence.

        Returns:
            result: Tuple of loss for Q-network update, and TD-error.
        """
        n = self.n_step
        B = len(q_online)
        L = self.seq_len + 1 - n

        # compute target
        q_target = q_target[:, n:]  # shift n-step
        if self.double_dqn:
            a = q_online[:, n:].max(dim=2).indices
            max_q = q_target.reshape(B * L, -1).gather(dim=1, index=a.reshape(-1, 1)).view(B, L)
        else:
            max_q = q_target.max(dim=2).values

        if self.apply_value_scaling:
            target_value = value_scaling(reward_sum[:, :L] \
                        + gamma.unsqueeze(1).expand((B, L))**n \
                        * inverse_value_scaling(max_q) * (1 - is_done[:, :L]))
        else:
            target_value = reward_sum[:, :L] + gamma.unsqueeze(1).expand((B, L))**n * max_q * (1 - is_done[:, :L])

        # prediction of online network
        prediction = q_online[:, :L].reshape(B * L, -1).gather(dim=1, index=action[:, :L].reshape(-1, 1)).view(B, L)

        # compute loss
        loss = torch.nn.MSELoss(reduction='none')(prediction, target_value.detach()) * loss_mask[:, :L]
        loss = loss.mean(dim=1)
        td_error = td_error = torch.abs(target_value - prediction).mean(dim=1).detach().cpu().numpy()  # [b, ]
        return loss, td_error

    def get_model_param(self) -> bytes:
        """Returns bytes expression of model parameters.

        Returns:
            param: Byte data of parameters.
        """
        param = self.online_model.get_param()
        if self.use_intrinsic_model:
            param += self.rnd_target_model.get_param() + self.rnd_predictor_model.get_param()
            if self.episodic_curiosity:
                param += self.episodic_curiosity_module.embedding_network.get_param()
                param += self.episodic_curiosity_module.inverse_model.get_param()
        return param

    def save_model(self, log_dir: Path, global_step: int) -> None:
        """Save online model parameters to disk.

        Args:
            log_dir: Directory to save pickle file(s).
        """
        torch.save(self.online_model.state_dict(), log_dir / f'online_model_{global_step}.pkl')
        if self.use_intrinsic_model:
            torch.save(self.rnd_target_model.state_dict(), log_dir / f'rnd_target_model_{global_step}.pkl')
            torch.save(self.rnd_predictor_model.state_dict(), log_dir / f'rnd_predictor_model_{global_step}.pkl')
            if self.episodic_curiosity:
                torch.save(self.episodic_curiosity_module.embedding_network.state_dict(),
                           log_dir / f'ec_embedding_model_{global_step}.pkl')
                torch.save(self.episodic_curiosity_module.inverse_model.state_dict(),
                           log_dir / f'ec_inverse_model_{global_step}.pkl')

    def load_online_model(self, snap_filename: Path) -> None:
        """Load online model parameters from disk.

        Args:
            snap_filename: Save path of pickle file.
        """
        self.online_model.load_state_dict(torch.load(snap_filename))

    def get_batch_size(self) -> int:
        """Returns size of mini-batch."""
        return self.batch_size
