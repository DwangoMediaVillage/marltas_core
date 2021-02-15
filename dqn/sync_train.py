"""Run synchronized training."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dqn.actor import ActorBase
from dqn.actor_manager import ActorManagerConfig, ActorManagerStatus
from dqn.evaluator import EvaluationRequest, EvaluatorBase
from dqn.learner import LearnerBase
from dqn.policy import PolicyParam
from dqn.replay_buffer import ReplayBufferBase
from dqn.utils import ConfigBase, EpsilonSampler


@dataclass
class SyncTrainerConfig(ConfigBase):
    """Configuration of synchronized training.

    Args:
        learner_step_per_global_step: Number of learner's updates at a global step.
        param_sync_interval: Global step interval to sync learner's model parameter with param distributor server.
        eval_interval: Global step interval to request evaluation.
        log_interval: Global step interval to write and print status of the modules.
        snap_interval: Global step interval to snap DQN model.
        initial_sample_size: Size of samples in replay buffer to start learner.
        max_global_step: Maximum global step size.
    """
    learner_step_per_global_step: int = 1
    param_sync_interval: int = 5
    eval_interval: int = 100
    log_interval: int = 10
    snap_interval: int = 100
    initial_sample_size: int = 100
    max_global_step: int = 100
    actor_manager: ActorManagerConfig = ActorManagerConfig()


def sync_train(log_dir: Path,
               replay_buffer: ReplayBufferBase,
               learner: LearnerBase,
               evaluator: EvaluatorBase,
               actor: ActorBase,
               concat_samples_from_actor: Callable[[List[Any]], Any],
               batch_from_sample: Callable[[Any], Any],
               config: SyncTrainerConfig,
               logger: logging.Logger = logging.getLogger(__name__)) -> None:
    """Helper for synchronized training.

    Args:
        log_dir: Directory to put log data.
        replay_buffer: Replay buffer.
        learner: Learner.
        evaluator: Evaluator.
        actor: Actor.
        concat_samples_from_actor: Function to concatenate samples from actor for replay buffer.
        batch_from_sample: Function to convert samples from replay buffer to learner's mini batch.
        config: Configuration of training.
    """
    # init summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # sync model parameter
    actor.update_model_param(learner.get_model_param())

    global_step = 0

    # wait initial samples
    while True:
        samples = actor.step()
        if len(samples): replay_buffer.append_sample(concat_samples_from_actor(samples))
        if replay_buffer.get_status().size >= config.initial_sample_size:
            logger.info('Initial samples are stored in replay buffer')
            break

    # init epsilon greedy param sampler
    epsilon_sampler = EpsilonSampler(init_eps_base=config.actor_manager.init_eps_base,
                                     final_eps_base=config.actor_manager.final_eps_base,
                                     init_alpha=config.actor_manager.init_alpha,
                                     final_alpha=config.actor_manager.final_alpha,
                                     final_step=config.actor_manager.final_eps_alpha_step,
                                     min_eps=config.actor_manager.min_eps)
    # update main
    while global_step < config.max_global_step:
        # update policy param (epsilon)
        policy_param: PolicyParam = actor.get_policy_param()
        policy_param.epsilon = epsilon_sampler(step=global_step,
                                               index=np.arange(actor.vector_env_size),
                                               max_index=actor.vector_env_size)
        actor.update_policy_param(policy_param)

        # sampling
        samples = actor.step()
        if len(samples): replay_buffer.append_sample(concat_samples_from_actor(samples))

        # replay and update priority
        for _ in range(config.learner_step_per_global_step):
            loss = learner.update(batch_from_sample(replay_buffer.get_sample(learner.get_batch_size())))
            replay_buffer.update_loss(loss)

        global_step += 1

        # snap model
        if global_step % config.snap_interval == 0:
            logger.info(f'[{global_step}] snap model')
            learner.save_model(log_dir=log_dir, global_step=global_step)

        # sync model parameter
        if global_step == 1 or global_step % config.param_sync_interval == 0:
            actor.update_model_param(learner.get_model_param())

        # run evaluation
        if global_step == 1 or global_step % config.eval_interval == 0:
            evaluator.update_model_param(learner.get_model_param())
            result = evaluator.evaluate(EvaluationRequest(global_step=global_step))
            result.write_summary(writer, global_step, namespace=Path('Evaluator'))
            logger.info(f'[{global_step}] Evaluator:\n{result.as_format_str()}')

        # logging
        if global_step == 1 or global_step % config.log_interval == 0:
            stats = {}
            stats['ReplayBuffer'] = replay_buffer.get_status()
            stats['Learner'] = learner.get_status()
            stats['Actor'] = ActorManagerStatus.from_actor_status([actor.get_status()])

            logger.info(f"=================== [{global_step}] ===================")
            logger.info(f"log_dir: {log_dir}")
            for name, stat in stats.items():
                logger.info(f'{name}:\n{stat.as_format_str()}')
                stat.write_summary(writer, global_step, namespace=Path(name))
