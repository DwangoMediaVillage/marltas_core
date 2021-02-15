"""Run an asynchronized training."""
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from torch.utils.tensorboard import SummaryWriter

from dqn.actor_manager import (ActorManagerClient, ActorManagerConfig,
                               MetaPolicyParam)
from dqn.evaluator import EvaluationRequest, EvaluatorClient
from dqn.learner import LearnerBase
from dqn.param_distributor import ParamDistributorClient
from dqn.utils import ConfigBase


@dataclass
class AsyncTrainerConfig(ConfigBase):
    """Configuration of asynchronized training.

    Args:
        initial_sample_size: Size of samples in replay buffer to start learner.
        max_global_step: Maximum global step size.
        epsilon_greedy_eps_base: Parameter to sample epsilon greedy params.
        epsilon_greedy_alpha_base: Parameter to sample epsilon greedy params.
        param_sync_interval: Global step interval to sync learner's model parameter with param distributor server.
        eval_interval: Global step interval to request evaluation.
        log_interval: Global step interval to write and print status of the modules.
        snap_interval: Global step interval to snap DQN model.
    """
    initial_sample_size: int = 100
    max_global_step: int = 100
    epsilon_greedy_eps_base: float = 0.7
    epsilon_greedy_alpha_base: float = 5.0
    param_sync_interval: int = 5
    eval_interval: int = 100
    log_interval: int = 10
    snap_interval: int = 100
    actor_manager: ActorManagerConfig = ActorManagerConfig()


def async_train(log_dir: Path,
                check_subprocess_func: Callable[[], None],
                actor_manager_client: ActorManagerClient,
                evaluator_client: EvaluatorClient,
                param_distributor_client: ParamDistributorClient,
                replay_buffer_server: Any,
                learner: LearnerBase,
                batch_from_sample: Callable[[Any], Any],
                config: AsyncTrainerConfig,
                logger: logging.Logger = logging.getLogger(__name__)) -> None:
    """Do asynchronous training.

    Args:
        log_dir: Directory to put log data.
        check_subprocess_func: Function to check whether subprocesses are alive or not.
        actor_manager_client: gRPC client of actor manager.
        evaluator_client: gRPC client of evaluator.
        param_distributor_client: gRPC client of parameter distributor server.
        replay_buffer_server: gRPC server of replay buffer.
        learner: Learner.
        batch_from_sample: Function to convert samples from replay buffer to learner's batch.
        config: Configuration of async training.
    """
    # init summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # sync model parameter
    param_distributor_client.update_param(learner.get_model_param())

    global_step = 0

    # wait initial samples
    while True:
        check_subprocess_func()

        size = replay_buffer_server.get_status().size
        if size >= config.initial_sample_size:
            logger.info("Initial samples are ready on replay buffer")
            break
        else:
            time.sleep(2)
            logger.info(f'Waiting initial samples {size}/{config.initial_sample_size} ....')

    # main
    while global_step < config.max_global_step:
        check_subprocess_func()  # check sub processes are alive

        # update meta epsilon greedy param
        actor_manager_client.update_meta_policy_param(MetaPolicyParam(step=global_step))

        # replay by sample from local replay buffer
        loss = learner.update(batch_from_sample(replay_buffer_server.get_sample(learner.get_batch_size())))

        # update priority
        replay_buffer_server.update_loss(loss)
        global_step += 1

        # snap model
        if global_step % config.snap_interval == 0:
            logger.info(f'[{global_step}] snap model')
            learner.save_model(log_dir=log_dir, global_step=global_step)

        # sync model parameter
        if global_step == 1 or global_step % config.param_sync_interval == 0:
            param_distributor_client.update_param(learner.get_model_param())

        # request async evaluation
        if global_step == 1 or global_step % config.eval_interval == 0:
            evaluator_client.request_evaluation(EvaluationRequest(global_step=global_step, walltime=time.time()))

        # logging
        if global_step == 1 or global_step % config.log_interval == 0:
            stats = {}
            stats['ReplayBuffer'] = replay_buffer_server.get_status()
            stats['Learner'] = learner.get_status()
            stats['Evaluator'] = evaluator_client.get_status()
            stats['Actor'] = actor_manager_client.get_status()
            stats['ParamDistributor'] = param_distributor_client.get_status()

            logger.info(f"=================== [{global_step}] ===================")
            logger.info(f"log_dir: {log_dir}")
            for name, stat in stats.items():
                logger.info(f'{name}:\n{stat.as_format_str()}')
                stat.write_summary(writer, global_step, namespace=Path(name))
