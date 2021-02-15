"""Asynchronized (distributed) cnn training."""
import os  # noqa isort:skip
os.environ['OMP_NUM_THREADS'] = '1'  # noqa isort:skip

import argparse
import logging
import pprint
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np

from dqn.actor_manager import ActorManagerClient, run_actor_manager_server
from dqn.actor_runner import ActorRunner
from dqn.async_train import AsyncTrainerConfig, async_train
from dqn.cnn.config import CNNConfigBase
from dqn.cnn.datum import Batch
from dqn.cnn.evaluator import run_evaluator_server
from dqn.cnn.learner import Learner
from dqn.cnn.replay_buffer import ReplayBufferServer
from dqn.cnn.run_actor import run_actor
from dqn.evaluator import EvaluatorClient, EvaluatorServerRunner
from dqn.param_distributor import (ParamDistributorClient,
                                   run_param_distributor_server)
from dqn.policy import PolicyParam
from dqn.subprocess_manager import SubprocessManager
from dqn.utils import init_log_dir, init_random_seed


@dataclass
class Config(CNNConfigBase):
    """Configuration of CNN asynchronized training."""
    trainer: AsyncTrainerConfig = AsyncTrainerConfig()


def init_actor_runner(config: Config) -> ActorRunner:
    """Initialize actor runner.

    Args:
        config: Configuration of training.
    """
    policy_param = PolicyParam(epsilon=np.ones(config.actor.vector_env_size),
                               gamma=np.ones(config.actor.vector_env_size) * config.gamma)
    actor_runner = ActorRunner(n_processes=config.n_actor_process,
                               run_actor_func=partial(run_actor, init_policy_param=policy_param, config=config))
    return actor_runner


def main_run_actor(config: Config, logger: logging.Logger = logging.getLogger(__name__)) -> None:
    """Run actor forever.

    Args:
        config: Training configuration.
        logger: Logger object.
    """
    actor_runner = init_actor_runner(config)
    logger.info("Actor runner initialized.")

    try:
        actor_runner.start()
        logger.info("Actor runner start.")
        while True:
            assert actor_runner.workers_alive, f"Actor runner's worker died."
            time.sleep(1)
    finally:
        logger.info(f"Finalize actor runner")
        actor_runner.finalize()


def main(log_dir: Path, enable_actor: bool, config: Config,
         logger: logging.Logger = logging.getLogger(__name__)) -> None:
    """Initialize and kick all the components of asynchronized training.

    Args:
        log_dir: Directory to put log data.
        config: Training configuration.
        logger: Logger object.
    """
    # show configuration
    logger.info(pprint.pformat(asdict(config)))

    # init config
    if not enable_actor:
        logger.warning('enable_actor is false. You should run actor in other process')
        config.n_actor_process = 0  # disable actor

    # NOTE: All child processes should be forked before init gRPC channel (https://github.com/grpc/grpc/issues/13873)
    subprocess_manager = SubprocessManager()

    # init actor manager
    subprocess_manager.append_worker(
        partial(run_actor_manager_server,
                url=config.actor_manager_url,
                gamma=config.gamma,
                config=config.trainer.actor_manager))

    # init param distributor
    subprocess_manager.append_worker(partial(run_param_distributor_server, url=config.param_distributor_url))

    # init evaluator
    evaluator_runner = EvaluatorServerRunner(run_evaluator_server_func=partial(run_evaluator_server, config=config))

    # may init actor
    actor_runner = init_actor_runner(config)

    # init replay buffer
    replay_buffer_server = ReplayBufferServer(config=config)

    # init learner
    learner = Learner(config=config)

    try:

        def check_subprocess_func():
            """Helper function to check child processes."""
            assert subprocess_manager.workers_alive, 'Subprocess manager worker has been dead'
            assert evaluator_runner.workers_alive, 'Evaluator runner worker has been dead'
            assert actor_runner.workers_alive, 'Actor runner worker has been dead'

        check_subprocess_func()

        # init gRPC clients
        evaluator_runner.start()
        actor_runner.start()

        evaluator_client = EvaluatorClient(url=config.evaluator_url)
        param_distributor_client = ParamDistributorClient(url=config.param_distributor_url)
        actor_manager_client = ActorManagerClient(url=config.actor_manager_url)

        # run train
        async_train(log_dir=log_dir,
                    check_subprocess_func=check_subprocess_func,
                    actor_manager_client=actor_manager_client,
                    evaluator_client=evaluator_client,
                    param_distributor_client=param_distributor_client,
                    replay_buffer_server=replay_buffer_server,
                    learner=learner,
                    batch_from_sample=Batch.from_buffer_sample,
                    config=config.trainer)
    finally:
        replay_buffer_server.finalize()
        subprocess_manager.finalize()
        evaluator_runner.finalize()
        actor_runner.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Asynchronized CNN-DQN training.")
    parser.add_argument('log_dir', type=Path, help="Directory to put log and snapshots")
    parser.add_argument('--log_level',
                        type=str,
                        choices=('debug', 'info', 'error', 'critical'),
                        default='info',
                        help="Logging level")
    parser.add_argument('--disable_actor', action='store_true', help="Disable actor module or not.")
    parser.add_argument('--run_only_actor', action='store_true', help="Running only actor module or not.")
    parser.add_argument('--config', type=Path, help="Path of DQN configuration YAML file.")
    parser.add_argument('--seed', type=int, default=1, help="Random seed value.")
    args = parser.parse_args()

    # init configuration
    config = Config.load_from_yaml(args.config) if args.config else Config()

    # init log_dir
    log_handlers = [logging.StreamHandler()]
    if not args.run_only_actor:
        args.log_dir.mkdir(exist_ok=False, parents=False)
        init_log_dir(args.log_dir, config)
        log_handlers.append(logging.FileHandler(args.log_dir / 'main.log'))

    # init logger
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='[%(asctime)s %(name)s %(levelname)s] %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S',
                        handlers=log_handlers)

    # init random seed
    init_random_seed(args.seed)

    # start training or exploration
    if args.run_only_actor:
        assert not args.disable_actor, 'run_actor should be specified without disable_actor.'
        main_run_actor(config)
    else:
        main(args.log_dir, not args.disable_actor, config)
