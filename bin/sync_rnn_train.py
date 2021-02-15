"""Synchronized rnn training."""
import argparse
import logging
import pprint
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np

from dqn.policy import PolicyParam
from dqn.rnn.actor import Actor
from dqn.rnn.config import RNNConfigBase
from dqn.rnn.datum import Batch, SampleFromActor
from dqn.rnn.evaluator import Evaluator
from dqn.rnn.learner import Learner
from dqn.rnn.replay_buffer import ReplayBuffer
from dqn.sync_train import SyncTrainerConfig, sync_train
from dqn.utils import init_log_dir, init_random_seed


@dataclass
class Config(RNNConfigBase):
    """Configuration of RNN synchronized training."""
    trainer: SyncTrainerConfig = SyncTrainerConfig()


def main(log_dir: Path, config: Config, logger: logging.Logger = logging.getLogger(__name__)) -> None:
    """Run sync training.

    Args:
        log_dir: Directory to put log data.
        config: Training configuration.
        logger: Logger object.
    """
    # init config
    logger.info(pprint.pformat(asdict(config)))

    # init replay buffer
    replay_buffer = ReplayBuffer(config=config)

    # init learner
    learner = Learner(config=config)

    # init actor with epsilons = 1.0 (random agent)
    policy_param = PolicyParam(epsilon=np.ones(config.actor.vector_env_size),
                               gamma=np.ones(config.actor.vector_env_size) * config.gamma)
    actor = Actor(init_policy_param=policy_param, config=config)

    # init evaluator
    evaluator = Evaluator(config=config)

    # train
    try:
        sync_train(log_dir=log_dir,
                   replay_buffer=replay_buffer,
                   learner=learner,
                   evaluator=evaluator,
                   actor=actor,
                   concat_samples_from_actor=partial(SampleFromActor.concat, np_defs=config.sample_from_actor_def),
                   batch_from_sample=Batch.from_buffer_sample,
                   config=config.trainer)
    finally:
        actor.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synchronized RNN-DQN training.")
    parser.add_argument('log_dir', type=Path, help="Directory to put log and snapshots")
    parser.add_argument('--log_level',
                        type=str,
                        choices=('debug', 'info', 'error', 'critical'),
                        default='info',
                        help="Logging level")
    parser.add_argument('--config', type=Path, help="Path of DQN configuration YAML file.")
    parser.add_argument('--seed', type=int, default=1, help="Random seed value.")
    args = parser.parse_args()

    # init configuration
    config = Config.load_from_yaml(args.config) if args.config else Config()

    # init log_dir
    args.log_dir.mkdir(exist_ok=False, parents=False)
    init_log_dir(args.log_dir, config)

    # init logger
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='[%(asctime)s %(name)s %(levelname)s] %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S',
                        handlers=[logging.FileHandler(args.log_dir / 'train.log'),
                                  logging.StreamHandler()])

    # init random seed
    init_random_seed(args.seed)

    # start training
    main(args.log_dir, config)
