"""Helper function to run RNN-DQN actor."""
import logging
import os
from collections import deque
from queue import Queue
from typing import Deque

from dqn.actor_manager import ActorManagerClient, ActorTag
from dqn.param_distributor import ParamDistributorClient
from dqn.policy import PolicyParam
from dqn.rnn.actor import Actor
from dqn.rnn.config import RNNConfigBase
from dqn.rnn.datum import SampleFromActor
from dqn.rnn.replay_buffer import ReplayBufferClient


def run_actor(init_policy_param: PolicyParam, process_index: int, start_queue: Queue, config: RNNConfigBase) -> None:
    """Run RNN-DQN actor.

    Args:
        init_policy_param: Initial policy param.
        process_index: Process identifier.
        start_queue: mp.Queue to sync gRPC client initializations.
        config: Configuration of actor running.
    """
    node_name = os.uname()[1]
    logger = logging.getLogger(f"[Actor runner {node_name}, process {process_index}]")

    # init actor tag
    tag = ActorTag(vector_env_size=config.actor.vector_env_size, node_name=node_name, process_index=process_index)
    logger.info(f"Init actor {tag}")

    # init local buffer
    local_buffer: Deque[SampleFromActor] = deque([], maxlen=config.local_buffer_size)

    # init actor
    actor = Actor(init_policy_param=init_policy_param, config=config, process_index=process_index)

    start_queue.get()  # block until kicked by main process
    logger.info('Start queue kicked')

    # initialize grpc client
    actor_manager_client = ActorManagerClient(url=config.actor_manager_url)
    replay_buffer_client = ReplayBufferClient(config=config)
    param_distributor_client = ParamDistributorClient(url=config.param_distributor_url)

    # regist actor
    actor_manager_client.regist_actor(tag)

    logger.info('Start exploration')
    step = 0
    try:
        while True:
            [local_buffer.append(s) for s in actor.step()]
            step += 1
            if step % config.send_sample_interval == 0:
                if len(local_buffer):
                    replay_buffer_client.append_sample(
                        SampleFromActor.concat([local_buffer.popleft() for _ in range(len(local_buffer))],
                                               np_defs=config.sample_from_actor_def))
            if step % config.update_param_interval == 0:
                param = param_distributor_client.get_param()
                if param: actor.update_model_param(param)
                actor.update_policy_param(
                    actor_manager_client.get_policy_param(actor_tag=tag, actor_status=actor.get_status()))
    except Exception as e:
        logger.error(f"Failed to run actor", exc_info=True)
        raise e
    finally:
        actor.finalize()
