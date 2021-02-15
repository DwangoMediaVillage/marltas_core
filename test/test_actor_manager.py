import logging
import tempfile
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dqn.actor import ActorStatus
from dqn.actor_manager import (ActorManagerClient, ActorManagerConfig,
                               ActorManagerServer, ActorManagerStatus,
                               ActorTag, MetaPolicyParam)
from dqn.proto_build import dqn_pb2_grpc


def gen_dammy_actor_status():
    return ActorStatus(reward_sum=[0.1, 0.2, None],
                       epsilon=np.array([0.1, 0.2, 0.3]),
                       episode_len_mean=[None, None, None],
                       episode=1,
                       step=1234,
                       gamma=np.array([0.99, 0.99, 0.99]),
                       intrinsic_reward_sum=[None],
                       ucb_arm_index=[0, 1, 2, 0, 1, 1, 1])


def test_actor_manager_status():
    with tempfile.TemporaryDirectory() as log_dir:
        with SummaryWriter(log_dir=Path(log_dir)) as writer:
            ActorManagerStatus.from_actor_status([gen_dammy_actor_status()]).write_summary(writer=writer, global_step=1)
            ActorManagerStatus.from_actor_status([gen_dammy_actor_status()
                                                  for _ in range(3)]).write_summary(writer=writer, global_step=1)


def test_actor_manager(url='localhost:1111'):
    actor_manager_server = ActorManagerServer(gamma=0.997,
                                              config=ActorManagerConfig(),
                                              logger=logging.getLogger(__name__))
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        grpc_server = grpc.server(executor)
        dqn_pb2_grpc.add_ActorManagerServicer_to_server(actor_manager_server, grpc_server)
        grpc_server.add_insecure_port(url)
        grpc_server.start()
        try:
            client = ActorManagerClient(url)

            # regist 3 actors
            actor_tag = ActorTag(node_name='foo', process_index=0, vector_env_size=3)
            client.regist_actor(actor_tag)
            assert client.get_status(
            ).n_actor == 3  # can get actor manager status even if no actor status has been cached

            # can get policy param
            param = client.get_policy_param(actor_tag=actor_tag, actor_status=gen_dammy_actor_status())
            assert len(param.gamma) == 3
            assert len(param.epsilon) == 3

            # update meta policy param
            client.update_meta_policy_param(MetaPolicyParam(step=1))

            # get status
            status = client.get_status()
            assert status.episode == 1
            assert status.step == 1234
            assert np.array_equal(status.reward_sum, np.array([0.1, 0.2]))
            assert np.array_equal(status.epsilon, np.array([0.1, 0.2, 0.3]))
            assert np.array_equal(status.gamma, np.array([0.99, 0.99, 0.99]))
            assert np.array_equal(status.episode_len, np.array([]))
            assert np.array_equal(status.intrinsic_reward_sum, np.array([]))
            assert np.array_equal(status.ucb_arm_index, np.array([0, 1, 2, 0, 1, 1, 1]))
            assert status.n_actor == 3

        except Exception as e:
            grpc_server.stop(0)
            raise e
