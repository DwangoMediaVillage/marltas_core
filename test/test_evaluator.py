import tempfile
from concurrent import futures
from pathlib import Path
from queue import Queue

import grpc
from torch.utils.tensorboard import SummaryWriter

from dqn.evaluator import (EvaluationRequest, EvaluationResult,
                           EvaluatorClient, EvaluatorServer, EvaluatorStatus)
from dqn.proto_build import dqn_pb2_grpc


def test_evaluator_status():
    stat = EvaluatorStatus()
    stat.append_result(
        EvaluationResult(walltime=0.123,
                         global_step=123,
                         reward_sum=0.1,
                         episode_len=10,
                         custom_metrics={'foo': 1234.5}))

    with tempfile.TemporaryDirectory() as log_dir:
        with SummaryWriter(log_dir=Path(log_dir)) as writer:
            stat.write_summary(writer, 0)


def test_evaluator():
    url = 'localhost:2222'
    eval_queue = Queue()
    eval_server = EvaluatorServer(eval_queue)
    with futures.ThreadPoolExecutor(max_workers=1) as exec:
        grpc_server = grpc.server(exec)
        dqn_pb2_grpc.add_EvaluatorServicer_to_server(eval_server, grpc_server)
        grpc_server.add_insecure_port(url)
        grpc_server.start()

        # init client
        client = EvaluatorClient(url)

        # can get stat
        stat = client.get_status()
        assert len(stat.walltime) + len(stat.global_step) + len(stat.reward_sum) + len(stat.episode_len) == 0

        # can request evaluation
        client.request_evaluation(EvaluationRequest(global_step=0))

        # finalize server
        grpc_server.stop(0)
