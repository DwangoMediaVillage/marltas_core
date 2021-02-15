"""Evaluating policy network with epsilon-greedy"""
import multiprocessing as mp
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from queue import Queue
from typing import Callable, Deque, Optional

import grpc
import gym
from torch.utils.tensorboard import SummaryWriter

from dqn.make_env import EnvConfig, make_env
from dqn.proto_build import dqn_pb2, dqn_pb2_grpc
from dqn.utils import EventObject


@dataclass
class EvaluationRequest(EventObject):
    """Request to start new evaluation.

    Attributes:
        global_step: Global step at when an evaluation is requested.
        walltime: UNIX time at when an evaluation is requested.
    """
    global_step: int
    walltime: float = field(default_factory=lambda: time.time())


@dataclass
class EvaluationResult(EventObject):
    """Result of evaluation.

    Attributes:
        walltime: UNIX time at when evaluation is requested.
        global_step: Global step at when an evaluation is requested.
        reward_sum: Average of episode reward sum.
        episode_len: Average of episode step length.
    """
    walltime: float
    global_step: int
    reward_sum: float
    episode_len: float
    custom_metrics: dict = field(default_factory=lambda: {})


@dataclass
class EvaluatorStatus(EventObject):
    """FIFO queue of `EvaluationResult`.

    Args:
        walltime: UNIX time at when evaluation is requested.
        global_step: Global step at when an evaluation is requested.
        reward_sum: Average of episode reward sum.
        episode_len: Average of episode step length.

    """
    walltime: Deque[float] = field(default_factory=lambda: deque([], maxlen=10))
    global_step: Deque[int] = field(default_factory=lambda: deque([], maxlen=10))
    reward_sum: Deque[float] = field(default_factory=lambda: deque([], maxlen=10))
    episode_len: Deque[float] = field(default_factory=lambda: deque([], maxlen=10))
    custom_metrics: Deque[dict] = field(default_factory=lambda: deque([], maxlen=10))

    def write_summary(self, writer: SummaryWriter, _global_step: int, namespace: Optional[Path] = None) -> None:
        """Write into tfevent file.

        Args:
            writer: SummaryWriter object
            _global_step: Global step of the written event (not used).
            namespace: Parent name of scalar event tags.
        """
        if namespace is None: namespace = Path(type(self).__name__)

        # write diff
        if len(self.global_step) == 0: return
        for i in range(len(self.walltime)):
            walltime = self.walltime[i]
            global_step = self.global_step[i]
            writer.add_scalar(str(namespace / 'reward_sum'), self.reward_sum[i], global_step, walltime)
            writer.add_scalar(str(namespace / 'episode_len'), self.episode_len[i], global_step, walltime)

            if self.custom_metrics[i] is not None:
                for key, val in self.custom_metrics[i].items():
                    writer.add_scalar(str(namespace / key), val, global_step, walltime)

    def append_result(self, result: EvaluationResult) -> None:
        """Append `EvaluationResult` to buffer.

        Args:
            result: EvaluationResult to be added.
        """
        self.walltime.append(result.walltime)
        self.global_step.append(result.global_step)
        self.reward_sum.append(result.reward_sum)
        self.episode_len.append(result.episode_len)
        self.custom_metrics.append(result.custom_metrics)

    def clear(self) -> None:
        """Clear all the fields."""
        self.walltime.clear()
        self.global_step.clear()
        self.reward_sum.clear()
        self.episode_len.clear()
        self.custom_metrics.clear()


class EvaluatorBase:
    """Evaluator.

    Args:
        process_index: Process identifier for using multi evaluators.
        make_env_func: Function to init an openAI gym environment.
    """
    def __init__(self, env_config: EnvConfig, make_env_func: Callable[[EnvConfig, int, int], gym.Env] = make_env):
        self.env = make_env_func(env_config)

    def update_model_param(self, param: bytes) -> None:
        """Update Q-Network model parameter.

        Args:
            param: Bytes expression of model parameters.
        """
        raise NotImplementedError

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Do evaluation.

        Args:
            request: Evaluation request object.
        """
        raise NotImplementedError


class EvaluatorClient:
    """gRPC client of `EvaluatorServer`.

    Args:
        url: URL of gRPC evaluator server.
        timeout_sec: Seconds to wait an launch of gRPC connection.
    """
    def __init__(self, url: str, timeout_sec: int = 10):
        self.channel = grpc.insecure_channel(url)
        grpc.channel_ready_future(self.channel).result(timeout=timeout_sec)
        self.stub = dqn_pb2_grpc.EvaluatorStub(self.channel)

    def request_evaluation(self, request: EvaluationRequest) -> None:
        """Send an evaluation request to server.

        Args:
            request: Evaluation request object to be sent to server.
        """
        self.stub.request_evaluate(dqn_pb2.BytesData(data=request.to_bytes()))

    def get_status(self) -> EvaluatorStatus:
        """Fetch status of evaluator.

        Returns:
            EvaluatorStatus (FIFO queue of evaluation results).
        """
        return EvaluatorStatus.from_bytes(self.stub.get_status(dqn_pb2.Void()).data)


class EvaluatorServer(dqn_pb2_grpc.EvaluatorServicer):
    """grpc server to evaluate q-network by running episode(s)."""
    def __init__(self, eval_queue: Queue):
        self.eval_queue = eval_queue
        self.status = EvaluatorStatus()

    def request_evaluate(self, request, context) -> dqn_pb2.Void:
        """grpc method to receive request to start evaluation."""
        request = EvaluationRequest.from_bytes(request.data)
        self.eval_queue.put(request)  # async kick evaluation
        return dqn_pb2.Void()

    def get_status(self, request, context) -> dqn_pb2.BytesData:
        """grpc method to server status of evaluator"""
        data = self.status.to_bytes()
        self.status.clear()
        return dqn_pb2.BytesData(data=data)


class EvaluatorServerRunner:
    """Helper class to run evaluator server in a subprocess.
    Created process should take mp.Queue object which will be invoked only at once, after all the gRPC servers are ready.

    Args:
        run_evaluator_server_func: Function to run evaluator server.
    """
    def __init__(self, run_evaluator_server_func: Callable[[
        Queue,
    ], None]):
        self.start_queue = mp.Queue(maxsize=1)
        self.p = mp.Process(target=partial(run_evaluator_server_func, self.start_queue), daemon=True)
        self.p.start()

    @property
    def workers_alive(self) -> bool:
        """Returns whether the subprocess is alive or not."""
        return self.p.is_alive()

    def start(self) -> None:
        """Kick mp.Queue in subprocess."""
        self.start_queue.put(True)

    def finalize(self) -> None:
        """Close subprocess."""
        self.p.terminate()
        self.p.join()
