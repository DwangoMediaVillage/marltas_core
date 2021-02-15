"""Implementation of evaluator for CNN-DQN."""
import logging
import time
from concurrent import futures
from queue import Empty, Queue
from typing import Callable

import grpc
import gym

from dqn.cnn.config import CNNConfigBase
from dqn.cnn.policy import Policy
from dqn.evaluator import (EvaluationRequest, EvaluationResult, EvaluatorBase,
                           EvaluatorServer)
from dqn.explorer import Explorer
from dqn.make_env import EnvConfig, make_env
from dqn.param_distributor import ParamDistributorClient
from dqn.proto_build import dqn_pb2_grpc
from dqn.utils import CustomMetrics, none_mean


class Evaluator(EvaluatorBase):
    """CNN evaluator.

    Args:
        config: CNN configuration.
        process_index: Index of process.
        make_env_func: Function to init an environment.
    """
    def __init__(self,
                 config: CNNConfigBase,
                 make_env_func: Callable[[EnvConfig], gym.Env] = make_env,
                 logger: logging.Logger = logging.getLogger(__name__)):
        super(Evaluator, self).__init__(env_config=config.env, make_env_func=make_env_func)
        self.n_eval = config.evaluator.n_eval
        self.logger = logger
        self.policy = Policy(config=config)

        self.custom_metric_keys = config.evaluator.custom_metric_keys
        self.custom_metric_types = config.evaluator.custom_metric_types

        self.explorer = Explorer(action_size=config.model.action_size,
                                 init_eps=config.evaluator.eps,
                                 init_beta=config.intrinsic_reward.reward_ratio,
                                 use_intrinsic_reward=False,
                                 use_ucb=False,
                                 apply_value_scaling=config.apply_value_scaling)

    def update_model_param(self, param: bytes) -> None:
        """Update model's parameter.

        Args:
            param: Byte expression of model parameter.
        """
        self.policy.update_model_param(param, only_online_model=True)

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate model.

        Args:
            request: Evaluation request object.

        Returns:
            evaluation_result: Result of evaluation(s).
        """
        self.logger.info(f"Start evaluation for {request}")
        reward_sum_mean = 0.0
        episode_len_mean = 0

        custom_metrics = []

        for _ in range(self.n_eval):
            reward_sum = 0.0
            episode_len = 0
            custom_metric = CustomMetrics(metric_keys=self.custom_metric_keys, metric_types=self.custom_metric_types)

            obs = self.env.reset()
            done = False
            while not done:
                prediction, _ = self.policy.infer([obs])
                q_e, _ = prediction.as_numpy_tuple()
                obs, reward, done, info = self.env.step(self.explorer.select_action(q_e[0]))
                reward_sum += reward
                episode_len += 1
                custom_metric.take_from_info(info)

            reward_sum_mean += reward_sum
            episode_len_mean += episode_len
            custom_metrics.append(custom_metric)

        # compute mean of custom metrics
        custom_metric_means = {}
        for k in self.custom_metric_keys:
            m = none_mean([c.as_dict().get(k) for c in custom_metrics])
            if m is not None: custom_metric_means[k] = m

        self.logger.info(f"End of evaluation for {request}")
        return EvaluationResult(walltime=time.time(),
                                global_step=request.global_step,
                                reward_sum=reward_sum_mean / self.n_eval,
                                episode_len=episode_len_mean / self.n_eval,
                                custom_metrics=custom_metric_means)


def run_evaluator_server(
    start_queue: Queue,
    config: CNNConfigBase,
    make_env_func: Callable[[int, int], gym.Env] = make_env,
    logger: logging.Logger = logging.getLogger(__name__)) -> None:
    """Start gRPC evaluator server.

    Args:
        start_queue: mp.Queue to sync gRPC client initializations.
        config: CNN configuration.
        make_env_fuc: Function to initialize environment.
    """
    eval_queue = Queue(maxsize=1)
    evaluator_server = EvaluatorServer(eval_queue)
    evaluator = Evaluator(config=config, make_env_func=make_env_func)

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        grpc_server = grpc.server(executor)
        dqn_pb2_grpc.add_EvaluatorServicer_to_server(evaluator_server, grpc_server)
        grpc_server.add_insecure_port(config.evaluator_url)
        grpc_server.start()

        logger.info(f"Evaluator server started {config.evaluator_url}")

        start_queue.get()  # block until maintained by main process
        param_distributor_client = ParamDistributorClient(url=config.param_distributor_url)
        logger.info('Init param distributor client')

        # run forever
        try:
            while True:
                try:
                    request = eval_queue.get(block=False)
                    param = param_distributor_client.get_param()
                    if param:
                        evaluator.update_model_param(param)
                    else:
                        logger.warning('Cannot fetch model parameter')
                    result = evaluator.evaluate(request)
                    evaluator_server.status.append_result(result)
                except Empty:
                    time.sleep(1)
                except Exception as e:
                    raise e
        except Exception as e:
            logger.error(f"Replay buffer server died {e}", exc_info=True)
            grpc_server.stop(0)
