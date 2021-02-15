"""gRPC server and client to manage actors."""
import collections
import itertools
import logging
import time
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type, TypeVar

import grpc
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dqn.actor import ActorStatus
from dqn.policy import PolicyParam
from dqn.proto_build import dqn_pb2, dqn_pb2_grpc
from dqn.utils import ConfigBase, EpsilonSampler, EventObject

T = TypeVar('T', bound='Parent')


@dataclass
class MetaPolicyParam(EventObject):
    """Parameter to decide Epsilon-greedy parameters."""
    step: int


@dataclass
class ActorTag(EventObject):
    """Tag to identify actor process."""
    vector_env_size: int
    node_name: str
    process_index: int

    def __hash__(self) -> int:
        return hash((self.node_name, self.process_index, self.vector_env_size))


@dataclass
class ActorManagerConfig(ConfigBase):
    """Configuration of actor management.

    Args:
        init_eps_base: Initial `epsilon-base`.
        final_eps_base: Final `epsilon-base`.
        init_alpha: Initial `alpha`.
        final_alpha: Final `alpha`.
        final_eps_alpha_step: Final step for scheduling `eps_base` and `alpha`.
        min_eps: Minimum value of sampled epsilon.
    """
    init_eps_base: float = 0.7
    final_eps_base: float = 0.1
    init_alpha: float = 7.0
    final_alpha: float = 3.0
    final_eps_alpha_step: int = 30000
    min_eps: float = 0.02


@dataclass
class ActorManagerStatus(EventObject):
    """Status of actor manager. Most of attributes are accumulated from actors.

    Attributes:
        episode: Number of episodes since training start.
        step: Number of environment steps since training start
        reward_sum: Moving average of non-discounted reward sum per episode.
        epsilon: Epsilon of psilon-greedy parameters.
        gamma: Discounted factors.
        episode_len: Moving average of episode length
        intrinsic_reward_sum: Moving average of non-discounted intrinsic reward sum per episode.
        ucb_arm_index: Arm indices of UCB mete contoller selected in each actor.
        n_actor: Number of registered actors.
    """
    episode: int
    step: int
    reward_sum: np.ndarray
    epsilon: np.ndarray
    gamma: np.ndarray
    episode_len: np.ndarray
    intrinsic_reward_sum: np.ndarray
    ucb_arm_index: List[int]
    n_actor: int

    @classmethod
    def from_actor_status(cls: Type[T], status: List[ActorStatus], n_actor: int = 1) -> T:
        reduce_optional_float = lambda x: np.array(list(filter(None, itertools.chain(*x))))
        return cls(episode=sum([s.episode for s in status]),
                   step=sum([s.step for s in status]),
                   reward_sum=reduce_optional_float([s.reward_sum for s in status]),
                   epsilon=np.concatenate([s.epsilon for s in status]) if len(status) else np.array([]),
                   gamma=np.concatenate([s.gamma for s in status]) if len(status) else np.array([]),
                   episode_len=reduce_optional_float([s.episode_len_mean for s in status]),
                   intrinsic_reward_sum=reduce_optional_float([s.intrinsic_reward_sum for s in status]),
                   ucb_arm_index=list(itertools.chain(*[s.ucb_arm_index for s in status if s.ucb_arm_index])),
                   n_actor=n_actor)

    def write_summary(self, writer: SummaryWriter, global_step: int, namespace: Optional[Path] = None) -> None:
        """Write status as tensorboard event file.

        args:
            writer: Writer object.
            global_step: Number of global step
            namespace: Namepace of event data.
        """
        if namespace is None: namespace = Path(type(self).__name__)

        writer.add_scalar(str(namespace / 'episode'), self.episode, global_step)
        writer.add_scalar(str(namespace / 'step'), self.step, global_step)

        if len(self.reward_sum):
            writer.add_histogram(str(namespace / 'reward_sum'), self.reward_sum, global_step)
            writer.add_scalar(str(namespace / 'reward_sum_mean'), self.reward_sum.mean(), global_step)

        if len(self.epsilon):
            writer.add_histogram(str(namespace / 'epsilon'), self.epsilon, global_step)
            writer.add_scalar(str(namespace / 'epsilon_mean'), self.epsilon.mean(), global_step)

        if len(self.gamma):
            writer.add_histogram(str(namespace / 'gamma'), self.gamma, global_step)
            writer.add_scalar(str(namespace / 'gamma_mean'), self.gamma.mean(), global_step)

        if len(self.episode_len):
            writer.add_histogram(str(namespace / 'episode_len'), self.episode_len, global_step)
            writer.add_scalar(str(namespace / 'episode_len_mean'), self.episode_len.mean(), global_step)

        if len(self.intrinsic_reward_sum):
            writer.add_histogram(str(namespace / 'intrinsic_reward_sum'), self.intrinsic_reward_sum, global_step)
            writer.add_scalar(str(namespace / 'intrinsic_reward_sum_mean'), self.intrinsic_reward_sum.mean(),
                              global_step)

        if len(self.ucb_arm_index):
            writer.add_histogram(str(namespace / 'ucb_arm_index'), np.array(self.ucb_arm_index), global_step)
            writer.add_scalar(str(namespace / 'ucb_arm_index_most_selected'),
                              int(collections.Counter(self.ucb_arm_index).most_common(1)[0][0]), global_step)

        writer.add_scalar(str(namespace / 'n_actor'), self.n_actor, global_step)


class ActorManagerClient:
    """gRPC client of actor manager.

    Args:
        url: URL of server.
        timeout_sec: Timeout seconds for initializing connection.
    """
    def __init__(self, url: str, timeout_sec: int = 10):
        self.channel = grpc.insecure_channel(url)
        grpc.channel_ready_future(self.channel).result(timeout=timeout_sec)
        self.stub = dqn_pb2_grpc.ActorManagerStub(self.channel)

    def regist_actor(self, actor_tag: ActorTag) -> None:
        """Regist `vector_env_size` envs on a process named `process_index` working on `nodename`.
        Args:
            actor_tag: Information of actor.
        """
        self.stub.regist_actor(dqn_pb2.BytesData(data=actor_tag.to_bytes()))

    def get_policy_param(self, actor_tag: ActorTag, actor_status: ActorStatus) -> PolicyParam:
        """Fetch parameter of policy.
        Args:
            actor_tag: Information of actor.
            actor_status: Status of actor subprocess
        Returns:
            policy_param: New parameter
        """
        return PolicyParam.from_bytes(
            self.stub.get_policy_param(
                dqn_pb2.PolicyParamRequest(tag=actor_tag.to_bytes(), status=actor_status.to_bytes())).data)

    def update_meta_policy_param(self, meta_policy_param: MetaPolicyParam) -> None:
        """Update new actor manager's parameter.
        Args:
            meta_manager_param: parameter to be set.
        """
        self.stub.update_meta_policy_param(dqn_pb2.BytesData(data=meta_policy_param.to_bytes()))

    def get_status(self) -> ActorManagerStatus:
        """Returns status of actor manager server"""
        status = self.stub.get_status(dqn_pb2.Void())
        return ActorManagerStatus.from_bytes(status.data)


class ActorManagerServer(dqn_pb2_grpc.ActorManagerServicer):
    """grpc server to manage actors on multiple nodes.

    Args:
        config: Actor manager config.
        gamma: Discount factor of actors.
    """
    def __init__(self, gamma: float, config: ActorManagerConfig, logger: logging.Logger):
        self.logger = logger
        self.actor_index = {}
        self.actor_tag_cache = {}
        self.actor_status_cache = {}
        self.n_actor = 0
        self.gamma = gamma
        self.meta_policy_param = MetaPolicyParam(step=0)
        self.epsilon_sampler = EpsilonSampler(init_eps_base=config.init_eps_base,
                                              final_eps_base=config.final_eps_base,
                                              init_alpha=config.init_alpha,
                                              final_alpha=config.final_alpha,
                                              final_step=config.final_eps_alpha_step,
                                              min_eps=config.min_eps)

    def regist_actor(self, request, context) -> dqn_pb2.Void:
        """gRPC method to regist a new actor."""
        actor_tag = ActorTag.from_bytes(request.data)
        if actor_tag not in self.actor_tag_cache:
            # assign new actor index
            self.logger.info(f"Regist actor {actor_tag}")
            self.actor_index[actor_tag] = np.arange(start=self.n_actor, stop=self.n_actor + actor_tag.vector_env_size)
            self.n_actor += actor_tag.vector_env_size
        else:
            self.logger.warning(f"actor {actor_tag} has been already registered")
        return dqn_pb2.Void()

    def get_policy_param(self, request, context) -> dqn_pb2.BytesData:
        """Returns new actor's parameters."""
        tag = ActorTag.from_bytes(request.tag)
        assert tag in self.actor_index
        index = self.actor_index[tag]

        status = ActorStatus.from_bytes(request.status)
        self.actor_status_cache[tag] = status

        # fixed gamma
        gamma = np.ones(len(index)) * self.gamma

        # sample epsilons
        epsilon = self.epsilon_sampler(step=self.meta_policy_param.step, index=index, max_index=self.n_actor)

        self.logger.debug(f"Send actor_param {gamma}, {epsilon} to {tag}")
        return dqn_pb2.BytesData(data=PolicyParam(gamma=gamma, epsilon=epsilon).to_bytes())

    def update_meta_policy_param(self, request, context) -> dqn_pb2.Void:
        """Set new meta policy parameter."""
        self.meta_policy_param = MetaPolicyParam.from_bytes(request.data)
        return dqn_pb2.Void()

    def get_status(self, request, context) -> dqn_pb2.BytesData:
        """Returns status of actor manager (also accumulated status of actors)."""
        return dqn_pb2.BytesData(
            data=ActorManagerStatus.from_actor_status(list(self.actor_status_cache.values()), self.n_actor).to_bytes())


def run_actor_manager_server(url: str,
                             gamma: float,
                             config: ActorManagerConfig,
                             logger: logging.Logger = logging.getLogger(__name__)) -> None:
    """Run actor management gRPC server forever.

    Args:
        url: URL of server.
        gamma: discount factor.
        config: Actor manager config.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Run actor manager server url = {url}')

    actor_manager_server = ActorManagerServer(gamma=gamma, config=config, logger=logger)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        grpc_server = grpc.server(executor)
        dqn_pb2_grpc.add_ActorManagerServicer_to_server(actor_manager_server, grpc_server)
        grpc_server.add_insecure_port(url)
        grpc_server.start()

        logger.info(f"Actor manager started {url}")
        try:
            while True:
                time.sleep(1)

        except Exception as e:
            logger.error(f"Replay buffer server died {e}", exc_info=True)

        finally:
            grpc_server.stop(0)
