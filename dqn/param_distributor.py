"""gRPC server of parameter distribution."""
import logging
import time
from concurrent import futures
from dataclasses import dataclass
from typing import Generator, Optional

import grpc

from dqn.proto_build import dqn_pb2, dqn_pb2_grpc
from dqn.utils import EventObject


@dataclass
class ParamDistributorStatus(EventObject):
    """Status of parameter distributor server to be logged.

    Attributes:
        n_get_param: Count of `get_param` call.
        n_update_param: Count of `update_param` call.
    """
    n_get_param: int
    n_update_param: int


class ParamDistributorClient:
    """gRPC client to sync parameter.

    Args:
        url: URL of server
        timeout_sec: Maximum seconds to wait launch of gRPC connection.
    """
    chunk_size = 3000000

    def __init__(self, url: str, timeout_sec: int = 10, logger: logging.Logger = logging.getLogger(__name__)):
        self.channel = grpc.insecure_channel(url)
        grpc.channel_ready_future(self.channel).result(timeout=timeout_sec)
        self.stub = dqn_pb2_grpc.ParamDistributorStub(self.channel)
        self.logger = logger

    def param_generator(self, param: bytes) -> Generator:
        """Yields parameter bytes into chunks.

        Args:
            param: Parameter data.

        Yields:
            bytes_data: chunk of `param` as `BytesData` object.
        """
        for i in range(0, len(param), self.chunk_size):
            yield dqn_pb2.BytesData(data=param[i:min(len(param), i + self.chunk_size)])

    def update_param(self, param: bytes) -> None:
        """Send param as bytes to server.

        Args:
            param: Parameter data.
        """
        self.stub.update_param(self.param_generator(param))

    def get_param(self) -> Optional[bytes]:
        """Fetch param as bytes from server.

        Returns:
            param: Parameter bytes data. In case the parameter distributor doesn't have any parameter, returns `None`.
        """
        try:
            bytes_data = b''
            for x in self.stub.get_param(dqn_pb2.Void()):
                bytes_data += x.data
            return bytes_data
        except grpc.RpcError as e:
            if e.code().name == 'NOT_FOUND':
                return None
            else:
                raise e

    def get_status(self) -> ParamDistributorStatus:
        """Get server status."""
        return ParamDistributorStatus.from_bytes(self.stub.get_status(dqn_pb2.Void()).data)


class ParamDistributorServer(dqn_pb2_grpc.ParamDistributorServicer):
    """grpc server to sync Q-network parameters."""

    chunk_size = 3000000

    def __init__(self):
        self.param_data: Optional[bytes] = None
        self.n_get_param = 0
        self.n_update_param = 0

    def get_param(self, request, context) -> Generator[dqn_pb2.BytesData, None, None]:
        """grpc method to serve cached parameter as bytes stream"""
        if self.param_data is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Not param data is set")
            yield dqn_pb2.BytesData()
        else:
            self.n_get_param += 1
            for i in range(0, len(self.param_data), self.chunk_size):
                yield dqn_pb2.BytesData(data=self.param_data[i:min(len(self.param_data), i + self.chunk_size)])

    def update_param(self, request_iterator, context) -> dqn_pb2.Void:
        """grpc method to renew parameted cache"""
        self.n_update_param += 1
        bytes_data = b''
        for req in request_iterator:
            bytes_data += req.data
        self.param_data = bytes_data

        return dqn_pb2.Void()

    def get_status(self, request, context) -> dqn_pb2.BytesData:
        """grpc method to server status of param distribution"""
        return dqn_pb2.BytesData(
            data=ParamDistributorStatus(n_get_param=self.n_get_param, n_update_param=self.n_update_param).to_bytes())


def run_param_distributor_server(url: str, logger: logging.Logger = logging.getLogger(__name__)) -> None:
    """Run `ParamDistributionServer`.

    Args:
        url: URL of gRPC server.
    """
    logger.info(f'start param distributor url = {url}')
    param_distributor_server = ParamDistributorServer()
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        grpc_server = grpc.server(executor)
        dqn_pb2_grpc.add_ParamDistributorServicer_to_server(param_distributor_server, grpc_server)
        grpc_server.add_insecure_port(url)
        grpc_server.start()

        logger.info(f"Param distributor server started {url}")

        try:
            while True:
                time.sleep(1)

        except Exception as e:
            logger.error(f"Replay buffer server died {e}", exc_info=True)
            grpc_server.stop(0)
