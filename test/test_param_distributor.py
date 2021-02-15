from concurrent import futures

import grpc

from dqn.param_distributor import (ParamDistributorClient,
                                   ParamDistributorServer)
from dqn.proto_build import dqn_pb2_grpc


def test_param_distributor():
    url = 'localhost:3333'
    param_distributor_server = ParamDistributorServer()

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        grpc_server = grpc.server(executor)
        dqn_pb2_grpc.add_ParamDistributorServicer_to_server(param_distributor_server, grpc_server)
        grpc_server.add_insecure_port(url)
        grpc_server.start()

        # init client
        client = ParamDistributorClient(url)

        # can get status
        stat = client.get_status()
        assert stat.n_get_param == 0 and stat.n_update_param == 0

        # can put / get param
        dammy_data = b'1234'
        client.update_param(dammy_data)
        assert param_distributor_server.param_data == dammy_data
        assert client.get_param() == dammy_data

        grpc_server.stop(0)
