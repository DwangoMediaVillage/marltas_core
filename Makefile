SRC=./proto
DST=./dqn/proto_build

all:
	mkdir -p $(DST)
	python -m grpc_tools.protoc -I $(SRC) --python_out=$(DST) --grpc_python_out=$(DST) $(SRC)/dqn.proto
	sed -i -e 's/import\ dqn_pb2/import\ dqn.proto_build.dqn_pb2/g' $(DST)/dqn_pb2_grpc.py
	touch $(DST)/__init__.py

clean:
	rm -rf $(DST)
