#!/bin/bash
python -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. dataset_feed.proto