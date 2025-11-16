#!/bin/bash
# Generate gRPC proto files
# This script should be run in the Docker container or with grpc-tools installed

set -e

PROTO_DIR="src/proto"
OUTPUT_DIR="src/proto"

echo "Generating gRPC proto files..."

python3 -m grpc_tools.protoc \
    -I./${PROTO_DIR} \
    --python_out=./${OUTPUT_DIR} \
    --grpc_python_out=./${OUTPUT_DIR} \
    --pyi_out=./${OUTPUT_DIR} \
    ./${PROTO_DIR}/translation.proto

echo "Proto files generated successfully!"



