#!/bin/bash
# Docker entrypoint script
# Generates proto files and starts the service

set -e

echo "Starting video translation ML service v2..."

# Generate proto files if they don't exist or if proto file is newer
if [ ! -f "src/proto/translation_pb2.py" ] || [ "src/proto/translation.proto" -nt "src/proto/translation_pb2.py" ]; then
    echo "Generating proto files..."
    python3 -m grpc_tools.protoc \
        -I./src/proto \
        --python_out=./src/proto \
        --grpc_python_out=./src/proto \
        --pyi_out=./src/proto \
        ./src/proto/translation.proto
    echo "Proto files generated."
fi

# Start the service
exec python3 -m src.main


