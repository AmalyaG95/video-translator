#!/bin/bash
set -e

echo "🔧 Initializing Python ML Service..."

# Always regenerate gRPC Python files to ensure version compatibility
echo "📦 Generating gRPC Python files..."
python -m grpc_tools.protoc \
    -I./src/proto \
    --python_out=./src/proto \
    --grpc_python_out=./src/proto \
    ./src/proto/translation.proto

# Fix the import statement in generated gRPC file
echo "🔧 Fixing gRPC imports..."
sed -i 's/^import translation_pb2/from . import translation_pb2/' ./src/proto/translation_pb2_grpc.py

echo "✅ Python ML Service initialized successfully"
echo "🚀 Starting gRPC server..."

# Start the main application
exec python src/main.py

