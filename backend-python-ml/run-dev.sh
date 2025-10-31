#!/bin/bash
# Development script to run Python ML service with venv
# Usage: ./run-dev.sh

set -e

cd "$(dirname "$0")"

# Activate venv
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements-dev.txt"
    exit 1
fi

source venv/bin/activate

# Set environment variables
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export GRPC_PORT=50051

echo "🚀 Starting Python ML Service (Development Mode)"
echo "📍 gRPC server will listen on: localhost:$GRPC_PORT"
echo "🐍 Python: $(python --version)"
echo "📦 edge-tts version: $(pip show edge-tts | grep Version || echo 'Not found')"
echo ""

# Run the service
python src/main.py


