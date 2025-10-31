#!/bin/bash

echo "🚀 Setting up Video Translator..."
echo "=========================================="

# Check if running on supported OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ Linux detected"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✅ macOS detected"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "✅ Windows detected"
else
    echo "⚠️  Unknown OS: $OSTYPE"
fi

# Check Python version
echo "🐍 Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo "✅ Python $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

# Check Node.js version
echo "📦 Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js $NODE_VERSION found"
else
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check FFmpeg
echo "🎬 Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)
    echo "✅ FFmpeg $FFMPEG_VERSION found"
else
    echo "❌ FFmpeg not found. Please install FFmpeg"
    echo "   Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   macOS: brew install ffmpeg"
    echo "   Windows: Download from https://ffmpeg.org/"
    exit 1
fi

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
cd backend
/usr/bin/python3 -m pip install -r requirements.txt
cd ..

# Create necessary directories in .data folder to consolidate data
echo "📁 Creating directories..."
mkdir -p .data/artifacts .data/temp_work .data/uploads .data/logs public backend/static

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# Video Translator Configuration
NODE_ENV=development
PYTHONPATH=./backend
LOG_LEVEL=INFO
MAX_FILE_SIZE_GB=50
CHUNK_DURATION=30
WHISPER_MODEL_SIZE=base
ENABLE_GPU=true
EOF
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run in development:"
echo "  npm run dev"
echo ""
echo "To build for production:"
echo "  npm run build"
echo "  npm run dist"
echo ""
echo "To run with Docker:"
echo "  docker-compose up --build"
echo ""
echo "🎉 Video Translator is ready!"
