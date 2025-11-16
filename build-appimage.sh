#!/bin/bash
set -e

echo "📦 Building Standalone Video Translator AppImage..."
echo "This will create an AppImage that users can double-click to run!"
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the project root."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Warning: python3 not found. Users will need Python 3.9+ installed."
else
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✓ Python found: $PYTHON_VERSION"
fi

# Install root dependencies
echo ""
echo "📥 Installing root dependencies..."
npm install

# Install frontend dependencies
echo "📥 Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
cd ..

# Build frontend (Next.js production build)
echo ""
echo "🏗️  Building frontend (Next.js production)..."
cd frontend
npm run build
if [ ! -d ".next" ]; then
    echo "❌ Error: Frontend build failed - '.next' directory not found"
    exit 1
fi
echo "✓ Frontend built successfully"
cd ..

# Build NestJS backend
if [ -d "backend-nestjs" ]; then
    echo ""
    echo "🏗️  Building NestJS backend..."
    cd backend-nestjs
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    npm run build
    if [ ! -d "dist" ]; then
        echo "❌ Error: NestJS build failed - 'dist' directory not found"
        exit 1
    fi
    echo "✓ NestJS backend built successfully"
    cd ..
fi

# Prepare Python virtual environment
if [ -d "backend-python-ml-v2" ]; then
    echo ""
    echo "🐍 Preparing Python virtual environment..."
    if [ ! -d "backend-python-ml-v2/venv" ]; then
        echo "  Creating Python venv with all dependencies..."
        ./prepare-python-venv.sh
    else
        echo "  ✓ Python venv already exists"
        echo "  Checking if venv is complete..."
        if ./check-venv.sh > /dev/null 2>&1; then
            echo "  ✓ Venv is complete and ready"
        else
            echo "  ⚠️  Venv appears incomplete, checking details..."
            ./check-venv.sh || {
                echo ""
                echo "  The venv is missing some packages."
                echo "  To fix, delete and recreate:"
                echo "    rm -rf backend-python-ml-v2/venv"
                echo "    ./prepare-python-venv.sh"
                echo ""
                read -p "Continue anyway? (y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            }
        fi
    fi
fi

# Build AppImage
echo ""
echo "📦 Creating standalone AppImage..."
echo "   This may take several minutes..."
npm run build:electron -- --linux

echo ""
echo "✅ Build complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 AppImage location:"
ls -lh dist/*.AppImage 2>/dev/null || echo "⚠️  No AppImage found in dist/ directory"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 To test the AppImage:"
echo "   chmod +x dist/Video\\ Translator-*.AppImage"
echo "   ./dist/Video\\ Translator-*.AppImage"
echo ""
echo "📋 User Requirements:"
echo "   - Linux (Ubuntu 20.04+, Debian 11+, etc.)"
echo "   - FUSE (usually pre-installed)"
echo "   - NO Python installation needed! (bundled in AppImage)"
echo ""
echo "💡 Users can double-click the AppImage to run!"
echo "💡 All Python dependencies are bundled - completely standalone!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

