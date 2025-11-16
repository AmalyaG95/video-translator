#!/bin/bash
set -e

echo "🐍 Preparing Python virtual environment for AppImage..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
  echo "❌ Error: package.json not found. Please run this script from the project root."
  exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "❌ Error: python3 not found. Please install Python 3.9+"
  exit 1
fi

# Check for build tools (needed for webrtcvad and other packages with C extensions)
echo ""
echo "🔧 Checking for build tools..."
if ! command -v gcc &> /dev/null; then
  echo "⚠️  Warning: gcc not found. Some packages may fail to build."
  echo "   Install with: sudo apt-get install build-essential (Debian/Ubuntu)"
  echo "   Or: sudo dnf install gcc gcc-c++ make (Fedora/RHEL)"
  echo ""
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
else
  echo "✓ Build tools found"
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python found: $(python3 --version)"

# Check Python version (need 3.9+)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
  echo "❌ Error: Python 3.9+ required, found $PYTHON_VERSION"
  exit 1
fi

# Create venv directory
VENV_DIR="backend-python-ml-v2/venv"
echo ""
echo "📦 Creating virtual environment..."

if [ -d "$VENV_DIR" ]; then
  echo "⚠️  Virtual environment already exists, removing old one..."
  rm -rf "$VENV_DIR"
fi

# Create venv with --copies flag for better portability (no symlinks to system Python)
# This makes the venv more relocatable for AppImage
echo "   Creating venv with --copies (for AppImage portability)..."
python3 -m venv --copies "$VENV_DIR"

# Verify that Python executable is a real file, not a symlink
if [ -L "$VENV_DIR/bin/python3" ]; then
  echo "⚠️  Warning: Python executable is still a symlink, converting to copy..."
  # Find the actual Python binary
  REAL_PYTHON=$(readlink -f "$VENV_DIR/bin/python3")
  if [ -f "$REAL_PYTHON" ]; then
    # Remove symlink and copy the actual file
    rm "$VENV_DIR/bin/python3"
    cp "$REAL_PYTHON" "$VENV_DIR/bin/python3"
    chmod +x "$VENV_DIR/bin/python3"
    echo "✓ Converted python3 symlink to copy"
  fi
fi

# Also check python3.x version and other Python executables
for PYTHON_BIN in "$VENV_DIR/bin/python"* "$VENV_DIR/bin/pip"*; do
  if [ -L "$PYTHON_BIN" ] && [ -f "$PYTHON_BIN" ]; then
    REAL_PYTHON=$(readlink -f "$PYTHON_BIN")
    if [ "$REAL_PYTHON" != "$PYTHON_BIN" ] && [ -f "$REAL_PYTHON" ]; then
      BIN_NAME=$(basename "$PYTHON_BIN")
      echo "   Converting $BIN_NAME from symlink to copy..."
      rm "$PYTHON_BIN"
      cp "$REAL_PYTHON" "$PYTHON_BIN"
      chmod +x "$PYTHON_BIN"
    fi
  fi
done

# Note: Python binaries have embedded paths, but they should still work
# as long as we don't set PYTHONHOME incorrectly
echo "✓ Venv Python executables prepared for portability"
echo "   Note: Python will auto-detect its standard library location"

# Activate venv and upgrade pip
echo "⬆️  Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "📥 Installing Python dependencies..."
echo "   This may take 5-10 minutes (downloading ML models and dependencies)..."

# Create a temporary requirements file without webrtcvad (install it separately if build tools available)
TEMP_REQUIREMENTS=$(mktemp)
grep -v "^webrtcvad" backend-python-ml-v2/requirements.txt > "$TEMP_REQUIREMENTS"

# Install main dependencies (this will fail if critical packages can't be installed)
if ! "$VENV_DIR/bin/pip" install -r "$TEMP_REQUIREMENTS"; then
  echo "❌ Error: Failed to install core dependencies"
  rm -f "$TEMP_REQUIREMENTS"
  exit 1
fi

# Try to install webrtcvad separately (it requires C compiler, but is optional)
echo ""
echo "📥 Installing webrtcvad (optional - requires C compiler)..."

# Create temporary directory for gcc symlink if needed
TEMP_BIN_DIR=""
CLEANUP_TEMP_BIN=""

# Check if gcc-11 exists, if not create a workaround
if ! command -v gcc-11 &> /dev/null && command -v gcc &> /dev/null; then
  GCC_PATH=$(command -v gcc)
  TEMP_BIN_DIR=$(mktemp -d)
  ln -s "$GCC_PATH" "$TEMP_BIN_DIR/gcc-11"
  export PATH="$TEMP_BIN_DIR:$PATH"
  CLEANUP_TEMP_BIN="$TEMP_BIN_DIR"
  echo "   Created temporary gcc-11 symlink (webrtcvad requires gcc-11)"
fi

# Try to install webrtcvad
WEBRTCVAD_OUTPUT=$("$VENV_DIR/bin/pip" install webrtcvad>=2.0.10 2>&1)
WEBRTCVAD_EXIT=$?

# Cleanup temp directory
if [ -n "$CLEANUP_TEMP_BIN" ] && [ -d "$CLEANUP_TEMP_BIN" ]; then
  rm -rf "$CLEANUP_TEMP_BIN"
  # Remove from PATH (though it doesn't matter after cleanup)
  export PATH=$(echo "$PATH" | sed "s|$CLEANUP_TEMP_BIN:||")
fi

if [ $WEBRTCVAD_EXIT -eq 0 ]; then
  echo "✓ webrtcvad installed successfully"
elif echo "$WEBRTCVAD_OUTPUT" | grep -q "gcc.*failed\|command.*failed\|No such file or directory"; then
  echo "⚠️  Warning: webrtcvad installation failed - C compiler issue"
  echo "   VAD preprocessing will be disabled (app will still work fine)"
  echo ""
  echo "   To fix (optional):"
  echo "   sudo ln -s $(command -v gcc 2>/dev/null || echo '/usr/bin/gcc') /usr/bin/gcc-11"
  echo "   Then rebuild the venv"
else
  echo "⚠️  Warning: webrtcvad installation failed"
  echo "   VAD preprocessing will be disabled (app will still work fine)"
  echo "$WEBRTCVAD_OUTPUT" | grep -i "error\|failed" | head -3
fi

rm -f "$TEMP_REQUIREMENTS"

# Verify installation
echo ""
echo "✅ Verifying installation..."
"$VENV_DIR/bin/python" -c "import grpc; import torch; import transformers; print('✓ Core dependencies installed')" || {
  echo "❌ Error: Dependency verification failed"
  exit 1
}

# Get venv size
VENV_SIZE=$(du -sh "$VENV_DIR" | cut -f1)
echo ""
echo "✅ Python virtual environment ready!"
echo "   Location: $VENV_DIR"
echo "   Size: $VENV_SIZE"
echo ""
echo "💡 This venv will be bundled into the AppImage"

