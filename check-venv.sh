#!/bin/bash
set -e

echo "🔍 Checking Python virtual environment completeness..."
echo ""

VENV_DIR="backend-python-ml-v2/venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
  echo "❌ Virtual environment not found at: $VENV_DIR"
  echo ""
  echo "To create it, run:"
  echo "  ./prepare-python-venv.sh"
  exit 1
fi

echo "✓ Virtual environment found: $VENV_DIR"
echo ""

# Check Python executable
PYTHON_BIN="$VENV_DIR/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
  echo "❌ Python executable not found: $PYTHON_BIN"
  exit 1
fi

# Check if it's a symlink or real file
if [ -L "$PYTHON_BIN" ]; then
  echo "⚠️  Python executable is a symlink (may cause issues in AppImage)"
  REAL_PYTHON=$(readlink -f "$PYTHON_BIN")
  echo "   Points to: $REAL_PYTHON"
else
  echo "✓ Python executable is a real file (good for AppImage)"
fi

# Get Python version
PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1)
echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Check site-packages
VENV_LIB="$VENV_DIR/lib"
if [ ! -d "$VENV_LIB" ]; then
  echo "❌ venv lib directory not found: $VENV_LIB"
  exit 1
fi

PYTHON_DIRS=$(find "$VENV_LIB" -maxdepth 1 -type d -name "python*" | head -1)
if [ -z "$PYTHON_DIRS" ]; then
  echo "❌ No python version directory found in: $VENV_LIB"
  exit 1
fi

SITE_PACKAGES="$PYTHON_DIRS/site-packages"
if [ ! -d "$SITE_PACKAGES" ]; then
  echo "❌ site-packages directory not found: $SITE_PACKAGES"
  exit 1
fi

echo "✓ Site-packages found: $SITE_PACKAGES"
echo ""

# Check required packages from requirements.txt
echo "📦 Checking required packages..."
echo ""

REQUIREMENTS_FILE="backend-python-ml-v2/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "⚠️  Requirements file not found: $REQUIREMENTS_FILE"
  exit 1
fi

# Read requirements and check each one
MISSING_PACKAGES=()
CHECKED=0
PASSED=0

while IFS= read -r line || [ -n "$line" ]; do
  # Skip comments and empty lines
  if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "$line" ]]; then
    continue
  fi
  
  # Extract package name (handle version specifiers and extras like uvicorn[standard])
  PACKAGE_NAME=$(echo "$line" | sed 's/\[.*\]//' | sed 's/[<>=!].*//' | xargs)
  
  if [ -z "$PACKAGE_NAME" ]; then
    continue
  fi
  
  CHECKED=$((CHECKED + 1))
  
  # Map package names to import names (some packages have different import names)
  IMPORT_NAME="$PACKAGE_NAME"
  case "$PACKAGE_NAME" in
    grpcio-tools)
      IMPORT_NAME="grpc_tools"
      ;;
    faster-whisper)
      IMPORT_NAME="faster_whisper"
      ;;
    edge-tts)
      IMPORT_NAME="edge_tts"
      ;;
    python-Levenshtein)
      IMPORT_NAME="Levenshtein"
      ;;
    uvicorn*)
      IMPORT_NAME="uvicorn"
      ;;
    opentelemetry-*)
      # opentelemetry packages use dots: opentelemetry-api -> opentelemetry.api
      IMPORT_NAME=$(echo "$PACKAGE_NAME" | sed 's/-/./g')
      ;;
    pydantic-settings)
      IMPORT_NAME="pydantic_settings"
      ;;
    grpcio-reflection)
      IMPORT_NAME="grpc_reflection"
      ;;
  esac
  
  # Check if package is installed by trying to import it
  if "$PYTHON_BIN" -c "import $IMPORT_NAME" 2>/dev/null; then
    PASSED=$((PASSED + 1))
    if [ "$IMPORT_NAME" != "$PACKAGE_NAME" ]; then
      echo "  ✓ $PACKAGE_NAME (imports as $IMPORT_NAME)"
    else
      echo "  ✓ $PACKAGE_NAME"
    fi
  else
    # Try checking if package directory exists in site-packages
    # Convert package name to directory name (hyphens to underscores, dots to underscores)
    DIR_NAME=$(echo "$PACKAGE_NAME" | sed 's/-/_/g' | sed 's/\./_/g')
    if find "$SITE_PACKAGES" -maxdepth 1 \( -type d -name "${PACKAGE_NAME}*" -o -name "${DIR_NAME}*" -o -name "${PACKAGE_NAME}" -o -name "${DIR_NAME}" \) 2>/dev/null | grep -q .; then
      PASSED=$((PASSED + 1))
      echo "  ✓ $PACKAGE_NAME (found in site-packages)"
    else
      MISSING_PACKAGES+=("$PACKAGE_NAME")
      echo "  ❌ $PACKAGE_NAME (MISSING)"
    fi
  fi
done < "$REQUIREMENTS_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Summary:"
echo "   Checked: $CHECKED packages"
echo "   Found: $PASSED packages"
echo "   Missing: ${#MISSING_PACKAGES[@]} packages"
echo ""

# Check critical packages explicitly
echo "🔍 Verifying critical packages..."
CRITICAL_PACKAGES=("grpc" "torch" "transformers" "numpy" "av" "faster_whisper")
CRITICAL_MISSING=()

for PKG in "${CRITICAL_PACKAGES[@]}"; do
  if "$PYTHON_BIN" -c "import $PKG" 2>/dev/null; then
    echo "  ✓ $PKG"
  else
    echo "  ❌ $PKG (CRITICAL - MISSING)"
    CRITICAL_MISSING+=("$PKG")
  fi
done

echo ""

# Final verdict
if [ ${#CRITICAL_MISSING[@]} -gt 0 ]; then
  echo "❌ VENV IS INCOMPLETE"
  echo ""
  echo "Missing critical packages: ${CRITICAL_MISSING[*]}"
  echo ""
  echo "To fix, rebuild the venv:"
  echo "  rm -rf $VENV_DIR"
  echo "  ./prepare-python-venv.sh"
  exit 1
elif [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
  echo "⚠️  VENV HAS SOME MISSING PACKAGES"
  echo ""
  echo "Missing packages: ${MISSING_PACKAGES[*]}"
  echo ""
  echo "You can try to install them:"
  echo "  $PYTHON_BIN -m pip install ${MISSING_PACKAGES[*]}"
  echo ""
  echo "Or rebuild the venv:"
  echo "  rm -rf $VENV_DIR"
  echo "  ./prepare-python-venv.sh"
  exit 1
else
  echo "✅ VENV IS COMPLETE"
  echo ""
  echo "All required packages are installed!"
  echo "The venv is ready to be bundled into the AppImage."
  exit 0
fi

