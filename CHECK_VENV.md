# How to Check if Python Venv is Complete

## Quick Check

Run the check script:

```bash
./check-venv.sh
```

## What It Checks

The script verifies:

1. ✅ **Venv exists** - Checks if `backend-python-ml-v2/venv` directory exists
2. ✅ **Python executable** - Verifies Python binary is present and is a real file (not symlink)
3. ✅ **Site-packages** - Confirms site-packages directory exists
4. ✅ **All dependencies** - Checks all packages from `requirements.txt` are installed
5. ✅ **Critical packages** - Verifies essential packages (grpc, torch, transformers, etc.)

## Output

### ✅ Complete Venv
```
✅ VENV IS COMPLETE

All required packages are installed!
The venv is ready to be bundled into the AppImage.
```

### ⚠️ Incomplete Venv
```
⚠️  VENV HAS SOME MISSING PACKAGES

Missing packages: package1 package2 ...

To fix:
  rm -rf backend-python-ml-v2/venv
  ./prepare-python-venv.sh
```

### ❌ Critical Packages Missing
```
❌ VENV IS INCOMPLETE

Missing critical packages: grpc torch ...

To fix:
  rm -rf backend-python-ml-v2/venv
  ./prepare-python-venv.sh
```

## How to Fix Incomplete Venv

1. **Delete the old venv:**
   ```bash
   rm -rf backend-python-ml-v2/venv
   ```

2. **Rebuild the venv:**
   ```bash
   ./prepare-python-venv.sh
   ```

3. **Verify it's complete:**
   ```bash
   ./check-venv.sh
   ```

4. **Rebuild AppImage:**
   ```bash
   ./build-appimage.sh
   ```

## Automatic Check

The `build-appimage.sh` script automatically checks the venv before building. If it's incomplete, it will warn you and ask if you want to continue.

## Notes

- The script handles packages with different import names (e.g., `grpcio-tools` imports as `grpc_tools`)
- Critical packages are: `grpc`, `torch`, `transformers`, `numpy`, `av`, `faster_whisper`
- If critical packages are missing, the AppImage won't work properly

