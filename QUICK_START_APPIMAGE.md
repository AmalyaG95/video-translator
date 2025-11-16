# Quick Start: Build Standalone AppImage

## One Command Build

```bash
./build-appimage.sh
```

This will:
1. ✅ Install all dependencies
2. ✅ Build frontend (Next.js)
3. ✅ Build NestJS backend
4. ✅ Bundle Python ML service (no build needed - Python is interpreted)
5. ✅ Create standalone AppImage

## Output

```
dist/Video Translator-1.0.0-x86_64.AppImage
```

## Test It

```bash
chmod +x "dist/Video Translator-1.0.0-x86_64.AppImage"
./dist/Video\ Translator-1.0.0-x86_64.AppImage
```

## Share It

Users just need to:
1. Download the AppImage
2. `chmod +x Video\ Translator-*.AppImage`
3. Double-click to run!

**Requirements:** Linux + Python 3.9+ (that's it!)

## What Happens When Users Run It

1. AppImage mounts
2. Electron starts
3. **Automatically starts:**
   - Python ML service (AI translation) - generates proto files if needed
   - NestJS API (backend)
   - Next.js frontend (UI)
4. Window opens - ready to use!
5. On close - everything stops automatically

No Docker, no setup, just double-click and go! 🚀

## Important: Python Dependencies

Users need to install Python dependencies **once** before first use:

```bash
# After extracting/running AppImage, install Python deps
cd /path/to/app/resources/backend-python-ml-v2
pip install -r requirements.txt
```

Or the launcher will attempt to use system Python and dependencies.

