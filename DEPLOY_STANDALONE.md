# Deploy Standalone AppImage (Double-Click to Run)

This guide explains how to create a **standalone AppImage** that users can **double-click to run** without any setup.

## What Gets Bundled

The standalone AppImage includes:
- ✅ Electron app (frontend UI)
- ✅ NestJS backend (Node.js API server)
- ✅ Python ML service (AI translation engine)
- ✅ All dependencies

**Note:** Python runtime is **not bundled** - users need Python 3.9+ installed. This keeps the AppImage size reasonable (~300-500MB instead of 1GB+).

## Prerequisites for Building

1. **Node.js** (v20+)
2. **Python 3.9+** (for building, not bundled)
3. **Build tools** (see below)
4. **Linux system** (for building Linux AppImage)

### Install Build Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  libnss3-dev \
  libatk-bridge2.0-dev \
  libdrm2 \
  libxkbcommon-dev \
  libxcomposite-dev \
  libxdamage-dev \
  libxrandr-dev \
  libgbm-dev \
  libxss1 \
  libasound2-dev \
  python3 \
  python3-pip
```

## Build Steps

### 1. Install All Dependencies

```bash
# Root dependencies
npm install

# Frontend dependencies
cd frontend && npm install && cd ..

# NestJS dependencies
cd backend-nestjs && npm install && cd ..
```

### 2. Build Frontend (Static Export)

```bash
cd frontend
npm run build
# Make sure it creates frontend/out/ directory
cd ..
```

### 3. Build NestJS Backend

```bash
cd backend-nestjs
npm run build
cd ..
```

### 4. Install Python ML Dependencies (for reference)

```bash
cd backend-python-ml-v2
pip3 install -r requirements.txt
cd ..
```

### 5. Build AppImage

```bash
npm run build:electron -- --linux
```

Or use the automated script:

```bash
./build-appimage.sh
```

## Output

After building, you'll find:

```
dist/
  └── Video Translator-1.0.0-x86_64.AppImage
```

**File size:** ~300-500MB (depends on bundled dependencies)

## How It Works

When users double-click the AppImage:

1. **AppImage mounts** and extracts files
2. **Electron starts** and detects standalone mode
3. **Standalone launcher** automatically starts:
   - Python ML service (gRPC on port 50051)
   - NestJS API (HTTP on port 3001)
4. **Frontend loads** as static files (no server needed)
5. **User can use the app** - everything works!

When users close the app:
- All services automatically stop
- No cleanup needed

## User Requirements

Users need:
- ✅ **Linux** (Ubuntu 20.04+, Debian 11+, Fedora 34+, etc.)
- ✅ **Python 3.9+** installed system-wide
- ✅ **FUSE** (usually pre-installed): `sudo apt-get install fuse`

**That's it!** No Docker, no Node.js, no manual setup.

## Distribution

### For End Users

1. **Download** the AppImage file
2. **Make executable:**
   ```bash
   chmod +x "Video Translator-1.0.0-x86_64.AppImage"
   ```
3. **Double-click** to run!

### Optional: Create Desktop Entry

Users can create a desktop entry for easier access:

```bash
# Create desktop entry
cat > ~/.local/share/applications/video-translator.desktop << 'EOF'
[Desktop Entry]
Name=Video Translator
Exec=/path/to/Video\ Translator-1.0.0-x86_64.AppImage
Icon=/path/to/Video\ Translator-1.0.0-x86_64.AppImage
Type=Application
Categories=Video;
StartupNotify=true
EOF
```

## Troubleshooting

### AppImage won't run

1. **Check permissions:**
   ```bash
   chmod +x "Video Translator-*.AppImage"
   ```

2. **Check Python:**
   ```bash
   python3 --version  # Should be 3.9+
   ```

3. **Run with debug:**
   ```bash
   ./Video\ Translator-*.AppImage --debug
   ```

### "Python not found" error

Users need Python 3.9+ installed:
```bash
sudo apt-get install python3 python3-pip
```

### Services won't start

Check logs in terminal:
```bash
./Video\ Translator-*.AppImage 2>&1 | tee app.log
```

Look for:
- `[Launcher] Starting Python ML service...`
- `[Launcher] Starting NestJS API...`
- Any error messages

### Port conflicts

If ports 3001 or 50051 are already in use:
- Close other instances of the app
- Or stop other services using those ports

## Advanced: Bundle Python Runtime

If you want to bundle Python (larger AppImage, ~1GB+):

1. Download Python AppImage or static build
2. Include it in `extraResources`
3. Update `standalone-launcher.js` to use bundled Python

This makes the AppImage completely standalone but much larger.

## File Structure in AppImage

```
AppImage (mounted)
├── resources/
│   ├── app.asar (Electron app)
│   └── app.asar.unpacked/
│       ├── backend-nestjs/ (NestJS backend)
│       ├── backend-python-ml-v2/ (Python ML service)
│       └── electron/
│           └── standalone-launcher.js
└── (Electron runtime)
```

## Testing Before Distribution

1. **Build the AppImage**
2. **Test on a clean system** (or VM) without your development environment
3. **Verify:**
   - AppImage runs
   - Services start automatically
   - Frontend loads
   - Video upload works
   - Translation works
4. **Check file size** (should be reasonable)
5. **Test on different Linux distributions** if possible

## Version Updates

To update the version:

1. Update `version` in `package.json`
2. Rebuild: `npm run build:electron -- --linux`
3. New AppImage will have updated version in filename

## Notes

- First launch may take 10-20 seconds (services starting)
- Subsequent launches are faster
- All user data (uploads, results) stored in `~/video-translator-uploads/`
- AppImage is portable - can be moved anywhere
- No installation required - just download and run!


