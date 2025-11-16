# Standalone AppImage - Double-Click to Run

## Overview

This creates a **standalone AppImage** that users can **double-click to run** without any setup. The AppImage automatically starts all required services.

## How It Works

When users double-click the AppImage:

1. ✅ **AppImage mounts** and extracts bundled files
2. ✅ **Electron starts** and detects standalone mode
3. ✅ **Standalone launcher** automatically starts:
   - **Python ML service** (gRPC on port 50051)
   - **NestJS API** (HTTP on port 3001) 
   - **Next.js Frontend** (HTTP on port 3000)
4. ✅ **Electron window opens** and loads the frontend
5. ✅ **User can use the app** - everything works!

When users close the app:
- ✅ All services automatically stop
- ✅ No cleanup needed

## Build the AppImage

### Quick Build

```bash
./build-appimage.sh
```

### Manual Build

```bash
# 1. Install dependencies
npm install
cd frontend && npm install && cd ..
cd backend-nestjs && npm install && cd ..

# 2. Build frontend
cd frontend && npm run build && cd ..

# 3. Build NestJS
cd backend-nestjs && npm run build && cd ..

# 4. Build AppImage
npm run build:electron -- --linux
```

## Output

The AppImage will be created at:
```
dist/Video Translator-1.0.0-x86_64.AppImage
```

**File size:** ~500MB-1GB (includes all dependencies)

## User Requirements

Users need:
- ✅ **Linux** (Ubuntu 20.04+, Debian 11+, Fedora 34+, etc.)
- ✅ **Python 3.9+** installed system-wide
- ✅ **FUSE** (usually pre-installed): `sudo apt-get install fuse`

**That's it!** No Docker, no Node.js installation, no manual setup.

## Distribution

### For End Users

1. **Download** the AppImage file
2. **Make executable:**
   ```bash
   chmod +x "Video Translator-1.0.0-x86_64.AppImage"
   ```
3. **Double-click** to run!

### Optional: Create Desktop Entry

```bash
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

## What's Bundled

- ✅ Electron app (UI)
- ✅ Next.js frontend (with node_modules)
- ✅ NestJS backend (with node_modules)
- ✅ Python ML service source code
- ✅ Node.js runtime (bundled by Electron)

**Note:** Python runtime is **NOT bundled** - users need Python 3.9+ installed. This keeps the AppImage size reasonable.

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
   ./Video\ Translator-*.AppImage --debug 2>&1 | tee app.log
   ```

### Services won't start

Check the terminal output for:
- `[Launcher] Starting Python ML service...`
- `[Launcher] Starting NestJS API...`
- `[Launcher] Starting Frontend...`

Look for any error messages.

### Port conflicts

If ports 3000, 3001, or 50051 are already in use:
- Close other instances of the app
- Or stop other services using those ports

### "Python not found" error

Users need Python 3.9+ installed:
```bash
sudo apt-get install python3 python3-pip
```

## First Launch

The first launch may take **10-20 seconds** while services start:
- Python ML service initializes (~5 seconds)
- NestJS API starts (~3 seconds)
- Next.js frontend starts (~2 seconds)

Subsequent launches are faster.

## File Locations

- **AppImage:** Wherever you place it (portable!)
- **User data:** `~/video-translator-uploads/` (created automatically)
- **Logs:** Check terminal output when running

## Advanced: Bundle Python Runtime

To make it completely standalone (no Python requirement):

1. Download Python static build or AppImage
2. Include in `extraResources`
3. Update `standalone-launcher.js` to use bundled Python

This increases AppImage size to ~1.5GB+ but removes Python requirement.

## Testing

Before distributing:

1. ✅ Build the AppImage
2. ✅ Test on a clean system (or VM)
3. ✅ Verify all services start
4. ✅ Test video upload
5. ✅ Test translation
6. ✅ Test on different Linux distributions

## Notes

- AppImage is **portable** - can be moved anywhere
- **No installation** required - just download and run
- All user data stored in `~/video-translator-uploads/`
- Services automatically stop when app closes


