# Deploy Electron App and Create AppImage

This guide explains how to build and deploy the Electron app as an AppImage for Linux.

## Prerequisites

1. **Node.js** (v20 or higher)
2. **npm** or **yarn**
3. **Build tools** for native dependencies
4. **Linux system** (for building Linux AppImage)

### Install Build Dependencies (Ubuntu/Debian)

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
  libasound2-dev
```

## Build Steps

### 1. Install Dependencies

```bash
# Install root dependencies
npm install

# Install frontend dependencies
cd frontend && npm install && cd ..

# Build frontend (required before packaging)
cd frontend && npm run build && cd ..
```

### 2. Build NestJS Backend (if needed)

```bash
cd backend-nestjs
npm install
npm run build
cd ..
```

### 3. Build AppImage

```bash
# Build AppImage for Linux
npm run build:electron -- --linux

# Or build for specific architecture
npm run build:electron -- --linux --x64
```

The AppImage will be created in the `dist/` directory.

## Output

After building, you'll find:

```
dist/
  └── Video Translator-1.0.0-x86_64.AppImage
```

## Running the AppImage

### Make it executable

```bash
chmod +x "dist/Video Translator-1.0.0-x86_64.AppImage"
```

### Run it

```bash
./dist/Video\ Translator-1.0.0-x86_64.AppImage
```

Or double-click it in your file manager.

## Complete Build Script

Create a `build-appimage.sh` script:

```bash
#!/bin/bash
set -e

echo "📦 Building Video Translator AppImage..."

# Install dependencies
echo "📥 Installing dependencies..."
npm install
cd frontend && npm install && cd ..

# Build frontend
echo "🏗️  Building frontend..."
cd frontend && npm run build && cd ..

# Build backend (if needed)
echo "🏗️  Building NestJS backend..."
cd backend-nestjs && npm run build && cd ..

# Build AppImage
echo "📦 Creating AppImage..."
npm run build:electron -- --linux

echo "✅ Build complete! AppImage is in dist/ directory"
ls -lh dist/*.AppImage
```

Make it executable:
```bash
chmod +x build-appimage.sh
./build-appimage.sh
```

## Troubleshooting

### AppImage won't run

1. **Check permissions:**
   ```bash
   chmod +x "dist/Video Translator-1.0.0-x86_64.AppImage"
   ```

2. **Run with debug output:**
   ```bash
   ./dist/Video\ Translator-1.0.0-x86_64.AppImage --debug
   ```

3. **Check system compatibility:**
   - AppImage requires FUSE (Filesystem in Userspace)
   - Install: `sudo apt-get install fuse`

### Build fails with missing dependencies

Install missing system libraries:
```bash
sudo apt-get install -y libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 libasound2
```

### Frontend not found error

Make sure you've built the frontend:
```bash
cd frontend && npm run build && cd ..
```

### Backend not found error

The AppImage includes the backend, but if you're running in development mode, make sure the backend services are running separately.

## Distribution

### For Users

1. Download the AppImage file
2. Make it executable: `chmod +x Video-Translator-*.AppImage`
3. Run it: `./Video-Translator-*.AppImage`

### Optional: Create Desktop Entry

Users can create a desktop entry for easier access:

```bash
# Create desktop entry
cat > ~/.local/share/applications/video-translator.desktop << EOF
[Desktop Entry]
Name=Video Translator
Exec=/path/to/Video\ Translator-*.AppImage
Icon=/path/to/Video\ Translator-*.AppImage
Type=Application
Categories=Video;
EOF
```

## Advanced Configuration

### Custom Icon

Place your icon at `electron/assets/logo.png` (512x512 or 1024x1024 PNG recommended).

### Update Version

Update version in `package.json`:
```json
{
  "version": "1.0.1"
}
```

### Build for Multiple Platforms

```bash
# Linux AppImage
npm run build:electron -- --linux

# Windows
npm run build:electron -- --win

# macOS
npm run build:electron -- --mac
```

## Notes

- The AppImage is self-contained and includes all dependencies
- No installation required - just download and run
- Works on most modern Linux distributions
- File size will be large (~200-500MB) due to bundled dependencies


