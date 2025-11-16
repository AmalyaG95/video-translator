# Running Electron App in Docker

This guide explains how to run the Electron desktop app inside a Docker container.

## Prerequisites

1. **X11 Server** (Linux only):
   - X11 must be running on your host
   - You need to allow Docker containers to access your X server

2. **Docker Services Running**:
   ```bash
   docker-compose -f docker-compose.v2.yml up -d
   ```

## Setup X11 Access (Linux)

Before running Electron in Docker, you need to allow the container to access your X11 display:

```bash
# Allow local Docker containers to access X11
xhost +local:docker

# Or more securely, allow only root user in containers:
xhost +SI:localuser:root
```

**Security Note**: This allows Docker containers to access your X server. Only do this on trusted systems.

To revoke access later:
```bash
xhost -local:docker
```

## Running Electron in Docker

### Option 1: Using Docker Compose (Recommended)

```bash
# 1. Allow X11 access (one-time setup per session)
xhost +local:docker

# 2. Start all services including Electron
docker-compose -f docker-compose.v2.yml up -d electron

# 3. View logs
docker-compose -f docker-compose.v2.yml logs -f electron
```

### Option 2: Using Docker Run

```bash
# 1. Allow X11 access
xhost +local:docker

# 2. Build the image
docker build -f electron/Dockerfile -t translate-v-electron .

# 3. Run the container
docker run -it --rm \
  --name translate-v-electron \
  -e DISPLAY=$DISPLAY \
  -e NODE_ENV=development \
  -e SKIP_BACKEND=true \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/electron:/app/electron \
  -v $(pwd)/package.json:/app/package.json \
  --network host \
  translate-v-electron
```

## Configuration

The Electron container is configured to:
- Connect to frontend at `http://localhost:3000` (from Docker services)
- Connect to NestJS API at `http://localhost:3001`
- Use gRPC at `localhost:50051` (Python ML service)

## Network Modes

### Host Network (Current Setup)
- Uses `network_mode: "host"` to access services on localhost
- Simplest setup, works out of the box
- Electron can access services on `localhost:3000`, `localhost:3001`, etc.

### Bridge Network (Alternative)
If you prefer using Docker's bridge network:

1. Update `docker-compose.v2.yml`:
   ```yaml
   electron:
     # Remove: network_mode: "host"
     networks:
       - translator-network
     depends_on:
       - frontend
       - nestjs-api
   ```

2. Update Electron's connection URLs in `electron/main.js`:
   ```javascript
   const startUrl = isDev
     ? "http://frontend:3000"  // Use service name
     : `file://${path.join(__dirname, "../frontend/out/index.html")}`;
   ```

## Troubleshooting

### "Cannot connect to X server"

**Error**: `Error: Cannot connect to X server`

**Solution**:
```bash
# Check if X11 is running
echo $DISPLAY

# Allow Docker access
xhost +local:docker

# Verify X11 socket exists
ls -la /tmp/.X11-unix
```

### "No display found"

**Error**: `No display found`

**Solution**:
```bash
# Set DISPLAY environment variable
export DISPLAY=:0

# Or for specific display
export DISPLAY=:1
```

### Electron Window Doesn't Appear

1. Check if X11 forwarding is working:
   ```bash
   docker exec translate-v-electron xeyes
   # If xeyes appears, X11 is working
   ```

2. Check Electron logs:
   ```bash
   docker-compose -f docker-compose.v2.yml logs electron
   ```

3. Verify services are accessible:
   ```bash
   curl http://localhost:3000
   curl http://localhost:3001/health
   ```

### Permission Denied for X11

**Error**: `X11 connection rejected because of wrong authentication`

**Solution**:
```bash
# Allow access
xhost +local:docker

# Or copy X11 auth
xauth list | grep $(hostname) | xargs -I {} xauth add {}  # On host
docker cp ~/.Xauthority translate-v-electron:/root/.Xauthority  # Copy to container
```

## Alternative: VNC/NoVNC (Headless)

If you don't have X11 or want a remote display, you can use VNC:

1. Install VNC server in the Dockerfile:
   ```dockerfile
   RUN apt-get install -y tigervnc-standalone-server tigervnc-common
   ```

2. Start VNC server in the container:
   ```bash
   vncserver :1 -geometry 1920x1080 -depth 24
   ```

3. Access via VNC client or noVNC in browser

## Security Considerations

⚠️ **Warning**: Allowing Docker containers to access your X11 server is a security risk:
- Containers can capture keystrokes
- Containers can take screenshots
- Containers can control your desktop

**Best Practices**:
1. Only allow on trusted systems
2. Revoke access when done: `xhost -local:docker`
3. Use a separate X server for Docker if possible
4. Consider using VNC for remote access instead

## Development Workflow

1. **Start Docker services**:
   ```bash
   docker-compose -f docker-compose.v2.yml up -d
   ```

2. **Allow X11 access** (one-time per session):
   ```bash
   xhost +local:docker
   ```

3. **Start Electron**:
   ```bash
   docker-compose -f docker-compose.v2.yml up electron
   ```

4. **Make changes** - Electron will reload when you refresh (F5)

5. **View logs**:
   ```bash
   docker-compose -f docker-compose.v2.yml logs -f electron
   ```

## Stopping

```bash
# Stop Electron container
docker-compose -f docker-compose.v2.yml stop electron

# Remove Electron container
docker-compose -f docker-compose.v2.yml rm -f electron

# Revoke X11 access (optional, for security)
xhost -local:docker
```


