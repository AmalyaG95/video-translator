# Hot Reload Setup for All Services

## Quick Start

### Development Mode (Hot Reload Enabled)
```bash
# Start all services with hot reload
NODE_ENV=development docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Production Mode (No Hot Reload)
```bash
# Start all services in production mode
docker-compose up
```

## What's Hot Reloaded

### ✅ Frontend (Next.js)
- **Status**: ✅ Fully hot reloaded
- **How**: Volume mount + `npm run dev`
- **Changes**: Instant (Fast Refresh)
- **Files**: All files in `frontend/`

### ✅ NestJS API
- **Status**: ✅ Hot reloaded with watch mode
- **How**: Volume mount + `npm run start:dev`
- **Changes**: Auto-restarts on file change
- **Files**: All files in `backend-nestjs/src/`

### ⚠️ Python ML Service
- **Status**: ⚠️ Source mounted, but requires manual restart
- **How**: Volume mount (code changes visible)
- **Restart**: `docker-compose restart python-ml`
- **Files**: All files in `backend-python-ml/src/`
- **Note**: Python/gRPC doesn't support hot reload easily, but you can restart quickly

## Quick Commands

### Start Development
```bash
NODE_ENV=development docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Restart Python ML Service (after code changes)
```bash
docker-compose restart python-ml
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f frontend
docker-compose logs -f nestjs-api
docker-compose logs -f python-ml
```

## Volume Mounts

### Frontend
- `./frontend:/app` - Source code
- `/app/node_modules` - Excluded (container's own)
- `/app/.next` - Excluded (container's own)

### NestJS
- `./backend-nestjs/src:/app/src` - Source code
- `./backend-nestjs/package.json:/app/package.json` - Package file
- `/app/node_modules` - Excluded (container's own)
- `/app/dist` - Excluded (container's own)

### Python ML
- `./backend-python-ml/src:/app/src` - Source code
- `./backend-python-ml/src/config:/app/src/config` - Config (already mounted)

## Tips

1. **Frontend**: Changes appear instantly, no restart needed
2. **NestJS**: Auto-restarts on file save (watch mode)
3. **Python**: Restart service after code changes: `docker-compose restart python-ml`
4. **Rebuild**: Only needed when adding new dependencies or changing Dockerfiles

