# Frontend Hot Reload Setup

## Quick Start

### Option 1: Use Development Override (Recommended)
```bash
# Start with hot reload enabled
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up frontend

# Or start all services with frontend in dev mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Option 2: Set Environment Variable
```bash
# Set NODE_ENV=development in docker-compose.yml or use env var
NODE_ENV=development docker-compose up frontend
```

### Option 3: Run Frontend Locally (Fastest)
```bash
cd frontend
npm install
npm run dev
```

Then access at `http://localhost:3000`

## How It Works

1. **Volume Mounts**: Source code is mounted from `./frontend` to `/app` in container
2. **Node Modules**: Excluded from mount (container uses its own)
3. **Next.js Cache**: `.next` directory excluded to prevent conflicts
4. **Watch Mode**: `npm run dev` enables Next.js Fast Refresh
5. **Polling**: `WATCHPACK_POLLING=true` ensures file changes are detected in Docker

## Making Changes

After setup, any changes to files in `frontend/` will automatically:
- Trigger Next.js Fast Refresh
- Update the browser without full page reload
- Preserve component state where possible

## Troubleshooting

If hot reload doesn't work:
1. Check container logs: `docker-compose logs -f frontend`
2. Verify volumes: `docker-compose exec frontend ls -la /app`
3. Restart container: `docker-compose restart frontend`
4. Clear Next.js cache: `docker-compose exec frontend rm -rf .next`

## Production Mode

For production builds, use the default docker-compose.yml (no dev override):
```bash
docker-compose up frontend
```

