# Integration Guide

## Integrating ML Service v2 with NestJS API

This guide explains how to integrate the new ML service v2 with your existing NestJS API.

## Quick Integration

### Option 1: Use docker-compose.v2.yml (Recommended)

The `docker-compose.v2.yml` file is already configured to run the new service alongside your existing infrastructure:

```bash
# Start all services with v2
docker-compose -f docker-compose.v2.yml up --build
```

This will:
- Start the new `python-ml-v2` service on port 50052
- Configure NestJS to use the new service (`GRPC_ML_SERVICE_URL=python-ml-v2:50051`)
- Keep all other services unchanged

### Option 2: Update Existing docker-compose.yml

If you want to replace the old service in your main compose file:

1. **Update the service name and build context:**
```yaml
python-ml-v2:  # Changed from python-ml
  build:
    context: ./backend-python-ml-v2  # Changed from backend-python-ml
```

2. **Update NestJS environment:**
```yaml
nestjs-api:
  environment:
    - GRPC_ML_SERVICE_URL=python-ml-v2:50051
```

3. **Restart services:**
```bash
docker-compose down
docker-compose up --build
```

## gRPC Compatibility

The new service maintains **100% gRPC compatibility** with the old service:

- ✅ Same proto definitions (`translation.proto`)
- ✅ Same service methods
- ✅ Same request/response formats
- ✅ Same progress streaming format

**No changes required to NestJS code!**

## Testing Integration

### 1. Verify Service Startup

```bash
# Check service logs
docker-compose -f docker-compose.v2.yml logs python-ml-v2

# Should see:
# - "Proto files generated" (first run)
# - "gRPC server started on port 50051"
# - "Service ready to accept requests"
```

### 2. Test gRPC Connection

```bash
# From NestJS container
docker-compose -f docker-compose.v2.yml exec nestjs-api \
  node -e "const grpc = require('@grpc/grpc-js'); \
           const client = new grpc.Client('python-ml-v2:50051', grpc.credentials.createInsecure()); \
           console.log('Connected:', client);"
```

### 3. Test Translation Request

1. Upload a video through the frontend
2. Start a translation
3. Monitor logs:
   ```bash
   docker-compose -f docker-compose.v2.yml logs -f python-ml-v2
   ```

## Environment Variables

### ML Service v2

The service reads configuration from:
- `src/config/config.yaml` (defaults)
- Environment variables (override)

Key variables:
```bash
MAX_MEMORY_GB=8          # Max memory for models
GRPC_PORT=50051          # gRPC server port
ENABLE_TRACING=true      # OpenTelemetry tracing
```

### NestJS API

Update to point to new service:
```bash
GRPC_ML_SERVICE_URL=python-ml-v2:50051
```

## Migration Strategy

### Phase 1: Parallel Running (Testing)
- Run both services simultaneously
- Use different ports (old: 50051, new: 50052)
- Test new service with subset of traffic
- Monitor for issues

### Phase 2: Gradual Migration
- Update NestJS to use new service
- Keep old service as backup
- Monitor performance and errors
- Rollback if needed

### Phase 3: Full Migration
- Remove old service
- Update all references
- Clean up old code

## Troubleshooting

### Service Won't Start

1. **Check proto files:**
   ```bash
   docker-compose -f docker-compose.v2.yml exec python-ml-v2 \
     ls -la src/proto/*_pb2.py
   ```
   Should see `translation_pb2.py` and `translation_pb2_grpc.py`

2. **Check logs:**
   ```bash
   docker-compose -f docker-compose.v2.yml logs python-ml-v2
   ```

3. **Verify imports:**
   ```bash
   docker-compose -f docker-compose.v2.yml exec python-ml-v2 \
     python3 verify_imports.py
   ```

### gRPC Connection Errors

1. **Verify service is listening:**
   ```bash
   docker-compose -f docker-compose.v2.yml exec python-ml-v2 \
     netstat -tlnp | grep 50051
   ```

2. **Check network connectivity:**
   ```bash
   docker-compose -f docker-compose.v2.yml exec nestjs-api \
     ping python-ml-v2
   ```

3. **Verify service name:**
   - In docker-compose, service name must match `GRPC_ML_SERVICE_URL`
   - Use service name, not `localhost`

### Translation Failures

1. **Check resource limits:**
   - Service needs sufficient memory (4-8GB recommended)
   - Check Docker resource limits

2. **Monitor logs:**
   ```bash
   docker-compose -f docker-compose.v2.yml logs -f python-ml-v2
   ```

3. **Check model downloads:**
   - First run downloads models (can take time)
   - Check disk space

## Performance Comparison

The new service includes:
- ✅ Better resource management
- ✅ Improved error handling
- ✅ Quality validation
- ✅ Checkpointing for resume
- ✅ Better logging

Expected improvements:
- More stable under load
- Better error recovery
- More detailed progress reporting

## Rollback Plan

If issues occur:

1. **Quick rollback:**
   ```bash
   # Update NestJS environment
   GRPC_ML_SERVICE_URL=python-ml:50051
   
   # Restart NestJS
   docker-compose restart nestjs-api
   ```

2. **Full rollback:**
   ```bash
   # Stop v2 service
   docker-compose -f docker-compose.v2.yml stop python-ml-v2
   
   # Start old service
   docker-compose up -d python-ml
   ```

## Support

- Review logs: `docker-compose logs python-ml-v2`
- Check `DEPLOYMENT.md` for deployment details
- See `CHECKLIST.md` for pre-deployment checklist
- Review `best-practices/` for implementation details



