# System Architecture - Video Translation Service

## High-Level Overview

A professional video translation service transforms video content from one language to another while preserving lip-sync, audio quality, and visual fidelity. The architecture must support:

- **Asynchronous Processing**: Long-running operations that can take minutes to hours
- **State Management**: Track progress, resume from failures, maintain session state
- **Resource Efficiency**: Manage memory, CPU, and storage for large video files
- **Quality Assurance**: Validate output at each stage
- **Scalability**: Handle multiple concurrent video processing requests
- **Reliability**: Recover from failures, handle edge cases gracefully

## Architectural Patterns

### 1. Pipeline Architecture

The system follows a **staged pipeline pattern** where each stage transforms input to output:

```
Video Input → Audio Extraction → Transcription → Translation → TTS → Audio Sync → Video Combination → Output
```

**Key Principles:**

- Each stage is **idempotent** where possible (can be retried safely)
- Stages are **loosely coupled** (can be tested independently)
- **Checkpointing** between stages enables resume capability
- **Parallel processing** where stages don't depend on each other

### 2. Service Architecture

**Recommended Approach: Microservices with Clear Boundaries**

```
┌─────────────────┐
│  API Gateway    │  (REST/gRPC/WebSocket)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│  API  │ │   ML    │
│Service│ │ Service │
└───────┘ └─────────┘
```

**Service Separation:**

- **API Service**: Handles HTTP/gRPC requests, session management, progress tracking
- **ML Service**: Core video processing pipeline, model management, heavy computation

**Benefits:**

- Independent scaling (ML service needs more resources)
- Technology flexibility (API in Node.js/NestJS, ML in Python)
- Fault isolation (ML service failures don't crash API)
- Resource optimization (ML service can use GPU, API doesn't need it)

### 3. Async Processing Model

**Event-Driven Architecture with Progress Tracking**

```
Client Request → Session Created → Processing Started → Progress Updates (SSE/WebSocket) → Completion
```

**Implementation Pattern:**

- Use **async/await** for non-blocking I/O operations
- **Background tasks** for long-running processing
- **Progress callbacks** for real-time updates
- **Session state** persisted to database/storage

**Key Requirements:**

- Non-blocking: API remains responsive during processing
- Observable: Clients can track progress
- Resumable: Processing can continue after interruption
- Cancellable: Clients can stop processing

### 4. State Management

**Session-Based State with Checkpointing**

Each video processing job is a **session** with:

- Unique session ID
- Current stage/status
- Progress percentage
- Checkpoint data (for resume)
- Artifacts (intermediate files)
- Quality metrics

**Storage Strategy:**

- **Metadata**: Database (PostgreSQL/MongoDB) for fast queries
- **Files**: Object storage (S3/MinIO) or filesystem for large files
- **Checkpoints**: JSON files in session-specific directories
- **Logs**: Structured JSONL files for analysis

### 5. Communication Patterns

**Multi-Protocol Support**

1. **REST API**: Standard HTTP for simple operations
2. **gRPC**: Efficient binary protocol for ML service communication
3. **WebSocket/SSE**: Real-time progress updates to clients
4. **Message Queue** (optional): For distributed processing

**Best Practice:**

- Use **gRPC** for service-to-service communication (efficient, type-safe)
- Use **WebSocket/SSE** for client progress updates (real-time, low overhead)
- Use **REST** for simple CRUD operations (familiar, easy to debug)

## Component Architecture

### Core Components

1. **Pipeline Orchestrator**

   - Manages stage execution order
   - Handles stage transitions
   - Coordinates parallel processing
   - Manages checkpoints

2. **Model Manager**

   - Loads/unloads ML models on demand
   - Manages model lifecycle
   - Handles model versioning
   - Optimizes memory usage

3. **Resource Manager**

   - Monitors memory/CPU usage
   - Triggers cleanup when needed
   - Manages temporary file lifecycle
   - Prevents resource exhaustion

4. **Quality Validator**

   - Validates output at each stage
   - Collects quality metrics
   - Generates quality reports
   - Flags issues for review

5. **Progress Tracker**
   - Tracks stage progress
   - Calculates ETA
   - Broadcasts updates
   - Handles client disconnections

## Data Flow Architecture

### Request Flow

```
1. Client uploads video → API Service
2. API Service creates session → Database
3. API Service sends processing request → ML Service (gRPC)
4. ML Service starts processing → Background task
5. ML Service sends progress updates → API Service (gRPC stream)
6. API Service broadcasts updates → Client (WebSocket/SSE)
7. ML Service completes → Sends final result
8. API Service updates session → Database
9. Client receives completion notification
```

### File Flow

```
Input Video → Temp Storage → Processing → Artifacts → Final Output
                ↓              ↓            ↓
            Checkpoints    Logs        Subtitles
```

**Best Practice:**

- Use **session-scoped directories** for all files
- **Cleanup automatically** after session completion or expiration
- **Separate temp files** from artifacts (artifacts persist, temp files cleaned)
- **Use absolute paths** to avoid path resolution issues

## Scalability Considerations

### Horizontal Scaling

**Stateless API Service:**

- Multiple instances behind load balancer
- Session state in shared database
- No session affinity required

**ML Service Scaling:**

- **Challenge**: Models are memory-intensive (2-8GB per model)
- **Solution**: Model pooling or dedicated model servers
- **Alternative**: Queue-based processing with worker pools

### Vertical Scaling

**Resource Requirements:**

- **CPU**: Multi-core for parallel processing
- **GPU**: Optional but recommended for faster inference
- **Memory**: 16GB+ for model loading and video processing
- **Storage**: Fast SSD for temporary files

**Optimization Strategies:**

- **Lazy model loading**: Load models only when needed
- **Model caching**: Keep frequently used models in memory
- **Batch processing**: Process multiple segments in parallel
- **Streaming**: Process video in chunks to reduce memory

## Security Architecture

### Input Validation

- **File type validation**: Only accept supported video formats
- **File size limits**: Prevent resource exhaustion
- **Content scanning**: Optional virus/malware scanning
- **Rate limiting**: Prevent abuse

### Data Isolation

- **Session isolation**: Each session has separate directory
- **No cross-session access**: Strict path validation
- **Automatic cleanup**: Remove session data after expiration
- **Secure file deletion**: Overwrite sensitive data before deletion

### API Security

- **Authentication**: API keys or OAuth tokens
- **Authorization**: Role-based access control
- **Input sanitization**: Validate all inputs
- **Output validation**: Verify outputs before serving

## Deployment Architecture

### Container-Based Deployment

**Recommended: Docker Compose for Development, Kubernetes for Production**

```
┌─────────────────────────────────────┐
│         Docker/Kubernetes           │
│  ┌──────────┐      ┌──────────────┐  │
│  │   API    │◄────►│     ML      │  │
│  │ Service  │ gRPC │   Service   │  │
│  └──────────┘      └──────────────┘  │
│       │                   │          │
│       ▼                   ▼          │
│  ┌──────────┐      ┌──────────────┐  │
│  │Database  │      │  File Store  │  │
│  └──────────┘      └──────────────┘  │
└─────────────────────────────────────┘
```

**Benefits:**

- **Isolation**: Services don't interfere with each other
- **Resource limits**: Prevent one service from consuming all resources
- **Easy scaling**: Add more containers as needed
- **Consistent environment**: Same behavior in dev and production

### Resource Allocation

**API Service:**

- CPU: 1-2 cores
- Memory: 512MB-1GB
- No GPU needed

**ML Service:**

- CPU: 4-8 cores
- Memory: 8-16GB
- GPU: Optional but recommended

**Storage:**

- Fast SSD for temp files (high IOPS)
- Slower storage acceptable for artifacts
- Consider network storage for shared access

## Monitoring Architecture

### Observability Stack

1. **Structured Logging**: JSONL format for easy parsing
2. **Metrics Collection**: Prometheus-compatible metrics
3. **Distributed Tracing**: OpenTelemetry for request tracking (see `MODERN-2025-PRACTICES.md`)
4. **Error Tracking**: Sentry or similar for error aggregation
5. **APM (Application Performance Monitoring)**: Datadog, New Relic, or Elastic APM
6. **Real User Monitoring (RUM)**: Frontend performance and user experience tracking

### Key Metrics

- **Processing time** per stage
- **Success/failure rates**
- **Resource usage** (CPU, memory, disk)
- **Queue depth** (if using queues)
- **Model load times**
- **Quality scores**

### Alerting

- **Processing failures**: Alert on consecutive failures
- **Resource exhaustion**: Alert when memory/disk is high
- **Quality degradation**: Alert when quality scores drop
- **Service downtime**: Alert when service is unreachable

## Best Practices Summary

1. **Separation of Concerns**: API and ML services are separate
2. **Async Everything**: Use async/await for all I/O operations
3. **State Persistence**: Always persist session state
4. **Checkpointing**: Enable resume from any stage
5. **Resource Management**: Monitor and manage resources proactively
6. **Quality Validation**: Validate output at each stage
7. **Structured Logging**: Use structured logs for analysis
8. **Error Recovery**: Handle errors gracefully with retries
9. **Cleanup**: Automatically clean up temporary files
10. **Security**: Validate inputs, isolate data, secure APIs

## Next Steps

- See `01-SYSTEM-DESIGN.md` for detailed design principles
- See `02-PIPELINE-OVERVIEW.md` for end-to-end flow
- See `stages/` for stage-specific best practices
- See `cross-cutting/` for cross-cutting concerns
- See `patterns/` for implementation patterns
