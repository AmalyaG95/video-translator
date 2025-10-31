#!/usr/bin/env python3
"""
Python ML Microservice - gRPC Server
Handles video translation with ML models (STT, Translation, TTS)
"""

import asyncio
import logging
import signal
from concurrent import futures

import grpc

import translation_pb2_grpc
from translation_service import TranslationServicer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def serve():
    """Start gRPC server"""
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
        ]
    )
    
    # Add servicer
    translation_pb2_grpc.add_TranslationServiceServicer_to_server(
        TranslationServicer(), server
    )
    
    # Listen on port 50051
    listen_addr = '0.0.0.0:50051'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"🚀 Python ML Service starting on {listen_addr}")
    logger.info("Ready to accept translation requests from NestJS API Gateway")
    
    await server.start()
    
    # Graceful shutdown handler
    async def shutdown(sig):
        logger.info(f"Received shutdown signal: {sig.name}")
        await server.stop(grace=5)
        logger.info("Server stopped gracefully")
    
    # Register shutdown handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown(s))
        )
    
    await server.wait_for_termination()


if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)

