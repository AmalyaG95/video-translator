"""
OpenTelemetry Tracing Setup

Follows best-practices/cross-cutting/MODERN-2025-PRACTICES.md observability patterns.
"""

from typing import Optional
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from ..config import Config
from ..app_logging import get_logger

logger = get_logger("observability.tracing")


def setup_observability(config: Config) -> None:
    """
    Setup OpenTelemetry tracing.
    
    Follows best-practices/cross-cutting/MODERN-2025-PRACTICES.md OpenTelemetry patterns.
    """
    if not config.settings.enable_tracing:
        logger.info("Tracing disabled in configuration")
        return
    
    try:
        # Create resource
        resource = Resource.create({
            "service.name": "video-translation-ml-service",
            "service.version": "2.0.0",
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter if endpoint provided
        if config.settings.tracing_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=config.settings.tracing_endpoint,
                insecure=True,  # Use insecure for development
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            logger.info(
                "OpenTelemetry tracing enabled",
                extra_data={"endpoint": config.settings.tracing_endpoint},
            )
        else:
            logger.info("OpenTelemetry tracing configured but no endpoint provided")
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
    except Exception as e:
        logger.warning(
            "Failed to setup OpenTelemetry tracing",
            exc_info=True,
            extra_data={"error": str(e)},
        )


