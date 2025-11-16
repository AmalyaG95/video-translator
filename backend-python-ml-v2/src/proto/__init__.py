"""
gRPC protocol definitions.

Proto files (translation_pb2.py, translation_pb2_grpc.py) are generated
at runtime by the entrypoint script from translation.proto.
"""

# These will be available after proto generation
try:
    from . import translation_pb2
    from . import translation_pb2_grpc
    __all__ = ["translation_pb2", "translation_pb2_grpc"]
except ImportError:
    # Proto files not yet generated
    __all__ = []
