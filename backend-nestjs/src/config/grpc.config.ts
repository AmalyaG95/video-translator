import { ClientOptions, Transport } from '@nestjs/microservices';
import { join } from 'path';

// Time constants (in milliseconds)
const SECONDS_MS = 1000;
const MINUTES_MS = 60 * SECONDS_MS;
const HOURS_MS = 60 * MINUTES_MS;

export const grpcClientOptions: ClientOptions = {
  transport: Transport.GRPC,
  options: {
    url: process.env.GRPC_ML_SERVICE_URL ?? 'localhost:50051',
    package: 'translation',
    protoPath: join(__dirname, '../ml-client/proto/translation.proto'),
    loader: {
      keepCase: true,
      longs: String,
      enums: String,
      defaults: true,
      oneofs: true,
    },
    maxSendMessageLength: 50 * 1024 * 1024, // 50MB
    maxReceiveMessageLength: 50 * 1024 * 1024, // 50MB
    // Keep-alive settings to prevent connection drops
    // Less aggressive settings to prevent RESOURCE_EXHAUSTED errors
    keepalive: {
      keepaliveTimeMs: 60000,  // Increase to 60 seconds (less aggressive)
      keepaliveTimeoutMs: 10000,  // Increase timeout
      keepalivePermitWithoutCalls: 1,
      http2MaxPingsWithoutData: 0,  // Allow unlimited pings
      http2MinTimeBetweenPingsMs: 10000,  // Minimum 10s between pings
      http2MinPingIntervalWithoutDataMs: 30000,  // 30s minimum interval
    },
    // Channel options for long-running connections
    // Match server settings to prevent "max connection age" errors
    channelOptions: {
      'grpc.max_connection_idle_ms': 1 * HOURS_MS,  // 1 hour (for idle connections)
      'grpc.max_connection_age_ms': 24 * HOURS_MS,  // 24 hours (for long video translations)
      'grpc.max_connection_age_grace_ms': 1 * MINUTES_MS,  // 1 minute grace period
      'grpc.keepalive_time_ms': 30 * SECONDS_MS,  // Match server (30 seconds)
      'grpc.keepalive_timeout_ms': 10 * SECONDS_MS,  // Match server (10 seconds)
      'grpc.keepalive_permit_without_calls': 1,
    },
  },
};
