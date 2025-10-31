import { ClientOptions, Transport } from '@nestjs/microservices';
import { join } from 'path';

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
  },
};
