import { NestFactory } from '@nestjs/core';
import { ValidationPipe, Logger } from '@nestjs/common';
import { AppModule } from './app.module';
import compression from 'compression';
import helmet from 'helmet';
import { NestExpressApplication } from '@nestjs/platform-express';
import { join } from 'path';

async function bootstrap() {
  const logger = new Logger('Bootstrap');

  const app = await NestFactory.create<NestExpressApplication>(AppModule, {
    logger: ['error', 'warn', 'log', 'debug'],
  });

  // Security
  app.use(
    helmet({
      crossOriginResourcePolicy: { policy: 'cross-origin' },
    }),
  );

  // Compression
  app.use(compression());

  // CORS - allow both Docker internal and localhost access
  app.enableCors({
    origin: true, // Allow all origins for development
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    exposedHeaders: ['Content-Disposition'],
  });

  // Global validation pipe
  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      forbidNonWhitelisted: true,
      transform: true,
    }),
  );

  // Create uploads directory in .data folder to consolidate data
  const fs = require('fs');
  const path = require('path');
  const projectRoot = path.resolve(__dirname, '..', '..');
  const dataDir = path.join(projectRoot, '.data');
  const uploadsDir = path.join(dataDir, 'uploads');
  
  // Create .data and uploads directories if they don't exist
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }

  // Serve static files from uploads directory with CORS headers
  app.useStaticAssets(uploadsDir, {
    prefix: '/uploads/',
    setHeaders: (res, path) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
      res.setHeader(
        'Access-Control-Expose-Headers',
        'Content-Length, Content-Range',
      );
    },
  });

  const port = process.env.PORT ?? 3001;
  await app.listen(port, '0.0.0.0');

  logger.log(`🚀 NestJS API Gateway running on: http://localhost:${port}`);
  logger.log(
    `🔗 gRPC ML Service URL: ${process.env.GRPC_ML_SERVICE_URL ?? 'localhost:50051'}`,
  );
}

bootstrap();
