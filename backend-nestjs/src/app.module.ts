import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ThrottlerModule } from '@nestjs/throttler';
import { SessionsModule } from './sessions/sessions.module';
import { VideoModule } from './video/video.module';
import { MlClientModule } from './ml-client/ml-client.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
    ThrottlerModule.forRoot([
      {
        ttl: 60000,
        limit: 100,
      },
    ]),
    SessionsModule,
    VideoModule,
    MlClientModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
