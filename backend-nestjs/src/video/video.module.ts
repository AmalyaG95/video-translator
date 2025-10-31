import { Module } from '@nestjs/common';
import { VideoService } from './video.service';
import { VideoController } from './video.controller';
import { SessionsModule } from '../sessions/sessions.module';
import { MlClientModule } from '../ml-client/ml-client.module';

@Module({
  imports: [SessionsModule, MlClientModule],
  providers: [VideoService],
  controllers: [VideoController],
  exports: [VideoService],
})
export class VideoModule {}
