import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import { Client } from '@nestjs/microservices';
import type { ClientGrpc } from '@nestjs/microservices';
import { grpcClientOptions } from '../config/grpc.config';
import { Observable, lastValueFrom } from 'rxjs';

interface TranslationService {
  translateVideo(data: any): Observable<any>;
  getTranslationResult(data: any): Observable<any>;
  cancelTranslation(data: any): Observable<any>;
  getAiInsights(data: any): Observable<any>;
  controlTranslation(data: any): Observable<any>;
}

@Injectable()
export class MlClientService implements OnModuleInit {
  @Client(grpcClientOptions)
  private client: ClientGrpc;

  private translationService: TranslationService;
  private readonly logger = new Logger(MlClientService.name);

  onModuleInit() {
    try {
      this.translationService =
        this.client.getService<TranslationService>('TranslationService');
      this.logger.log('gRPC client initialized');
    } catch (error) {
      this.logger.error(`Failed to initialize gRPC client: ${error.message}`);
      throw error;
    }
  }

  translateVideo(
    sessionId: string,
    filePath: string,
    sourceLang: string,
    targetLang: string,
  ): Observable<any> {
    this.logger.log(`Starting translation for session: ${sessionId}`);

    if (!this.translationService) {
      this.logger.error('TranslationService not initialized!');
      throw new Error(
        'TranslationService not initialized. Check gRPC connection.',
      );
    }

    return this.translationService.translateVideo({
      session_id: sessionId,
      file_path: filePath,
      source_lang: sourceLang,
      target_lang: targetLang,
    });
  }

  async getResult(sessionId: string): Promise<any> {
    this.logger.log(`Getting result for session: ${sessionId}`);

    try {
      return await lastValueFrom(
        this.translationService.getTranslationResult({
          session_id: sessionId,
        }),
      );
    } catch (error) {
      this.logger.error(`Failed to get result: ${error.message}`);
      throw error;
    }
  }

  async cancelTranslation(sessionId: string): Promise<{ success: boolean }> {
    this.logger.log(`Canceling session: ${sessionId}`);

    try {
      const response = await lastValueFrom(
        this.translationService.cancelTranslation({
          session_id: sessionId,
        }),
      );
      return { success: response.success ?? false };
    } catch (error) {
      this.logger.error(`Failed to cancel: ${error.message}`);
      throw error;
    }
  }

  async controlTranslation(
    sessionId: string,
    action: 'pause' | 'resume' | 'cancel',
  ): Promise<{ success: boolean }> {
    this.logger.log(
      `Controlling translation for session ${sessionId}: ${action}`,
    );

    try {
      const response = await lastValueFrom(
        this.translationService.controlTranslation({
          session_id: sessionId,
          action: action,
        }),
      );
      return { success: response.success ?? false };
    } catch (error) {
      this.logger.error(`Failed to control translation: ${error.message}`);
      throw error;
    }
  }

  async pauseTranslation(sessionId: string): Promise<{ success: boolean }> {
    return this.controlTranslation(sessionId, 'pause');
  }

  async resumeTranslation(sessionId: string): Promise<{ success: boolean }> {
    return this.controlTranslation(sessionId, 'resume');
  }

  async getAIInsights(sessionId: string): Promise<any> {
    this.logger.log(`Getting AI insights for session: ${sessionId}`);

    try {
      const response = await lastValueFrom(
        this.translationService.getAiInsights({
          session_id: sessionId,
        }),
      );
      return response;
    } catch (error) {
      this.logger.error(`Failed to get AI insights: ${error.message}`);
      throw error;
    }
  }
}
