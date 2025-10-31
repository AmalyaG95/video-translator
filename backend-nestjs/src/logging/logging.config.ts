import { Logger } from '@nestjs/common';

export class LoggingConfig {
  private static instance: LoggingConfig;
  private logger: Logger;

  private constructor() {
    this.logger = new Logger('LoggingConfig');
  }

  public static getInstance(): LoggingConfig {
    if (!LoggingConfig.instance) {
      LoggingConfig.instance = new LoggingConfig();
    }
    return LoggingConfig.instance;
  }

  public getLogLevel(): string {
    return process.env.LOG_LEVEL || 'info';
  }

  public getLogFormat(): 'json' | 'simple' {
    return (process.env.LOG_FORMAT as 'json' | 'simple') || 'json';
  }

  public isDevelopment(): boolean {
    return process.env.NODE_ENV === 'development';
  }

  public getLogFile(): string {
    return process.env.LOG_FILE || '../.data/logs/nestjs-api.log';
  }

  public shouldLogToConsole(): boolean {
    return process.env.LOG_TO_CONSOLE !== 'false';
  }

  public shouldLogToFile(): boolean {
    return process.env.LOG_TO_FILE !== 'false';
  }

  public getMaxLogFileSize(): string {
    return process.env.MAX_LOG_FILE_SIZE || '10MB';
  }

  public getMaxLogFiles(): number {
    return parseInt(process.env.MAX_LOG_FILES || '5');
  }

  public getLogRotationInterval(): string {
    return process.env.LOG_ROTATION_INTERVAL || 'daily';
  }

  public logServiceStart(serviceName: string, port: number): void {
    this.logger.log(`🚀 ${serviceName} started on port ${port}`);
  }

  public logServiceStop(serviceName: string): void {
    this.logger.log(`🛑 ${serviceName} stopped`);
  }

  public logRequest(
    method: string,
    url: string,
    statusCode: number,
    responseTime: number,
  ): void {
    const level = statusCode >= 400 ? 'error' : 'log';
    const message = `${method} ${url} ${statusCode} ${responseTime}ms`;

    if (level === 'error') {
      this.logger.error(message);
    } else {
      this.logger.log(message);
    }
  }

  public logError(error: Error, context?: string): void {
    this.logger.error(
      `❌ ${context || 'Error'}: ${error.message}`,
      error.stack,
    );
  }

  public logWarning(message: string, context?: string): void {
    this.logger.warn(`⚠️ ${context || 'Warning'}: ${message}`);
  }

  public logInfo(message: string, context?: string): void {
    this.logger.log(`ℹ️ ${context || 'Info'}: ${message}`);
  }

  public logDebug(message: string, context?: string): void {
    this.logger.debug(`🐛 ${context || 'Debug'}: ${message}`);
  }

  public logSuccess(message: string, context?: string): void {
    this.logger.log(`✅ ${context || 'Success'}: ${message}`);
  }

  public logSessionEvent(sessionId: string, event: string, data?: any): void {
    const message = `📊 Session ${sessionId}: ${event}`;
    if (data) {
      this.logger.log(`${message} - ${JSON.stringify(data)}`);
    } else {
      this.logger.log(message);
    }
  }

  public logTranslationProgress(
    sessionId: string,
    progress: number,
    step: string,
  ): void {
    this.logger.log(`🔄 Translation ${sessionId}: ${progress}% - ${step}`);
  }

  public logTranslationComplete(sessionId: string, duration: number): void {
    this.logger.log(`✅ Translation ${sessionId} completed in ${duration}ms`);
  }

  public logTranslationError(sessionId: string, error: string): void {
    this.logger.error(`❌ Translation ${sessionId} failed: ${error}`);
  }

  public logSSEConnection(
    sessionId: string,
    action: 'connected' | 'disconnected' | 'error',
  ): void {
    const emoji =
      action === 'connected' ? '🔗' : action === 'disconnected' ? '🔌' : '❌';
    this.logger.log(`${emoji} SSE ${action} for session ${sessionId}`);
  }

  public logGRPCCall(
    service: string,
    method: string,
    success: boolean,
    duration?: number,
  ): void {
    const emoji = success ? '✅' : '❌';
    const durationStr = duration ? ` (${duration}ms)` : '';
    this.logger.log(`${emoji} gRPC ${service}.${method}${durationStr}`);
  }

  public logFileUpload(
    fileName: string,
    fileSize: number,
    sessionId: string,
  ): void {
    this.logger.log(
      `📁 Upload: ${fileName} (${fileSize} bytes) -> Session ${sessionId}`,
    );
  }

  public logLanguageDetection(
    sessionId: string,
    language: string,
    confidence: number,
  ): void {
    this.logger.log(
      `🌐 Language detected for ${sessionId}: ${language} (${confidence}%)`,
    );
  }

  public logAIInsight(sessionId: string, insight: string, data?: any): void {
    const message = `🤖 AI Insight for ${sessionId}: ${insight}`;
    if (data) {
      this.logger.log(`${message} - ${JSON.stringify(data)}`);
    } else {
      this.logger.log(message);
    }
  }
}











