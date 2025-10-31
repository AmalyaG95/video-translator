import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
  Logger,
} from '@nestjs/common';
import { Observable } from 'rxjs';
import { tap, catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';
import { Request, Response } from 'express';
import { LoggingConfig } from './logging.config';

@Injectable()
export class LoggingInterceptor implements NestInterceptor {
  private readonly logger = new Logger(LoggingInterceptor.name);
  private readonly loggingConfig = LoggingConfig.getInstance();

  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    const request = context.switchToHttp().getRequest<Request>();
    const response = context.switchToHttp().getResponse<Response>();
    const { method, url, ip, headers } = request;
    const userAgent = headers['user-agent'] || 'Unknown';
    const startTime = Date.now();

    // Log request start
    this.logger.log(`📥 ${method} ${url} - ${ip} - ${userAgent}`);

    return next.handle().pipe(
      tap((data) => {
        const responseTime = Date.now() - startTime;
        const { statusCode } = response;

        // Log successful response
        this.loggingConfig.logRequest(method, url, statusCode, responseTime);

        // Log specific events based on endpoint
        if (url.includes('/upload')) {
          this.logger.log(`📁 File upload completed: ${responseTime}ms`);
        } else if (url.includes('/translate')) {
          this.logger.log(
            `🔄 Translation request processed: ${responseTime}ms`,
          );
        } else if (url.includes('/sessions')) {
          this.logger.log(`📊 Session operation completed: ${responseTime}ms`);
        } else if (url.includes('/detect-language')) {
          this.logger.log(`🌐 Language detection completed: ${responseTime}ms`);
        }
      }),
      catchError((error) => {
        const responseTime = Date.now() - startTime;
        const { statusCode } = response;

        // Log error response
        this.logger.error(
          `❌ ${method} ${url} - ${statusCode} - ${responseTime}ms - ${error.message}`,
        );

        // Log specific error types
        if (error.status === 400) {
          this.logger.error(`🚫 Bad Request: ${error.message}`);
        } else if (error.status === 401) {
          this.logger.error(`🔒 Unauthorized: ${error.message}`);
        } else if (error.status === 403) {
          this.logger.error(`🚫 Forbidden: ${error.message}`);
        } else if (error.status === 404) {
          this.logger.error(`🔍 Not Found: ${error.message}`);
        } else if (error.status === 500) {
          this.logger.error(`💥 Internal Server Error: ${error.message}`);
        }

        return throwError(() => error);
      }),
    );
  }
}











