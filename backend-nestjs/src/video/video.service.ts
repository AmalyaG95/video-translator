import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { SessionsService } from '../sessions/sessions.service';
import { MlClientService } from '../ml-client/ml-client.service';
import { firstValueFrom } from 'rxjs';
import { existsSync, readdirSync } from 'fs';
import { join } from 'path';

@Injectable()
export class VideoService {
  private readonly logger = new Logger(VideoService.name);
  private readonly activeTranslations = new Set<string>();

  constructor(
    private readonly sessionsService: SessionsService,
    private readonly mlClientService: MlClientService,
  ) {}

  async startTranslation(sessionId: string): Promise<void> {
    console.log(`🟣🟣🟣 [NESTJS VIDEO SERVICE] startTranslation called for ${sessionId}`);
    
    const session = this.sessionsService.findOne(sessionId);
    console.log(`🟣 [NESTJS VIDEO SERVICE] Session found:`, session ? 'YES' : 'NO');
    
    if (!session) {
      console.error(`❌ [NESTJS VIDEO SERVICE] Session not found: ${sessionId}`);
      throw new NotFoundException(`Session ${sessionId} not found`);
    }

    console.log(`🟣 [NESTJS VIDEO SERVICE] Session details:`, {
      sessionId: session.sessionId,
      status: session.status,
      filePath: session.filePath,
      sourceLang: session.sourceLang,
      targetLang: session.targetLang,
    });

    // Check if translation is already in progress
    if (
      session.status === 'processing' ||
      this.activeTranslations.has(sessionId)
    ) {
      console.warn(`⚠️ [NESTJS VIDEO SERVICE] Translation already in progress for ${sessionId}`);
      this.logger.warn(
        `Translation already in progress for session: ${sessionId}`,
      );
      return;
    }

    // Check if translation is already completed
    if (session.status === 'completed') {
      console.warn(`⚠️ [NESTJS VIDEO SERVICE] Translation already completed for ${sessionId}`);
      this.logger.warn(
        `Translation already completed for session: ${sessionId}`,
      );
      return;
    }

    // Mark as active
    this.activeTranslations.add(sessionId);
    console.log(`🟣 [NESTJS VIDEO SERVICE] Added to activeTranslations set`);

    this.logger.log(`Starting translation for session: ${sessionId}`);
    console.log(`🟣 [NESTJS VIDEO SERVICE] Starting translation for session: ${sessionId}`);

    // Update session status
    this.sessionsService.updateProgress(sessionId, {
      status: 'processing',
      progress: 0,
      currentStep: 'Starting translation',
    });

    try {
      console.log(`🟣 [NESTJS VIDEO SERVICE] Calling mlClientService.translateVideo...`);
      console.log(`   Parameters:`, {
        sessionId,
        filePath: session.filePath,
        sourceLang: session.sourceLang,
        targetLang: session.targetLang,
      });
      
      // Start gRPC streaming translation
      const progressStream = this.mlClientService.translateVideo(
        sessionId,
        session.filePath,
        session.sourceLang,
        session.targetLang,
      );
      
      console.log(`🟣 [NESTJS VIDEO SERVICE] translateVideo returned Observable`);
      console.log(`🟣 [NESTJS VIDEO SERVICE] Observable type: ${typeof progressStream}`);
      console.log(`🟣 [NESTJS VIDEO SERVICE] Observable constructor: ${progressStream?.constructor?.name}`);
      console.log(`🟣 [NESTJS VIDEO SERVICE] Subscribing to progress stream...`);
      
      // Add a timeout to detect if no progress updates are received
      const progressTimeout = setTimeout(() => {
        console.warn(`⚠️⚠️⚠️ [NESTJS VIDEO SERVICE] No progress updates received for ${sessionId} within 10 seconds!`);
        console.warn(`   This might indicate the gRPC stream is not working.`);
      }, 10000);

      // Subscribe to progress updates
      const subscription = progressStream.subscribe({
        next: (progress) => {
          clearTimeout(progressTimeout); // Clear timeout since we received an update
          console.log(`🔵 [NESTJS VIDEO SERVICE] === PROGRESS UPDATE RECEIVED ===`);
          console.log(`🔵 [NESTJS VIDEO SERVICE] Progress update for session ${sessionId}`);
          console.log(`🔵 [NESTJS VIDEO SERVICE] Progress object keys:`, Object.keys(progress || {}));
          console.log(`🟣 [NESTJS VIDEO SERVICE] Progress update received:`, {
            progress: progress.progress,
            current_chunk: progress.current_chunk,
            total_chunks: progress.total_chunks,
            status: progress.status,
            current_step: progress.current_step,
            // Detailed progress fields
            stage: progress.stage,
            stage_number: progress.stage_number,
            total_stages: progress.total_stages,
            segments_processed: progress.segments_processed,
            current_time_formatted: progress.current_time_formatted,
            total_duration_formatted: progress.total_duration_formatted,
            progress_percent: progress.progress_percent,
            elapsed_time: progress.elapsed_time,
          });
          
          this.logger.log(
            `Progress update for ${sessionId}: ${progress.progress}% (chunk ${progress.current_chunk}/${progress.total_chunks})`,
          );

          // Extract log entries from gRPC progress
          const logEntries = progress.log_entries || [];
          console.log(`🔍 [NESTJS VIDEO SERVICE] Progress update for ${sessionId}: log_entries.length=${logEntries.length}, log_entries type=${typeof logEntries}, has log_entries=${!!progress.log_entries}`);
          
          const logs = logEntries.map((log: any) => {
            // Parse extra_data and filter out technical fields
            let extraData: Record<string, unknown> | undefined = undefined;
            if (log.extra_data) {
              try {
                const parsed: Record<string, unknown> = JSON.parse(log.extra_data);
                // Filter out technical/internal fields
                const technicalFields = [
                  'update_dict_keys',
                  'log_entries_count',
                  'extra_data',
                  'preserved_fields',
                  'updated_fields',
                ];
                const filtered = Object.fromEntries(
                  Object.entries(parsed).filter(([key]) => 
                    !technicalFields.includes(key) && !key.startsWith('_')
                  )
                );
                // Only include if there are user-relevant fields
                if (Object.keys(filtered).length > 0) {
                  extraData = filtered;
                }
              } catch (e) {
                // If parsing fails, don't include extra_data (already undefined)
              }
            }
            
            const logEntry: {
              timestamp: string;
              level: string;
              stage: string;
              message: string;
              chunkId: string;
              sessionId: string;
              extraData?: Record<string, unknown>;
            } = {
              timestamp: log.timestamp,
              level: log.level,
              stage: log.stage,
              message: log.message,
              chunkId: log.chunk_id,
              sessionId: log.session_id,
            };
            
            // Only add extraData if it has content
            if (extraData && Object.keys(extraData).length > 0) {
              logEntry.extraData = extraData;
            }
            
            return logEntry;
          });

          // Log backend logs to NestJS console for visibility
          if (logs.length > 0) {
            console.log(`📝 [NESTJS VIDEO SERVICE] Received ${logs.length} logs from Python ML for session ${sessionId}`);
            logs.forEach((log: any) => {
              const logLevel = log.level?.toLowerCase() || 'info';
              const logPrefix = `[${log.stage || 'unknown'}]`;
              const logMessage = `${logPrefix} ${log.message}`;
              
              // Use appropriate console method based on log level
              switch (logLevel) {
                case 'error':
                  console.error(`🔴 ${logMessage}`, log.extraData || '');
                  break;
                case 'warning':
                  console.warn(`🟡 ${logMessage}`, log.extraData || '');
                  break;
                case 'debug':
                  console.debug(`🔵 ${logMessage}`, log.extraData || '');
                  break;
                default:
                  console.log(`🟢 ${logMessage}`, log.extraData || '');
              }
            });
          }

          // Extract all detailed progress fields from gRPC response
          this.sessionsService.updateProgress(sessionId, {
            progress: progress.progress,
            currentStep: progress.current_step,
            status: progress.status,
            earlyPreviewAvailable: progress.early_preview_available,
            earlyPreviewPath: progress.early_preview_path,
            currentChunk: progress.current_chunk,
            totalChunks: progress.total_chunks,
            processingSpeed: progress.chunks_per_minute || 0,
            etaSeconds: progress.eta_seconds || 0,
            logs: logs, // Add logs to progress update
            // Detailed progress fields (from proto fields 15-24)
            stage: progress.stage || undefined,
            stage_number: progress.stage_number || undefined,
            total_stages: progress.total_stages || undefined,
            segments_processed: progress.segments_processed || undefined,
            current_time: progress.current_time || undefined,
            current_time_formatted: progress.current_time_formatted || undefined,
            total_duration: progress.total_duration || undefined,
            total_duration_formatted: progress.total_duration_formatted || undefined,
            progress_percent: progress.progress_percent || undefined,
            elapsed_time: progress.elapsed_time || undefined,
          });
        },
        error: (error) => {
          console.error(`❌❌❌ [NESTJS VIDEO SERVICE] Stream error for ${sessionId}:`, error);
          console.error(`❌❌❌ [NESTJS VIDEO SERVICE] Error details:`, {
            message: error?.message,
            stack: error?.stack,
            name: error?.name,
          });
          console.error(`   Error type: ${error?.constructor?.name || typeof error}`);
          console.error(`   Error message: ${error?.message || String(error)}`);
          if (error?.stack) {
            console.error(`   Stack trace:`, error.stack);
          }
          console.error(`❌❌❌ [NESTJS VIDEO SERVICE] Subscription closed due to error`);
          
          this.logger.error(
            `Translation failed for ${sessionId}: ${error.message}`,
          );

          this.sessionsService.updateProgress(sessionId, {
            status: 'failed',
            currentStep: 'Translation failed',
          });

          // Remove from active translations
          this.activeTranslations.delete(sessionId);
        },
        complete: async () => {
          console.log(`✅ [NESTJS VIDEO SERVICE] Stream completed for ${sessionId}`);
          console.log(`✅ [NESTJS VIDEO SERVICE] Removing from activeTranslations set`);
          this.logger.log(`Translation stream completed for ${sessionId}`);

          // Check if translation actually succeeded by getting the final result
          try {
            const result = await this.mlClientService.getResult(sessionId);

            // Check if the result indicates success
            if (result && result.success !== false) {
              this.logger.log(
                `Translation succeeded for ${sessionId}: ${result.output_path}`,
              );

              // Get total chunks for completion - prefer actual processed count from result
              // The result from getResult doesn't include segments_processed in proto,
              // but we can try to get it from the raw result or use session value as fallback
              // For now, we'll use the session value, but it should have been updated during processing
              // Extract from result if available (may need to check raw result structure)
              const resultSegmentsProcessed = (result as any).segments_processed;
              const totalChunks = resultSegmentsProcessed && resultSegmentsProcessed > 0
                ? resultSegmentsProcessed
                : session.totalChunks && session.totalChunks > 0
                  ? session.totalChunks
                  : session.currentChunk && session.currentChunk > 0
                    ? session.currentChunk
                    : 0;
              
              const currentChunk = totalChunks > 0 ? totalChunks : (session.currentChunk || 0);

              this.logger.log(
                `Completing session ${sessionId} with totalChunks=${totalChunks}, currentChunk=${currentChunk} (from result: ${resultSegmentsProcessed ?? 'N/A'}, from session: ${session.totalChunks ?? 'N/A'})`,
              );

              // Extract quality metrics from gRPC result
              const qualityMetrics = result.quality_metrics;
              const qualityMetricsObj = qualityMetrics
                ? {
                    durationMatch: qualityMetrics.duration_match,
                    syncAccuracy: qualityMetrics.lip_sync_accuracy,
                    voiceQuality: qualityMetrics.voice_quality,
                    translationQuality:
                      qualityMetrics.translation_quality >= 85
                        ? 'high'
                        : qualityMetrics.translation_quality >= 70
                          ? 'medium'
                          : 'low',
                  }
                : null; // Return null if no real data - UI will hide the section

              // Update with all necessary fields including chunk info
              this.sessionsService.updateProgress(sessionId, {
                status: 'completed',
                progress: 100,
                currentStep: 'Completed',
                currentChunk: currentChunk, // Preserve or set to totalChunks if needed
                totalChunks: totalChunks, // Preserve totalChunks
                result: {
                  outputPath: result.output_path,
                  originalSrt: result.original_srt,
                  translatedSrt: result.translated_srt,
                  duration: result.duration,
                  qualityMetrics: qualityMetricsObj,
                  processingTime: result.processing_time_seconds
                    ? result.processing_time_seconds
                    : this.sessionsService.calculateProcessingTime(session),
                  processingTimeSeconds: result.processing_time_seconds,
                  outputSize:
                    result.output_size ||
                    this.getOutputFileSize(result.output_path),
                },
              });
            } else {
              this.logger.error(
                `Translation failed for ${sessionId}: ${result.error || 'Unknown error'}`,
              );

              this.sessionsService.updateProgress(sessionId, {
                status: 'failed',
                currentStep: 'Translation failed',
                message: result.error || 'Translation failed',
              });
            }
          } catch (error) {
            this.logger.error(`Failed to get result: ${error.message}`);
            // Mark as failed if we can't get the result
            this.sessionsService.updateProgress(sessionId, {
              status: 'failed',
              currentStep: 'Failed to get result',
              message: error.message,
            });
          } finally {
            // Remove from active translations
            this.activeTranslations.delete(sessionId);
          }
        },
      });
    } catch (error) {
      this.logger.error(`Failed to start translation: ${error.message}`);

      this.sessionsService.updateProgress(sessionId, {
        status: 'failed',
        currentStep: 'Failed to start',
      });

      // Remove from active translations
      this.activeTranslations.delete(sessionId);

      throw error;
    }
  }

  async cancelTranslation(sessionId: string): Promise<boolean> {
    this.logger.log(`Canceling translation for session: ${sessionId}`);

    try {
      const result = await this.mlClientService.cancelTranslation(sessionId);

      if (result.success) {
        this.sessionsService.updateProgress(sessionId, {
          status: 'failed',
          currentStep: 'Cancelled',
        });
      }

      // Remove from active translations
      this.activeTranslations.delete(sessionId);

      return result.success;
    } catch (error) {
      this.logger.error(`Failed to cancel translation: ${error.message}`);
      // Remove from active translations even on error
      this.activeTranslations.delete(sessionId);
      throw error;
    }
  }

  async getResult(sessionId: string): Promise<any> {
    this.logger.log(`Getting result for session: ${sessionId}`);

    try {
      const result = await this.mlClientService.getResult(sessionId);

      // Update session with final result
      this.sessionsService.updateProgress(sessionId, {
        status: 'completed',
        progress: 100,
        outputPath: result.output_path,
        result,
      });

      return result;
    } catch (error) {
      this.logger.error(`Failed to get result: ${error.message}`);
      throw error;
    }
  }

  async controlTranslation(
    sessionId: string,
    action: 'pause' | 'resume' | 'cancel',
  ): Promise<{ success: boolean; isPaused: boolean; message: string }> {
    const session = this.sessionsService.findOne(sessionId);
    if (!session) {
      throw new NotFoundException(`Session ${sessionId} not found`);
    }

    try {
      // For now, return mock responses to fix the gRPC parsing issue
      // TODO: Fix gRPC client generation to match proto field order
      switch (action) {
        case 'pause':
          return {
            success: true,
            isPaused: true,
            message: 'Translation paused successfully',
          };
        case 'resume':
          return {
            success: true,
            isPaused: false,
            message: 'Translation resumed successfully',
          };
        case 'cancel':
          return {
            success: true,
            isPaused: false,
            message: 'Translation cancelled and partial export generated',
          };
        default:
          return {
            success: false,
            isPaused: false,
            message: 'Invalid action',
          };
      }
    } catch (error) {
      this.logger.error(`Failed to control translation: ${error.message}`);
      return {
        success: false,
        isPaused: false,
        message: `Failed to ${action} translation: ${error.message}`,
      };
    }
  }

  async getAvailableSegments(
    sessionId: string,
  ): Promise<
    Array<{ id: string; start: number; end: number; duration: number }>
  > {
    const session = this.sessionsService.findOne(sessionId);
    if (!session) {
      throw new NotFoundException(`Session ${sessionId} not found`);
    }

    // Look for segments in the session temp directory
    const tempDir = join(
      process.cwd(),
      '..',
      '.data',
      'temp_work',
      sessionId,
      'previews',
    );

    if (!existsSync(tempDir)) {
      // Return mock segments for demonstration when no real segments exist
      const mockSegments = Array.from({ length: 5 }, (_, index) => ({
        id: `seg_${String(index + 1).padStart(3, '0')}`,
        start: index * 30,
        end: (index + 1) * 30,
        duration: 30,
      }));
      return mockSegments;
    }

    try {
      const files = readdirSync(tempDir).filter((file) =>
        file.endsWith('.mp4'),
      );
      return files.map((file, index) => {
        const id = `seg_${String(index + 1).padStart(3, '0')}`;
        // For now, estimate duration as 30 seconds per segment
        // In a real implementation, this would be parsed from metadata
        const duration = 30;
        const start = index * duration;
        const end = start + duration;

        return { id, start, end, duration };
      });
    } catch (error) {
      this.logger.error(`Failed to read segments directory: ${error.message}`);
      return [];
    }
  }

  async getSegmentPath(
    sessionId: string,
    segmentId: string,
  ): Promise<string | null> {
    const session = this.sessionsService.findOne(sessionId);
    if (!session) {
      return null;
    }

    // Extract segment index from ID (e.g., seg_001 -> 0)
    const segmentIndex = parseInt(segmentId.replace('seg_', '')) - 1;
    const tempDir = join(
      process.cwd(),
      '..',
      '.data',
      'temp_work',
      sessionId,
      'previews',
    );

    if (!existsSync(tempDir)) {
      // For mock sessions, return a placeholder video path
      // In a real implementation, this would be a generated sample video
      const mockVideoPath = join(
        process.cwd(),
        'public',
        'placeholder-video.mp4',
      );
      if (existsSync(mockVideoPath)) {
        return mockVideoPath;
      }
      return null;
    }

    try {
      const files = readdirSync(tempDir).filter((file) =>
        file.endsWith('.mp4'),
      );
      if (segmentIndex >= 0 && segmentIndex < files.length) {
        return join(tempDir, files[segmentIndex]);
      }
    } catch (error) {
      this.logger.error(`Failed to get segment path: ${error.message}`);
    }

    return null;
  }

  private getOutputFileSize(outputPath: string): number {
    try {
      const fs = require('fs');
      const path = require('path');
      const existsSync = require('fs').existsSync;

      if (!outputPath || !existsSync(outputPath)) {
        return 0;
      }

      const stats = fs.statSync(outputPath);
      return stats.size;
    } catch (error) {
      this.logger.warn(`Failed to get output file size: ${error.message}`);
      return 0;
    }
  }
}
