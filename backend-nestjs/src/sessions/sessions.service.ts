import { Injectable, NotFoundException } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import { MlClientService } from '../ml-client/ml-client.service';

export interface Session {
  sessionId: string;
  status:
    | 'uploaded'
    | 'processing'
    | 'completed'
    | 'failed'
    | 'paused'
    | 'cancelled';
  progress: number;
  currentStep: string;
  message?: string;
  sourceLang: string;
  targetLang: string;
  filePath: string;
  outputPath?: string;
  earlyPreviewAvailable?: boolean;
  earlyPreviewPath?: string;
  createdAt: Date;
  updatedAt: Date;
  fileName?: string;
  fileSize?: number;
  duration?: number;
  metadata?: {
    duration: number;
    resolution: string;
    codec: string;
    size: number;
    name: string;
    earlyPreviewAvailable?: boolean;
    earlyPreviewPath?: string;
    outputPath?: string;
  };
  result?: any;
  // Processing fields
  totalChunks?: number;
  currentChunk?: number;
  etaSeconds?: number;
  processingSpeed?: number;
  hardwareInfo?: {
    cpu: string;
    gpu: string;
    vram_gb: number;
    ram_gb: number;
  };
  availableSegments?: string[];
  isPaused?: boolean;
  logs?: Array<{
    timestamp: string;
    level: string;
    stage: string;
    message: string;
    chunkId?: string;
    sessionId?: string;
    extraData?: any;
  }>;
  // Detailed progress fields (from backend)
  stage?: string;
  stage_number?: number;
  total_stages?: number;
  segments_processed?: number;
  current_time?: number;
  current_time_formatted?: string;
  total_duration?: number;
  total_duration_formatted?: string;
  progress_percent?: number;
  elapsed_time?: number;
}

@Injectable()
export class SessionsService {
  #sessions = new Map<string, Session>();

  constructor(private readonly mlClient?: MlClientService) {
    // No test sessions - only real uploaded videos
  }

  createSession(data: Partial<Session>): Session {
    const sessionId = uuidv4();
    console.log('Creating session with data:', {
      sourceLang: data.sourceLang,
      targetLang: data.targetLang,
    });
    const session: Session = {
      sessionId,
      status: 'uploaded',
      progress: 0,
      currentStep: 'Uploaded',
      sourceLang: data.sourceLang ?? 'en',
      targetLang: data.targetLang ?? 'hy', // Default to Armenian
      filePath: data.filePath ?? '',
      fileName: data.fileName,
      fileSize: data.fileSize,
      createdAt: new Date(),
      updatedAt: new Date(),
      totalChunks: 0, // Initialize to 0, will be updated by gRPC stream
      currentChunk: 0,
    };
    console.log('Created session with languages:', {
      sourceLang: session.sourceLang,
      targetLang: session.targetLang,
    });

    this.#sessions.set(sessionId, session);
    return structuredClone(session);
  }

  findOne(sessionId: string): Session {
    const session = this.#sessions.get(sessionId);
    if (!session) {
      throw new NotFoundException(`Session ${sessionId} not found`);
    }
    return structuredClone(session);
  }

  updateProgress(sessionId: string, progress: Partial<Session>): void {
    const session = this.#sessions.get(sessionId);
    if (!session) {
      return;
    }

    // Handle logs: append new logs, maintain max 200 logs
    if (progress.logs && progress.logs.length > 0) {
      const existingLogs = session.logs || [];
      const newLogs = progress.logs;
      
      console.log(`💾 [SESSIONS SERVICE] Storing ${newLogs.length} new logs for session ${sessionId} (existing: ${existingLogs.length})`);
      
      // Merge logs, avoiding duplicates (by timestamp + message)
      const logMap = new Map<string, typeof newLogs[0]>();
      
      // Add existing logs
      existingLogs.forEach(log => {
        const key = `${log.timestamp}-${log.message}`;
        logMap.set(key, log);
      });
      
      // Add new logs
      newLogs.forEach(log => {
        const key = `${log.timestamp}-${log.message}`;
        logMap.set(key, log);
      });
      
      // Convert back to array, sort by timestamp, limit to 200
      const allLogs = Array.from(logMap.values())
        .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
        .slice(-200);
      
      console.log(`💾 [SESSIONS SERVICE] Session ${sessionId} now has ${allLogs.length} total logs`);
      progress.logs = allLogs;
    }

    // Handle nested result object properly
    if (progress.result && session.result) {
      progress.result = { ...session.result, ...progress.result };
    }

    // Preserve existing values if not provided in progress update
    const preservedFields = {
      totalChunks:
        progress.totalChunks !== undefined
          ? progress.totalChunks
          : session.totalChunks,
      currentChunk:
        progress.currentChunk !== undefined
          ? progress.currentChunk
          : session.currentChunk,
      availableSegments:
        progress.availableSegments || session.availableSegments,
      hardwareInfo: progress.hardwareInfo || session.hardwareInfo,
    };

    Object.assign(session, progress, preservedFields, {
      updatedAt: new Date(),
    });
  }

  calculateProcessingTime(session: Session): string {
    if (!session.createdAt) return '0s';
    const startTime = new Date(session.createdAt).getTime();
    const endTime = new Date().getTime();
    const durationMs = endTime - startTime;
    const seconds = Math.floor(durationMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  }

  findAll(): Session[] {
    return [...this.#sessions.values()].map((s) => structuredClone(s));
  }

  delete(sessionId: string): void {
    if (!this.#sessions.delete(sessionId)) {
      throw new NotFoundException(`Session ${sessionId} not found`);
    }
  }

  exists(sessionId: string): boolean {
    return this.#sessions.has(sessionId);
  }

  getMLClient(): MlClientService | undefined {
    return this.mlClient;
  }
}
