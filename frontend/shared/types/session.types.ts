export type SessionStatus =
  | "uploaded"
  | "processing"
  | "completed"
  | "failed"
  | "paused"
  | "cancelled";

export type ProcessingSession = {
  sessionId: string;
  status: SessionStatus;
  progress: number;
  currentStep: string;
  message: string;
  eta?: string;
  sourceLang: string;
  targetLang: string;

  // File paths (backend returns these directly)
  filePath?: string;
  fileName?: string;
  fileSize?: number;
  outputPath?: string; // ADD THIS - direct output path from backend

  // Preview fields
  early_preview_available?: boolean;
  early_preview_path?: string;

  // Processing fields
  isPaused?: boolean;
  availableSegments?: string[];
  hardwareInfo?: HardwareInfo;
  etaSeconds?: number;
  processingSpeed?: number; // chunks per minute
  currentChunk?: number;
  totalChunks?: number;
  duration?: number;
  
  // Detailed progress fields (from backend)
  segments_processed?: number;
  current_time?: number;
  current_time_formatted?: string;
  total_duration?: number;
  total_duration_formatted?: string;
  progress_percent?: number;
  elapsed_time?: number;
  stage?: string;
  stage_number?: number;
  total_stages?: number;
  stage_progress_percent?: number;

  // File metadata
  metadata?: VideoMetadata;

  // Result data (when completed)
  result?: {
    originalDuration: number;
    finalDuration: number;
    outputPath: string;
    qualityMetrics?: {
      durationMatch: boolean;
      syncAccuracy: number;
      voiceQuality: number;
      translationQuality: string;
    };
    processingTime?: string | number;
    processingTimeSeconds?: number;
    outputSize?: number;
    segments?: Array<{
      id: string;
      startTime: number;
      endTime: number;
      originalText: string;
      translatedText: string;
      audioPath: string;
    }>;
  };

  // Timestamps
  createdAt?: Date;
  updatedAt?: Date;
  startedAt?: Date;
  completedAt?: Date;

  // Logs for real-time display
  logs?: Array<{
    id: string;
    timestamp: string;
    level: "info" | "warning" | "error" | "debug" | "success";
    stage: string;
    message: string;
    chunkId?: string;
    sessionId?: string;
    data?: any;
  }>;
};

export type HardwareInfo = {
  cpu: string;
  gpu: string;
  vram_gb: number;
  ram_gb: number;
};

import type { VideoMetadata } from "./video.types";

export type SessionProgress = {
  sessionId: string;
  progress: number;
  currentStep: string;
  status: SessionStatus;
  message: string;
  eta?: string;
  earlyPreviewAvailable?: boolean;
  earlyPreviewPath?: string;
  isPaused?: boolean;
  availableSegments?: string[];
  hardwareInfo?: HardwareInfo;
  etaSeconds?: number;
  processingSpeed?: number;
  currentChunk?: number;
  totalChunks?: number;
};
