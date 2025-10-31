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
    processingTime?: string;
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
