import type { VideoFile, VideoMetadata } from "@/shared/types";

// Re-export for local use
export type { VideoMetadata };

export type FileUploadProps = {
  onFileSelect: (file: File) => void;
  maxSize?: number;
  onMetadataDetected?: (metadata: VideoMetadata) => void;
  onLanguageDetected?: (language: string, confidence: number) => void;
  onFileUnload?: () => void;
};

export type LanguageSelectorProps = {
  languages: readonly Language[];
  selectedLanguage: string;
  onLanguageChange: (language: string) => void;
  disabled?: boolean;
  label?: string;
  value?: string;
  onChange?: (language: string) => void;
  disabledLanguages?: string[];
};

export type Language = {
  code: string;
  name: string;
  nativeName: string;
  flag: string;
};

export type QualitySelectorProps = {
  selectedQuality: string;
  onQualityChange: (quality: string) => void;
  disabled?: boolean;
  value?: string;
  onChange?: (quality: string) => void;
  preserveTiming?: boolean;
  onPreserveTimingChange?: (enabled: boolean) => void;
  lipSyncEnabled?: boolean;
  onLipSyncEnabledChange?: (enabled: boolean) => void;
};

export type VideoMetadataCardProps = {
  metadata: VideoMetadata | null;
  fileName: string;
  fileSize: number;
  onUnload?: () => void;
  onPreview?: () => void;
};

export type UploadResponse = {
  sessionId: string;
  status: string;
  progress: number;
  currentStep: string;
  message: string;
  sourceLang: string;
  targetLang: string;
  filePath: string;
  fileName: string;
  fileSize: number;
  createdAt: string;
  updatedAt: string;
  totalChunks?: number;
  currentChunk?: number;
  etaSeconds?: number;
  processingSpeed?: number;
  hardwareInfo?: any;
  availableSegments?: string[];
  early_preview_available?: boolean;
  early_preview_path?: string;
  isPaused?: boolean;
};
