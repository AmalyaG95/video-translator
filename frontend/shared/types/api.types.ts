
export type ApiResponse<T> = {
  data: T;
  success: boolean;
  message?: string;
};

export type ApiError = {
  message: string;
  code: string;
  details?: unknown;
};

export type TranslationRequest = {
  file: File;
  sourceLang: string;
  targetLang: string;
  qualityPreset?: string;
  preserveTiming?: boolean;
  lipSyncEnabled?: boolean;
};

export type UploadResponse = {
  sessionId: string;
  fileName: string;
  fileSize: number;
  message: string;
};

export type ControlTranslationRequest = {
  action: "pause" | "resume" | "cancel";
};

export type ControlTranslationResponse = {
  success: boolean;
  isPaused: boolean;
  message: string;
};

export type LanguageDetectionRequest = {
  file: File;
};

export type LanguageDetectionResponse = {
  language: string;
  confidence: number;
};

export type Language = {
  code: string;
  name: string;
  nativeName: string;
  flag: string;
};

export type AIInsight = {
  id: string;
  timestamp: string;
  stage: string;
  type: "info" | "warning" | "decision" | "error";
  message: string;
  data?: unknown;
};

export type LogEntry = {
  timestamp: string;
  level: "info" | "warn" | "error" | "debug";
  stage: string;
  message: string;
  data?: unknown;
};
