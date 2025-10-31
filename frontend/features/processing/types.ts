
import type { ProcessingStep } from "./components/StepBadges";

export type StepBadgesProps = {
  steps: ProcessingStep[];
  currentStep?: string;
};

export type UnifiedProcessingTimelineProps = {
  session: {
    sessionId: string;
    status: string;
    progress: number;
    currentStep: string;
    isPaused?: boolean;
    currentChunk?: number;
    totalChunks?: number;
    processingSpeed?: number;
  };
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
  onViewResults: () => void;
};

export type DynamicETAProps = {
  etaSeconds?: number;
  hardwareInfo?: {
    cpu: string;
    gpu: string;
    vram_gb: number;
    ram_gb: number;
  };
  processingSpeed?: number;
  currentChunk?: number;
  totalChunks?: number;
};

export type AIReasoningLog = {
  id: string;
  timestamp: string;
  stage: string;
  type: "info" | "warning" | "decision" | "error";
  message: string;
  data?: unknown;
};

export type AIReasoningPanelProps = {
  logs: AIReasoningLog[];
  className?: string;
};
