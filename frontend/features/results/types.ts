
import type { Spot, Segment, QualityMetrics } from "@/shared/types";

export type ResultsScreenProps = {
  session: {
    sessionId: string;
    status: string;
    progress: number;
    currentStep: string;
    sourceLang: string;
    targetLang: string;
    fileName: string;
    fileSize: number;
    earlyPreviewAvailable: boolean;
    earlyPreviewPath: string;
    result?: {
      outputPath: string;
      originalSrt: string;
      translatedSrt: string;
      duration: number;
      qualityMetrics: QualityMetrics;
    };
  };
  onDownload: () => void;
  onRefineVoice?: () => void;
  onNewTranslation?: () => void;
  onBack?: () => void;
};

export type RandomSpotCheckerProps = {
  sessionId: string;
  spots: Spot[];
  onSpotSelect: (spot: Spot) => void;
  selectedSpot: number | null;
  availableSegments?: string[];
};

export type SegmentPreviewSelectorProps = {
  sessionId: string;
  availableSegments: string[];
  onSegmentSelect?: (segmentId: string) => void;
};

export type DurationMatchBannerProps = {
  originalDuration: number;
  translatedDuration: number;
  tolerance?: number; // in seconds
};

export type AIInsight = {
  id: string;
  timestamp: string;
  stage: string;
  type: "info" | "warning" | "decision" | "error";
  message: string;
  data?: unknown;
};

export type AIInsightsTabProps = {
  insights: AIInsight[];
  isLoading?: boolean;
  className?: string;
};

