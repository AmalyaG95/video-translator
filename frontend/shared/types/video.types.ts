
export type VideoFile = {
  file: File;
  name: string;
  size: number;
  type: string;
  preview?: string;
};

export type VideoMetadata = {
  name?: string;
  duration: number;
  width: number;
  height: number;
  fps: number;
  bitrate: number;
  codec: string;
  audioCodec: string;
  audioChannels: number;
  audioSampleRate: number;
  size?: number;
  resolution?: string;
  thumbnail?: string;
};

export type VideoPreviewProps = {
  src: string;
  metadata?: VideoMetadata | null;
  onTimeUpdate?: (time: number) => void;
  onDurationChange?: (duration: number) => void;
  className?: string;
};

export type VideoPreviewModalProps = {
  isOpen: boolean;
  onClose: () => void;
  videoFile?: File | null;
  videoUrl?: string | null;
  metadata?: VideoMetadata | null;
};

export type QualityMetrics = {
  lipSyncAccuracy: number;
  voiceNaturalness: number;
  translationAccuracy: number;
  durationMatch: boolean;
  processingTime: number;
  fileSize: number;
  originalSize: number;
};

export type Segment = {
  id: string;
  start: number;
  end: number;
  duration: number;
  text?: string;
  translatedText?: string;
};

export type Spot = {
  id: number;
  timestamp: number;
  originalText: string;
  translatedText?: string;
};
