export const QUALITY_THRESHOLDS = {
  EXCELLENT: 95,
  GOOD: 85,
  ACCEPTABLE: 75,
} as const;

export const METRIC_LABELS = {
  DURATION_MATCH: "Duration Match",
  SYNC_ACCURACY: "Sync Accuracy",
  VOICE_QUALITY: "Voice Quality",
  TRANSLATION_ACCURACY: "Translation Accuracy",
} as const;

export const QUALITY_COLORS = {
  EXCELLENT: "text-green-600 dark:text-green-400",
  GOOD: "text-blue-600 dark:text-blue-400",
  ACCEPTABLE: "text-yellow-600 dark:text-yellow-400",
  POOR: "text-red-600 dark:text-red-400",
} as const;

export const getQualityColor = (value: number): string => {
  if (value >= QUALITY_THRESHOLDS.EXCELLENT) return QUALITY_COLORS.EXCELLENT;
  if (value >= QUALITY_THRESHOLDS.GOOD) return QUALITY_COLORS.GOOD;
  if (value >= QUALITY_THRESHOLDS.ACCEPTABLE) return QUALITY_COLORS.ACCEPTABLE;
  return QUALITY_COLORS.POOR;
};










