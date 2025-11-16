import type { ProcessingSession } from "@/shared/types/session.types";

export const isSessionComplete = (
  session: ProcessingSession | null
): boolean => {
  return session?.status === "completed" || false;
};

export const hasVideoOutput = (session: ProcessingSession | null): boolean => {
  if (!session) return false;

  const outputPath = session.outputPath || session.result?.outputPath;
  return !!outputPath;
};

export const validateDownload = (
  session: ProcessingSession | null
): {
  valid: boolean;
  error?: string;
} => {
  if (!session) {
    return { valid: false, error: "No session found" };
  }

  if (!isSessionComplete(session)) {
    return { valid: false, error: "Translation not completed yet" };
  }

  if (!hasVideoOutput(session)) {
    return { valid: false, error: "No video available" };
  }

  return { valid: true };
};

export const isQualityMetricAvailable = (
  session: ProcessingSession | null
): boolean => {
  if (!session) return false;
  return !!session.result?.qualityMetrics;
};










