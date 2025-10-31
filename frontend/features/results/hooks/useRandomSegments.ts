"use client";

import { useState, useCallback, useEffect } from "react";

interface RandomSegment {
  id: number;
  timestamp: number;
  duration: number;
  title: string;
  originalText: string;
  translatedText: string;
}

interface UseRandomSegmentsReturn {
  segments: RandomSegment[];
  isLoading: boolean;
  error: string | null;
  fetchRandomSegments: () => void;
}

export function useRandomSegments(duration: number): UseRandomSegmentsReturn {
  const [segments, setSegments] = useState<RandomSegment[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const generateRandomSegments = useCallback(() => {
    if (!duration || duration <= 0) {
      setSegments([]);
      setError("Invalid video duration");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Adaptive parameters based on video duration
      const minGap = duration < 120 ? 3 : 10; // Reduced from 5 to 3 for short videos
      const minDuration = duration < 120 ? 6 : 10; // Reduced from 8 to 6 for short videos
      const maxDuration = duration < 120 ? 15 : 25;

      // Generate fewer spots for shorter videos
      let numSegments;
      if (duration < 60) {
        numSegments = 2;
      } else if (duration < 120) {
        numSegments = 3;
      } else {
        numSegments = 5;
      }

      // Early return if video is too short
      // Each spot needs at least minDuration, plus gaps between spots
      const minRequiredDuration =
        numSegments * minDuration + (numSegments - 1) * minGap;

      if (duration < minRequiredDuration) {
        setSegments([]);
        setError(
          `Video too short for random spots (needs ${minRequiredDuration}s minimum for ${numSegments} spots)`
        );
        setIsLoading(false);
        setHasRun(true);
        return;
      }

      // Generate consecutive segments that cover the entire video
      const generatedSegments: RandomSegment[] = [];

      let currentTime = 0;
      const remainingDuration = duration;
      const avgDurationPerSegment = remainingDuration / numSegments;

      for (let i = 0; i < numSegments; i++) {
        // Last segment should take remaining time
        const isLastSegment = i === numSegments - 1;

        let segmentDuration;
        if (isLastSegment) {
          // Last segment takes whatever time is left
          segmentDuration = duration - currentTime;
        } else {
          // Vary duration around average, ensuring enough space for remaining segments
          const maxAllowed =
            duration - currentTime - (numSegments - i - 1) * minDuration;
          const minForThis = Math.min(minDuration, maxAllowed);
          const maxForThis = Math.min(maxDuration, maxAllowed);

          segmentDuration =
            minForThis + Math.random() * (maxForThis - minForThis);
          segmentDuration = Math.max(minDuration, segmentDuration);
        }

        const endTime = currentTime + segmentDuration;
        const timeInMinutes = Math.floor(currentTime / 60);
        const timeInSeconds = Math.floor(currentTime % 60);
        const timeStr = `${timeInMinutes}:${timeInSeconds
          .toString()
          .padStart(2, "0")}`;
        const endMinutes = Math.floor(endTime / 60);
        const endSeconds = Math.floor(endTime % 60);
        const endStr = `${endMinutes}:${endSeconds
          .toString()
          .padStart(2, "0")}`;

        generatedSegments.push({
          id: i + 1,
          timestamp: currentTime,
          duration: segmentDuration,
          title: `Spot ${i + 1} (${timeStr})`,
          originalText: `Quality check spot from ${timeStr} to ${endStr}. Validating translation accuracy and lip-sync quality at this point in the video.`,
          translatedText: `Контрольная точка с ${timeStr} до ${endStr}. Проверка точности перевода и качества синхронизации губ в этом месте видео.`,
        });

        currentTime = endTime;
      }

      setSegments(generatedSegments);
      if (generatedSegments.length === 0) {
        setError("No random segments could be generated for this video.");
      }
    } catch (err) {
      console.error("Error generating random segments:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to generate random segments"
      );
    } finally {
      setIsLoading(false);
      setHasRun(true);
    }
  }, [duration]);

  // Auto-generate segments when duration is available
  useEffect(() => {
    // Generate if we have a valid duration and haven't run yet
    if (duration > 0 && !hasRun) {
      generateRandomSegments();
    }
    // Also regenerate if duration changes significantly
    else if (duration > 0 && hasRun && segments.length === 0) {
      setHasRun(false);
      generateRandomSegments();
    }
  }, [duration, hasRun, generateRandomSegments, segments.length]);

  return {
    segments,
    isLoading,
    error,
    fetchRandomSegments: generateRandomSegments,
  };
}
