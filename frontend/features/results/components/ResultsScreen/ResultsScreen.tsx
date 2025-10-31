"use client";

import React, { useState, useEffect, useCallback, Suspense } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { useTranslationStore } from "@/stores/translationStore";
import { useDownload } from "../../hooks/useDownload";
import { useVoiceRefinement } from "../../hooks/useVoiceRefinement";
import { useSessionValidation as useResultsSessionValidation } from "../../hooks/useSessionValidation";
import { useAIInsights } from "@/features/results/hooks/useAIInsights";

// Dynamic imports for code splitting and lazy loading
const RandomSpotChecker = dynamic(
  () =>
    import("../RandomSpotChecker").then(mod => ({
      default: mod.RandomSpotChecker,
    })),
  {
    loading: () => (
      <div className="h-64 animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
    ),
    ssr: false,
  }
);

const ActionButtons = dynamic(() => import("./ActionButtons"), {
  loading: () => (
    <div className="h-12 w-full animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
  ),
});

const VideoPlayer = dynamic(() => import("./VideoPlayer"), {
  loading: () => (
    <div className="h-64 w-full animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
  ),
});

const QualityMetrics = dynamic(() => import("./QualityMetrics"), {
  loading: () => (
    <div className="h-48 w-full animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
  ),
});

const ProcessingStatistics = dynamic(() => import("./ProcessingStatistics"), {
  loading: () => (
    <div className="h-48 w-full animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
  ),
});

const NoResultsAvailable = dynamic(() => import("./NoResultsAvailable"), {
  ssr: false,
});

const ResultsScreen = () => {
  const router = useRouter();
  const { currentSession } = useResultsSessionValidation();
  const { handleDownload } = useDownload();
  const { handleRefineVoice } = useVoiceRefinement();
  const { insights: aiInsights } = useAIInsights(
    currentSession?.sessionId || null
  );

  const [selectedSpot, setSelectedSpot] = useState<number | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isLoadingVideo, setIsLoadingVideo] = useState(false);
  const [actualVideoDuration, setActualVideoDuration] = useState<number>(0);
  const [videoFileSize, setVideoFileSize] = useState<number>(0);
  const [fullSessionData, setFullSessionData] = useState<any>(null);

  // Fetch full session data to get result field
  useEffect(() => {
    const sessionId = currentSession?.sessionId;
    const isCompleted = currentSession?.status === "completed";

    if (!sessionId || !isCompleted) return;

    const fetchFullSession = async () => {
      try {
        const response = await fetch(
          `http://localhost:3001/sessions/${sessionId}`
        );
        if (response.ok) {
          const data = await response.json();
          setFullSessionData(data);
        } else if (response.status === 404) {
          // Session not found - likely backend was restarted
          // This is expected, just use currentSession data
          return;
        }
      } catch (error) {
        // Silently handle network errors
      }
    };

    fetchFullSession();
  }, [currentSession?.sessionId, currentSession?.status]);

  // Fetch final video when session is completed
  useEffect(() => {
    const fetchFinalVideo = async () => {
      if (
        !currentSession?.sessionId ||
        currentSession.status !== "completed" ||
        isLoadingVideo ||
        videoUrl
      )
        return;

      try {
        setIsLoadingVideo(true);

        // Use AbortController for timeout handling
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

        let response: Response;
        try {
          response = await fetch(
            `http://localhost:3001/download/${currentSession.sessionId}`,
            {
              signal: controller.signal,
            }
          );
        } catch (fetchError: any) {
          clearTimeout(timeoutId);
          if (fetchError.name === "AbortError") {
            console.warn("Video download timeout");
            return;
          }
          throw fetchError;
        }

        clearTimeout(timeoutId);

        if (response.status === 404) {
          // Video not found - likely backend was restarted or video was deleted
          // This is expected, silently handle it
          return;
        }

        if (!response.ok) {
          throw new Error(
            `Video not available: ${response.status} ${response.statusText}`
          );
        }

        // Handle large blob responses more carefully
        let blob: Blob;
        try {
          blob = await response.blob();
        } catch (blobError: any) {
          // If blob conversion fails, it might be a network issue
          console.warn("Failed to read video blob:", blobError);
          return;
        }

        // Create video element to get actual duration and size
        const tempUrl = URL.createObjectURL(blob);
        setVideoFileSize(blob.size); // Set file size from blob

        const video = document.createElement("video");
        video.src = tempUrl;

        await new Promise<void>(resolve => {
          video.onloadedmetadata = () => {
            const duration = video.duration || 0;
            setActualVideoDuration(Math.floor(duration));
            URL.revokeObjectURL(tempUrl);
            resolve();
          };
          video.onerror = () => {
            URL.revokeObjectURL(tempUrl);
            resolve();
          };
        });

        // Clean up old URL if exists
        if (videoUrl) {
          URL.revokeObjectURL(videoUrl);
        }

        const url = URL.createObjectURL(blob);
        setVideoUrl(url);
      } catch (error: any) {
        // Handle network errors gracefully
        if (error.name === "AbortError") {
          console.warn("Video download was aborted");
        } else if (
          error.message?.includes("Failed to fetch") ||
          error.name === "TypeError"
        ) {
          console.warn("Network error fetching video:", error);
        } else {
          console.warn("Error fetching video:", error);
        }
        // Silently handle errors - video may not be available
      } finally {
        setIsLoadingVideo(false);
      }
    };

    fetchFinalVideo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSession?.sessionId, currentSession?.status]);

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  const handleDownloadClick = useCallback(() => {
    if (!currentSession?.sessionId) return;
    handleDownload(currentSession.sessionId);
  }, [currentSession?.sessionId, handleDownload]);

  const handleRefineVoiceClick = useCallback(() => {
    if (!currentSession?.sessionId) return;
    handleRefineVoice(currentSession.sessionId);
  }, [currentSession?.sessionId, handleRefineVoice]);

  const handleNewTranslationClick = useCallback(() => {
    useTranslationStore.getState().setCurrentSession(null);
    router.push("/");
  }, [router]);

  // Quality metrics from session data - only show if real data exists
  // Use fullSessionData if available, otherwise fallback to currentSession
  const sessionWithResult = fullSessionData || currentSession;
  const hasQualityMetrics = sessionWithResult?.result?.qualityMetrics;

  // Debug: log session data to see what we're getting
  console.log("[ResultsScreen] Current session:", currentSession);
  console.log("[ResultsScreen] Full session data from API:", fullSessionData);
  console.log("[ResultsScreen] Session with result:", sessionWithResult);
  console.log("[ResultsScreen] Has quality metrics:", hasQualityMetrics);
  console.log(
    "[ResultsScreen] Quality metrics value:",
    sessionWithResult?.result?.qualityMetrics
  );
  const qualityMetrics = {
    lipSyncAccuracy: hasQualityMetrics?.syncAccuracy || 0,
    voiceNaturalness: hasQualityMetrics?.voiceQuality || 0,
    translationAccuracy:
      hasQualityMetrics?.translationQuality === "high"
        ? 90
        : hasQualityMetrics?.translationQuality === "medium"
          ? 75
          : hasQualityMetrics?.translationQuality === "low"
            ? 60
            : 0,
    durationMatch: hasQualityMetrics?.durationMatch || false,
    processingTime: (() => {
      // First try to parse from result.processingTime string (handles various formats)
      if (currentSession?.result?.processingTime) {
        const timeStr = currentSession.result.processingTime;
        // Try different patterns
        const patterns = [
          { regex: /(\d+)h\s*(\d+)m/, mult: [3600, 60] }, // hours and minutes
          { regex: /(\d+)m\s*(\d+)s/, mult: [60, 1] }, // minutes and seconds
          { regex: /(\d+)h/, mult: [3600] }, // hours only
          { regex: /(\d+)m/, mult: [60] }, // minutes only
          { regex: /(\d+)s/, mult: [1] }, // seconds only
        ];

        for (const { regex, mult } of patterns) {
          const match = timeStr.match(regex);
          if (match) {
            let seconds = 0;
            for (let i = 0; i < mult.length; i++) {
              seconds += parseInt(match[i + 1] || "0") * mult[i];
            }
            return seconds;
          }
        }
      }

      // Calculate from timestamps if available
      if (currentSession?.completedAt && currentSession?.createdAt) {
        const duration =
          new Date(currentSession.completedAt).getTime() -
          new Date(currentSession.createdAt).getTime();
        return Math.floor(duration / 1000); // Convert to seconds
      }

      // Don't calculate from createdAt to now - that would keep growing
      // Just return 0 if we don't have proper data

      // Fallback: estimate based on video duration (roughly 2x realtime)
      if (currentSession?.duration || actualVideoDuration) {
        return Math.floor(
          (currentSession?.duration || actualVideoDuration) / 2
        );
      }

      return 0;
    })(),
    fileSize:
      videoFileSize ||
      currentSession?.result?.outputSize ||
      currentSession?.metadata?.outputSize ||
      0, // in bytes
    originalSize:
      currentSession?.fileSize || currentSession?.metadata?.size || 0, // in bytes
  };

  // Calculate compression ratio, handle division by zero and missing data
  const compressionRatio =
    qualityMetrics.originalSize > 0 && qualityMetrics.fileSize > 0
      ? ((qualityMetrics.originalSize - qualityMetrics.fileSize) /
          qualityMetrics.originalSize) *
        100
      : 0;

  if (!currentSession || currentSession.status !== "completed") {
    return (
      <Suspense fallback={<div className="h-screen w-full" />}>
        <NoResultsAvailable />
      </Suspense>
    );
  }

  // Calculate duration - use actual video duration if available, otherwise from session
  const videoDuration =
    actualVideoDuration ||
    currentSession?.duration ||
    currentSession?.metadata?.duration ||
    0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Translation Complete!
            </h2>
            <p className="mt-1 text-gray-600 dark:text-gray-400">
              Your video has been successfully translated and is ready for
              download.
            </p>
          </div>
          <Suspense
            fallback={
              <div className="h-12 w-full animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
            }
          >
            <ActionButtons
              onDownload={handleDownloadClick}
              onNewTranslation={handleNewTranslationClick}
            />
          </Suspense>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Video Player */}
        <div className="space-y-4">
          <Suspense
            fallback={
              <div className="h-64 w-full animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
            }
          >
            <VideoPlayer
              videoUrl={videoUrl}
              originalUrl={
                currentSession?.filePath
                  ? `http://localhost:3001/uploads/${currentSession.filePath.split("/").pop()}`
                  : null
              }
              durationMatch={qualityMetrics.durationMatch}
              isLoading={isLoadingVideo}
              originalSubtitleUrl={
                currentSession?.sessionId
                  ? `http://localhost:3001/subtitles/${currentSession.sessionId}`
                  : null
              }
              translatedSubtitleUrl={
                currentSession?.sessionId
                  ? `http://localhost:3001/subtitles/${currentSession.sessionId}/translated`
                  : null
              }
            />
          </Suspense>

          {/* Random Spot Checker - Only show when duration is available */}
          {videoDuration > 0 && (
            <Suspense
              fallback={
                <div className="h-64 w-full animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
              }
            >
              <RandomSpotChecker
                videoUrl={videoUrl || ""}
                originalUrl={
                  currentSession?.filePath
                    ? `http://localhost:3001/uploads/${currentSession.filePath.split("/").pop()}`
                    : ""
                }
                duration={videoDuration}
                sessionId={currentSession?.sessionId || ""}
                onSpotSelect={setSelectedSpot}
                selectedSpot={selectedSpot}
                availableSegments={currentSession?.availableSegments || []}
                session={currentSession}
              />
            </Suspense>
          )}
        </div>

        {/* Stats Panel */}
        <div className="space-y-4">
          {/* Quality Metrics - Only show if real data is available */}
          {hasQualityMetrics && (
            <Suspense
              fallback={
                <div className="h-48 w-full animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
              }
            >
              <QualityMetrics
                lipSyncAccuracy={qualityMetrics.lipSyncAccuracy}
                voiceNaturalness={qualityMetrics.voiceNaturalness}
                translationAccuracy={qualityMetrics.translationAccuracy}
              />
            </Suspense>
          )}

          {/* Processing Stats - Show loading if data not ready */}
          <Suspense
            fallback={
              <div className="h-48 w-full animate-pulse rounded-lg bg-gray-200 dark:bg-gray-700" />
            }
          >
            {isLoadingVideo ||
            (qualityMetrics.fileSize === 0 &&
              qualityMetrics.originalSize === 0) ? (
              <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
                <div className="space-y-4">
                  <div className="h-6 w-32 animate-pulse rounded bg-gray-200 dark:bg-gray-700"></div>
                  <div className="space-y-2">
                    <div className="h-4 w-full animate-pulse rounded bg-gray-200 dark:bg-gray-700"></div>
                    <div className="h-4 w-3/4 animate-pulse rounded bg-gray-200 dark:bg-gray-700"></div>
                    <div className="h-4 w-5/6 animate-pulse rounded bg-gray-200 dark:bg-gray-700"></div>
                  </div>
                </div>
              </div>
            ) : (
              <ProcessingStatistics
                processingTime={qualityMetrics.processingTime}
                originalSize={qualityMetrics.originalSize}
                fileSize={qualityMetrics.fileSize}
                compressionRatio={compressionRatio}
              />
            )}
          </Suspense>
        </div>
      </div>
    </div>
  );
};

export default ResultsScreen;
