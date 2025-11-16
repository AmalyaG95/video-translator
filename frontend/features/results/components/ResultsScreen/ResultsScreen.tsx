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

  // Set video URL directly for streaming (supports Range requests for large files)
  useEffect(() => {
    if (
      !currentSession?.sessionId ||
      currentSession.status !== "completed" ||
      isLoadingVideo ||
      videoUrl
    )
      return;

    try {
      setIsLoadingVideo(true);

      // Use direct URL streaming instead of blob for large files
      // This allows the browser to use Range requests (206 Partial Content)
      // Essential for 15+ hour videos that can't be loaded entirely into memory
      const directVideoUrl = `http://localhost:3001/download/${currentSession.sessionId}`;
      
      // Verify the video URL is accessible with a HEAD request
      fetch(directVideoUrl, { method: 'HEAD' })
        .then((headResponse) => {
          if (headResponse.status === 404) {
            console.warn("Video file not found (404)");
            setIsLoadingVideo(false);
            return;
          }

          if (!headResponse.ok) {
            console.error(`Video HEAD request failed: ${headResponse.status} ${headResponse.statusText}`);
            setIsLoadingVideo(false);
            return;
          }

          // Get file size from Content-Length header
          const contentLength = headResponse.headers.get('content-length');
          if (contentLength) {
            const size = parseInt(contentLength, 10);
            setVideoFileSize(size);
          }

          // Set the direct URL for streaming
          // The video element will handle Range requests automatically
          setVideoUrl(directVideoUrl);
          setIsLoadingVideo(false);

          // Optionally load metadata asynchronously without blocking
          const video = document.createElement("video");
          video.preload = "metadata";
          video.src = directVideoUrl;
          video.crossOrigin = "anonymous";

          video.onloadedmetadata = () => {
            const duration = video.duration || 0;
            if (duration > 0) {
              setActualVideoDuration(Math.floor(duration));
            }
          };

          video.onerror = () => {
            console.warn("Failed to load video metadata, but video URL is set for streaming");
          };
        })
        .catch((error) => {
          console.error("Failed to verify video URL:", error);
          setIsLoadingVideo(false);
        });

    } catch (error: any) {
      console.error("Error setting up video streaming:", error);
      setIsLoadingVideo(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSession?.sessionId, currentSession?.status]);

  // Cleanup blob URL on unmount
  // Cleanup: No need to revoke URLs since we're using direct URLs, not blob URLs

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
    lipSyncAccuracy: Math.round(hasQualityMetrics?.syncAccuracy || 0),
    voiceNaturalness: Math.round(hasQualityMetrics?.voiceQuality || 0),
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
      // First, try to use processingTimeSeconds directly from result (most accurate - from Python ML)
      if (currentSession?.result?.processingTimeSeconds !== undefined && currentSession?.result?.processingTimeSeconds !== null) {
        const seconds = typeof currentSession.result.processingTimeSeconds === 'number' 
          ? currentSession.result.processingTimeSeconds 
          : parseFloat(String(currentSession.result.processingTimeSeconds));
        if (!isNaN(seconds) && seconds > 0) {
          return Math.floor(seconds);
        }
      }

      // Second, try elapsed_time from session (real-time processing time from backend)
      if (currentSession?.elapsed_time !== undefined && currentSession?.elapsed_time !== null) {
        const elapsed = typeof currentSession.elapsed_time === 'number'
          ? currentSession.elapsed_time
          : parseFloat(String(currentSession.elapsed_time));
        if (!isNaN(elapsed) && elapsed > 0) {
          return Math.floor(elapsed);
        }
      }

      // Third, if processingTime is a number (milliseconds or seconds), use it directly
      if (currentSession?.result?.processingTime !== undefined && currentSession?.result?.processingTime !== null) {
        const timeValue = typeof currentSession.result.processingTime === 'number'
          ? currentSession.result.processingTime
          : parseFloat(String(currentSession.result.processingTime));
        
        if (!isNaN(timeValue) && timeValue > 0) {
          // If it's a large number (> 1000), assume it's milliseconds, convert to seconds
          if (timeValue > 1000) {
            return Math.floor(timeValue / 1000);
          }
          // Otherwise assume it's already in seconds
          return Math.floor(timeValue);
        }
      }

      // Try to parse from result.processingTime string (handles various formats)
      if (currentSession?.result?.processingTime && typeof currentSession.result.processingTime === 'string') {
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
              const matchValue = match[i + 1];
              const multiplier = mult[i];
              if (matchValue !== undefined && multiplier !== undefined) {
                seconds += parseInt(matchValue, 10) * multiplier;
              }
            }
            if (seconds > 0) {
              return seconds;
            }
          }
        }
      }

      // Calculate from timestamps if available (only if session is completed)
      // Use startedAt to completedAt for actual processing duration
      // Fall back to createdAt if startedAt is not available
      if (currentSession?.status === 'completed') {
        const startTime = currentSession?.startedAt || currentSession?.createdAt;
        if (currentSession?.completedAt && startTime) {
          const completedTime = new Date(currentSession.completedAt).getTime();
          const startTimeMs = new Date(startTime).getTime();
          const duration = completedTime - startTimeMs;
          const seconds = Math.floor(duration / 1000);
          if (seconds > 0) {
            return seconds;
          }
        }
      }

      // Fallback: estimate based on video duration (roughly 2-3x realtime for processing)
      // Only use this if we have no other data
      if (currentSession?.duration || actualVideoDuration) {
        const videoDuration = currentSession?.duration || actualVideoDuration;
        // Processing typically takes 2-3x video duration (transcription + translation + TTS + sync)
        return Math.floor(videoDuration * 2.5);
      }

      return 0;
    })(),
    fileSize:
      videoFileSize ||
      currentSession?.result?.outputSize ||
      (currentSession?.metadata?.size) ||
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
              videoUrl={videoUrl || null}
              originalUrl={
                currentSession?.sessionId
                  ? `http://localhost:3001/original/${currentSession.sessionId}`
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
                  currentSession?.sessionId
                    ? `http://localhost:3001/original/${currentSession.sessionId}`
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
