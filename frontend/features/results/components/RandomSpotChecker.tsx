"use client";

import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { CheckCircle, AlertCircle, Clock, Volume2, Eye } from "lucide-react";
import { VideoPreview } from "@/shared/components/video/VideoPreview";
import { useRandomSegments } from "@/features/results/hooks/useRandomSegments";

// Use nullish coalescing for default API URL (ES2020)
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

interface Spot {
  id: number;
  timestamp: number;
  duration: number;
  title: string;
  originalText?: string;
  translatedText?: string;
}

interface RandomSpotCheckerProps {
  videoUrl: string;
  originalUrl: string;
  duration: number;
  sessionId: string;
  onSpotSelect: (spotId: number | null) => void;
  selectedSpot: number | null;
  availableSegments?: string[];
  session?: any;
}

export function RandomSpotChecker({
  videoUrl,
  originalUrl,
  duration,
  sessionId,
  onSpotSelect,
  selectedSpot,
  availableSegments = [],
  session,
}: RandomSpotCheckerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [showOriginal, setShowOriginal] = useState(false);

  // Use the hook to generate random segments based on duration
  const {
    segments: spots,
    isLoading,
    error,
    fetchRandomSegments,
  } = useRandomSegments(duration);

  const resetSpotData = useCallback(() => {
    setCurrentTime(0);
    setIsPlaying(false);
  }, []);

  const handleDurationChange = useCallback((duration: number) => {
    setVideoDuration(duration);
  }, []);

  const handleSpotClick = useCallback(
    (spot: Spot) => {
      // Reset previous spot data first
      resetSpotData();

      // Set new spot data
      onSpotSelect(spot.id);
      setCurrentTime(spot.timestamp);

      // Use setTimeout to ensure the video component is mounted before playing
      setTimeout(() => {
        setIsPlaying(true);
      }, 100);
    },
    [onSpotSelect, resetSpotData]
  );

  const formatTime = (seconds: number) => {
    // Ensure time is not negative and is a valid number
    const validTime = Math.max(0, isNaN(seconds) ? 0 : seconds);
    const mins = Math.floor(validTime / 60);
    const secs = Math.floor(validTime % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const currentSpot = spots.find(spot => spot.id === selectedSpot);

  // Auto-generate spots when duration is available
  useEffect(() => {
    if (duration > 0 && spots.length === 0) {
      fetchRandomSegments();
    }
  }, [duration, spots.length, fetchRandomSegments]);

  // Reset spot data when selectedSpot changes
  useEffect(() => {
    if (selectedSpot === null) {
      resetSpotData();
    }
  }, [selectedSpot, resetSpotData]);

  if (isLoading) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
        <div className="flex h-32 items-center justify-center">
          <div className="text-center">
            <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-b-2 border-blue-500"></div>
            <p className="text-gray-600 dark:text-gray-400">
              Loading random segments...
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
        <div className="flex h-32 items-center justify-center">
          <div className="text-center">
            <AlertCircle className="mx-auto mb-4 h-8 w-8 text-red-500" />
            <p className="text-red-600 dark:text-red-400">{error}</p>
            <button
              onClick={fetchRandomSegments}
              className="mt-2 rounded-lg bg-blue-50 px-4 py-2 text-sm font-medium text-blue-600 transition-colors hover:bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400 dark:hover:bg-blue-900/40"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Check Random Spots
        </h3>
      </div>

      <p className="mb-4 text-sm text-gray-600 dark:text-gray-400">
        Click on any spot to preview that segment and verify translation
        quality.
      </p>

      {/* Show message if no spots */}
      {spots.length === 0 && (
        <div className="mb-6 rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-800 dark:bg-yellow-900/20">
          <div className="flex items-start space-x-2">
            <AlertCircle className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
            <div>
              <h4 className="text-sm font-medium text-yellow-800 dark:text-yellow-400">
                No Random Spots Available
              </h4>
              <p className="mt-1 text-xs text-yellow-700 dark:text-yellow-500">
                The video is too short to generate quality check spots. Please
                use the main video player above for quality verification.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Spot List */}
      <div className="mb-6 grid grid-cols-1 gap-3 md:grid-cols-2">
        {spots.map(spot => {
          // Find corresponding segment for this spot
          const segmentId = availableSegments.find(seg =>
            seg.includes(`seg_${String(spot.id).padStart(3, "0")}`)
          );

          return (
            <motion.button
              key={spot.id}
              onClick={() => handleSpotClick(spot)}
              className={`rounded-lg border p-3 text-left transition-colors ${
                selectedSpot === spot.id
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                  : "border-gray-200 bg-gray-50 hover:bg-gray-100 dark:border-gray-700 dark:bg-gray-700 dark:hover:bg-gray-600"
              }`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="font-medium text-gray-900 dark:text-white">
                    {spot.title}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {formatTime(spot.timestamp)} -{" "}
                    {formatTime(spot.timestamp + spot.duration)}
                  </div>
                  {/* Segment preview video - disabled as these files don't exist */}
                  {/* {segmentId && (
                    <div className="mt-2">
                      <video
                        className="h-16 w-full rounded border object-cover"
                        src={`${API_URL}/translate/${sessionId}/segments?segmentId=${segmentId}&download=true&t=${Date.now()}`}
                        controls={false}
                        muted
                        onError={e => {
                          // Suppress error to prevent retry loop
                          e.currentTarget.style.display = "none";
                        }}
                        onMouseEnter={e => e.currentTarget.play()}
                        onMouseLeave={e => e.currentTarget.pause()}
                      />
                    </div>
                  )} */}
                </div>
                <div className="ml-2 flex items-center space-x-1">
                  <Clock className="h-4 w-4 text-gray-500" />
                  {selectedSpot === spot.id && (
                    <CheckCircle className="h-4 w-4 text-blue-500" />
                  )}
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* Video Preview for Selected Spot */}
      {currentSpot && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-gray-900 dark:text-white">
              Previewing: {currentSpot.title}
            </h4>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowOriginal(!showOriginal)}
                className={`flex items-center space-x-1 rounded-lg px-3 py-1 text-sm transition-colors ${
                  showOriginal
                    ? "bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300"
                    : "bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400"
                }`}
              >
                <Eye className="h-4 w-4" />
                <span>
                  {showOriginal ? "Show Translated" : "Show Original"}
                </span>
              </button>
            </div>
          </div>

          {/* Video Player */}
          <div className="relative">
            <VideoPreview
              key={`${currentSpot.id}-${showOriginal ? "original" : "translated"}`}
              src={showOriginal ? originalUrl : videoUrl}
              className="h-32 w-full rounded-lg"
              startTime={currentSpot.timestamp}
              duration={currentSpot.duration}
              onTimeUpdate={setCurrentTime}
              onDurationChange={handleDurationChange}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            />
          </div>

          {/* Quality Indicators */}
          <div className="flex items-center justify-center space-x-6 text-sm">
            <div className="flex items-center space-x-1">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span className="text-gray-600 dark:text-gray-400">
                Sync Good
              </span>
            </div>
            <div className="flex items-center space-x-1">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span className="text-gray-600 dark:text-gray-400">
                Voice Natural
              </span>
            </div>
            <div className="flex items-center space-x-1">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span className="text-gray-600 dark:text-gray-400">
                Translation Accurate
              </span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Instructions */}
      <div className="mt-6 rounded-lg border border-blue-200 bg-blue-50 p-4 dark:border-blue-800 dark:bg-blue-900/20">
        <div className="flex items-start space-x-2">
          <AlertCircle className="mt-0.5 h-5 w-5 text-blue-500" />
          <div>
            <h4 className="text-sm font-medium text-blue-700 dark:text-blue-400">
              Quality Check Instructions
            </h4>
            <p className="mt-1 text-xs text-blue-600 dark:text-blue-400">
              Click on each spot to verify: 1) Lip-sync accuracy, 2) Voice
              naturalness, 3) Translation quality. Use the toggle to compare
              with original audio.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
