"use client";

import { useState } from "react";
import { CheckCircle } from "lucide-react";
import { VideoPreview } from "@/shared/components/video/VideoPreview";

interface VideoPlayerProps {
  videoUrl: string | null;
  originalUrl?: string | null;
  durationMatch: boolean;
  isLoading?: boolean;
  originalSubtitleUrl?: string | null;
  translatedSubtitleUrl?: string | null;
}

export default function VideoPlayer({
  videoUrl,
  originalUrl,
  durationMatch,
  isLoading = false,
  originalSubtitleUrl,
  translatedSubtitleUrl,
}: VideoPlayerProps) {
  const [showOriginal, setShowOriginal] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
        <div className="text-center">
          <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-blue-600"></div>
          <p className="text-gray-600 dark:text-gray-400">
            Loading final video...
          </p>
        </div>
      </div>
    );
  }

  if (!videoUrl) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg bg-gray-100 dark:bg-gray-800">
        <p className="text-gray-600 dark:text-gray-400">No video available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
        <div className="mb-2 flex items-center justify-end">
          {originalUrl && videoUrl && (
            <button
              onClick={() => setShowOriginal(!showOriginal)}
              className="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
            >
              {showOriginal ? "Show Translated" : "Show Original"}
            </button>
          )}
        </div>

        {durationMatch && (
          <div className="mb-3 flex items-center gap-2 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-700 dark:bg-green-900/20 dark:text-green-400">
            <CheckCircle className="h-4 w-4" />
            Duration match verified
          </div>
        )}

        <VideoPreview
          src={showOriginal && originalUrl ? originalUrl : videoUrl}
          title={showOriginal ? "Original Video" : "Final Result"}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          subtitleUrl={
            showOriginal ? originalSubtitleUrl : translatedSubtitleUrl
          }
          subtitleLabel={
            showOriginal ? "Original Subtitles" : "Translated Subtitles"
          }
        />
      </div>
    </div>
  );
}
