"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Clock, FileVideo, Loader2 } from "lucide-react";
import { VideoPreviewModal } from "@/shared/components/video/VideoPreviewModal";

// Use nullish coalescing for default API URL (ES2020)
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

interface Segment {
  id: string;
  start: number;
  end: number;
  duration: number;
}

interface SegmentPreviewSelectorProps {
  sessionId: string;
  availableSegments: string[];
  className?: string;
}

export function SegmentPreviewSelector({
  sessionId,
  availableSegments,
  className = "",
}: SegmentPreviewSelectorProps) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedSegment, setSelectedSegment] = useState<Segment | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Fetch segments when availableSegments changes
  useEffect(() => {
    if (availableSegments.length > 0) {
      fetchSegments();
    } else {
      // No segments available yet
      setSegments([]);
    }
  }, [availableSegments, sessionId]);

  const fetchSegments = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `${API_URL}/translate/${sessionId}/segments`
      );
      if (response.ok) {
        const data = await response.json();
        setSegments(data.segments || []);
      }
    } catch (error) {
      console.error("Failed to fetch segments:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSegmentClick = async (segment: Segment) => {
    try {
      // Create preview URL for the segment
      const previewUrl = `${API_URL}/translate/${sessionId}/segments?segmentId=${segment.id}&download=true`;
      setPreviewUrl(previewUrl);
      setSelectedSegment(segment);
      setIsPreviewOpen(true);
    } catch (error) {
      console.error("Failed to load segment preview:", error);
    }
  };

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, "0")}`;
  };

  const getSegmentStatus = (segment: Segment) => {
    if (availableSegments.includes(segment.id)) {
      return "ready";
    } else {
      return "processing";
    }
  };

  return (
    <div
      className={`rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800 ${className}`}
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="flex items-center text-lg font-semibold text-gray-900 dark:text-white">
          <FileVideo className="mr-2 h-5 w-5" />
          Segment Previews
        </h3>
        {loading && <Loader2 className="h-4 w-4 animate-spin text-gray-500" />}
      </div>

      {segments.length === 0 ? (
        <div className="py-8 text-center text-gray-500 dark:text-gray-400">
          <FileVideo className="mx-auto mb-3 h-12 w-12 opacity-50" />
          <p>No segments available yet</p>
          <p className="text-sm">Segments will appear as they are processed</p>
        </div>
      ) : (
        <div className="grid max-h-96 grid-cols-1 gap-3 overflow-y-auto sm:grid-cols-2">
          <AnimatePresence>
            {segments.map((segment, index) => {
              const status = getSegmentStatus(segment);
              const isReady = status === "ready";

              return (
                <motion.div
                  key={segment.id}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  whileHover={{ scale: isReady ? 1.02 : 1 }}
                  whileTap={{ scale: isReady ? 0.98 : 1 }}
                  transition={{ delay: index * 0.1 }}
                  className={`group relative cursor-pointer overflow-hidden rounded-lg border-2 transition-all duration-200 ${
                    isReady
                      ? "border-gray-200 hover:border-blue-300 hover:shadow-lg dark:border-gray-600 dark:hover:border-blue-500"
                      : "border-gray-100 opacity-60 dark:border-gray-700"
                  }`}
                  onClick={() => isReady && handleSegmentClick(segment)}
                >
                  {/* Thumbnail Area */}
                  <div className="relative aspect-video overflow-hidden bg-gradient-to-br from-blue-100 to-blue-200 dark:from-blue-900/30 dark:to-blue-800/30">
                    <div className="absolute inset-0 bg-black/10 dark:bg-black/20" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      {isReady ? (
                        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white/90 transition-transform duration-200 group-hover:scale-110 dark:bg-gray-800/90">
                          <Play className="ml-1 h-6 w-6 text-blue-600 dark:text-blue-400" />
                        </div>
                      ) : (
                        <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
                      )}
                    </div>
                    {/* Duration Badge */}
                    <div className="absolute bottom-2 right-2 rounded bg-black/70 px-2 py-1 text-xs text-white">
                      {segment.duration}s
                    </div>
                  </div>

                  {/* Card Content */}
                  <div className="p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        Segment {segment.id.replace("seg_", "")}
                      </span>
                      {isReady ? (
                        <Play className="h-4 w-4 text-blue-500 group-hover:text-blue-600" />
                      ) : (
                        <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                      )}
                    </div>

                    <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                      <div className="flex items-center">
                        <Clock className="mr-1 h-3 w-3" />
                        {formatTime(segment.start)} - {formatTime(segment.end)}
                      </div>
                      <div className="text-gray-500 dark:text-gray-500">
                        {segment.duration}s duration
                      </div>
                    </div>
                  </div>

                  {/* Hover Overlay */}
                  {isReady && (
                    <div className="absolute inset-0 bg-blue-500/10 opacity-0 transition-opacity duration-200 group-hover:opacity-100 dark:bg-blue-400/10" />
                  )}

                  {/* Status overlay */}
                  {!isReady && (
                    <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-gray-50 dark:bg-gray-700/50">
                      <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                        Processing...
                      </span>
                    </div>
                  )}
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}

      {/* Video Preview Modal */}
      {isPreviewOpen && selectedSegment && previewUrl && (
        <VideoPreviewModal
          isOpen={isPreviewOpen}
          onClose={() => {
            setIsPreviewOpen(false);
            setSelectedSegment(null);
            setPreviewUrl(null);
          }}
          videoUrl={previewUrl}
          metadata={{
            name: `Segment ${selectedSegment.id.replace("seg_", "")}`,
            duration: selectedSegment.duration,
            width: 1920, // Default assumption
            height: 1080, // Default assumption
            fps: 30, // Default assumption
            bitrate: 0, // Unknown
            codec: "MP4",
            audioCodec: "AAC", // Default assumption
            audioChannels: 2, // Default assumption
            audioSampleRate: 44100, // Default assumption
            size: 0, // Unknown for URL
            resolution: "1920x1080", // Default assumption
          }}
        />
      )}
    </div>
  );
}
