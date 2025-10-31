"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Play, Pause, Volume2, VolumeX, Maximize2 } from "lucide-react";
import type { VideoMetadata } from "@/shared/types";

interface VideoPreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  videoFile?: File | null;
  videoUrl?: string | null;
  metadata?: VideoMetadata | null;
}

export function VideoPreviewModal({
  isOpen,
  onClose,
  videoFile,
  videoUrl,
  metadata,
}: VideoPreviewModalProps) {
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Create video URL when modal opens
  useEffect(() => {
    if (isOpen) {
      if (videoFile) {
        const url = URL.createObjectURL(videoFile);
        setCurrentVideoUrl(url);
        return () => {
          URL.revokeObjectURL(url);
        };
      } else if (videoUrl) {
        setCurrentVideoUrl(videoUrl);
      }
    } else {
      setCurrentVideoUrl(null);
    }
  }, [isOpen, videoFile, videoUrl]);

  // Handle video events
  const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const video = e.currentTarget;
    setDuration(video.duration);
  };

  const handleTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    setCurrentTime(e.currentTarget.currentTime);
  };

  const handlePlayPause = () => {
    const video = document.getElementById("preview-video") as HTMLVideoElement;
    if (video) {
      if (isPlaying) {
        video.pause();
      } else {
        video.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleMuteToggle = () => {
    const video = document.getElementById("preview-video") as HTMLVideoElement;
    if (video) {
      video.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = document.getElementById("preview-video") as HTMLVideoElement;
    if (video) {
      const time = parseFloat(e.target.value);
      video.currentTime = time;
      setCurrentTime(time);
    }
  };

  const handleFullscreen = () => {
    const video = document.getElementById("preview-video") as HTMLVideoElement;
    if (video) {
      if (!isFullscreen) {
        if (video.requestFullscreen) {
          video.requestFullscreen();
        }
      } else {
        if (document.exitFullscreen) {
          document.exitFullscreen();
        }
      }
      setIsFullscreen(!isFullscreen);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ["Bytes", "KB", "MB", "GB"];
    if (bytes === 0) return "0 Bytes";
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + " " + sizes[i];
  };

  if (!isOpen || !videoFile) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="relative mx-4 w-full max-w-4xl overflow-hidden rounded-lg bg-white shadow-2xl dark:bg-gray-900"
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between border-b border-gray-200 p-4 dark:border-gray-700">
            <h3 className="flex-1 text-center text-lg font-semibold text-gray-900 dark:text-white">
              {metadata?.name || videoFile.name}
            </h3>
            <button
              onClick={onClose}
              className="rounded-full p-2 transition-colors hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              <X className="h-5 w-5 text-gray-500" />
            </button>
          </div>

          {/* Video Player */}
          <div className="relative bg-black">
            <video
              id="preview-video"
              src={currentVideoUrl || undefined}
              className="h-auto max-h-[60vh] w-full"
              onLoadedMetadata={handleLoadedMetadata}
              onTimeUpdate={handleTimeUpdate}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
              controls={false}
            />

            {/* Custom Controls Overlay */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
              {/* Progress Bar */}
              <div className="mb-4">
                <input
                  type="range"
                  min="0"
                  max={duration || 0}
                  value={currentTime}
                  onChange={handleSeek}
                  className="slider h-1 w-full cursor-pointer appearance-none rounded-lg bg-gray-600"
                />
              </div>

              {/* Controls */}
              <div className="flex items-center justify-between text-white">
                <div className="flex items-center space-x-4">
                  <button
                    onClick={handlePlayPause}
                    className="rounded-full p-2 transition-colors hover:bg-white/20"
                  >
                    {isPlaying ? (
                      <Pause className="h-6 w-6" />
                    ) : (
                      <Play className="h-6 w-6" />
                    )}
                  </button>

                  <button
                    onClick={handleMuteToggle}
                    className="rounded-full p-2 transition-colors hover:bg-white/20"
                  >
                    {isMuted ? (
                      <VolumeX className="h-5 w-5" />
                    ) : (
                      <Volume2 className="h-5 w-5" />
                    )}
                  </button>

                  <span className="font-mono text-sm">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </span>
                </div>

                <button
                  onClick={handleFullscreen}
                  className="rounded-full p-2 transition-colors hover:bg-white/20"
                >
                  <Maximize2 className="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Video Info */}
          {metadata && (
            <div className="bg-gray-50 p-4 dark:bg-gray-800">
              <div className="grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
                <div>
                  <span className="text-gray-500 dark:text-gray-400">
                    Duration
                  </span>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {formatTime(metadata.duration)}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Size</span>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {formatFileSize(metadata?.size ?? 0)}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">
                    Resolution
                  </span>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {metadata.resolution}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">
                    Codec
                  </span>
                  <p className="font-medium text-gray-900 dark:text-white">
                    {metadata.codec}
                  </p>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
