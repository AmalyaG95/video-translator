"use client";

import React from "react";
import { motion } from "framer-motion";
import { FileVideo, Clock, Monitor, HardDrive, Play, X } from "lucide-react";
import Image from "next/image";
import type { VideoMetadataCardProps } from "../../../../types";

function VideoMetadataCard({
  metadata,
  onPreview,
  onUnload,
}: VideoMetadataCardProps) {
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1_024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3_600);
    const minutes = Math.floor((seconds % 3_600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
    }
    return `${minutes}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800"
    >
      {/* Header with title and unload button */}
      <div className="mb-4 flex items-start justify-between">
        <h3 className="truncate text-lg font-semibold text-gray-900 dark:text-white">
          {metadata?.name}
        </h3>
        {onUnload && (
          <button
            onClick={onUnload}
            className="flex-shrink-0 rounded-full p-1 transition-colors hover:bg-gray-100 dark:hover:bg-gray-700"
            title="Unload file"
          >
            <X className="h-4 w-4 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200" />
          </button>
        )}
      </div>

      {/* Thumbnail Block */}
      <div className="relative mb-4 h-48 w-full overflow-hidden rounded-lg bg-gray-100 dark:bg-gray-700">
        {metadata?.thumbnail ? (
          <Image
            src={metadata?.thumbnail}
            alt="Video thumbnail"
            fill
            className="object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <div className="flex flex-col items-center space-y-2">
              <FileVideo className="h-12 w-12 text-gray-400" />
              <div className="text-sm text-gray-500 dark:text-gray-400">
                No preview
              </div>
            </div>
          </div>
        )}
        {onPreview && (
          <button
            onClick={onPreview}
            className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 transition-opacity hover:opacity-100"
          >
            <Play className="h-8 w-8 text-white" />
          </button>
        )}
      </div>

      {/* Metadata Blocks */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="flex items-center space-x-3 rounded-lg bg-gray-50 p-3 dark:bg-gray-700">
          <Clock className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Duration
            </div>
            <div className="text-sm font-medium text-gray-900 dark:text-white">
              {formatDuration(metadata?.duration ?? 0)}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-3 rounded-lg bg-gray-50 p-3 dark:bg-gray-700">
          <Monitor className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Resolution
            </div>
            <div className="text-sm font-medium text-gray-900 dark:text-white">
              {metadata?.resolution}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-3 rounded-lg bg-gray-50 p-3 dark:bg-gray-700">
          <HardDrive className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              File Size
            </div>
            <div className="text-sm font-medium text-gray-900 dark:text-white">
              {formatFileSize(metadata?.size ?? 0)}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-3 rounded-lg bg-gray-50 p-3 dark:bg-gray-700">
          <FileVideo className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          <div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Format
            </div>
            <div className="text-sm font-medium text-gray-900 dark:text-white">
              {metadata?.codec}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default VideoMetadataCard;
