"use client";

import React from "react";
import { motion } from "framer-motion";
import { Clock, Cpu, HardDrive, Zap, Monitor } from "lucide-react";

interface DynamicETAProps {
  etaSeconds?: number;
  hardwareInfo?: {
    cpu: string;
    gpu: string;
    vram_gb: number;
    ram_gb: number;
  };
  processingSpeed?: number;
  currentChunk?: number;
  totalChunks?: number;
  isPaused?: boolean;
  status?: string;
  progress?: number;
}

const DynamicETA = ({
  etaSeconds,
  hardwareInfo,
  processingSpeed,
  currentChunk,
  totalChunks,
  isPaused = false,
  status,
  progress,
}: DynamicETAProps) => {
  const formatTime = (seconds: number) => {
    if (seconds < 60) {
      return `${seconds}s`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      return `${minutes}m`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  };

  const getSpeedIndicator = () => {
    if (!processingSpeed) return null;

    // More realistic thresholds for video translation
    // 10+ chunks/min = Very fast (high-end GPU)
    // 5-10 chunks/min = Good (mid-range GPU or large chunks)
    // < 5 chunks/min = Slow (CPU-only or very large chunks)
    if (processingSpeed >= 10) {
      return { color: "text-green-500", icon: "🚀", text: "Fast" };
    } else if (processingSpeed >= 5) {
      return { color: "text-blue-500", icon: "⚡", text: "Good" };
    } else if (processingSpeed >= 1) {
      return { color: "text-yellow-500", icon: "🐢", text: "Steady" };
    } else {
      return { color: "text-red-500", icon: "🐌", text: "Slow" };
    }
  };

  const speedIndicator = getSpeedIndicator();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-full rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800"
    >
      <div className="mb-3 flex items-center justify-between">
        <h3 className="flex items-center text-lg font-semibold text-gray-900 dark:text-white">
          <Clock className="mr-2 h-5 w-5" />
          Processing Status
        </h3>
        {isPaused && (
          <span className="rounded-full bg-yellow-100 px-2 py-1 text-xs font-medium text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400">
            PAUSED
          </span>
        )}
      </div>

      {/* ETA Display */}
      <div className="mb-4">
        <div className="text-2xl font-bold text-gray-900 dark:text-white">
          {status === "completed" ? (
            <span className="flex items-center text-green-600 dark:text-green-400">
              ✓ Translation Complete
            </span>
          ) : etaSeconds && etaSeconds > 0 ? (
            <span className="flex items-center">
              {progress && progress <= 5 ? (
                <span>
                  Initializing... ~{formatTime(etaSeconds)} estimated
                </span>
              ) : (
                <>
                  ~{formatTime(etaSeconds)} remaining
                  {speedIndicator && (
                    <span className={`ml-2 text-sm ${speedIndicator.color}`}>
                      {speedIndicator.icon} {speedIndicator.text}
                    </span>
                  )}
                </>
              )}
            </span>
          ) : (
            <span className="text-gray-500 dark:text-gray-400">
              {progress && progress <= 5 ? "Initializing pipeline..." : "Calculating..."}
            </span>
          )}
        </div>

        {status === "completed" ? (
          <div className="mt-1 text-sm text-green-600 dark:text-green-400">
            All {totalChunks || currentChunk || 0} chunks processed successfully
          </div>
        ) : (
          !!currentChunk &&
          !!totalChunks && (
            <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Processing chunk {currentChunk + 1} of {totalChunks}
              {processingSpeed && (
                <span className="ml-2">
                  ({processingSpeed.toFixed(1)} chunks/min)
                </span>
              )}
            </div>
          )
        )}
      </div>

      {/* Hardware Info */}
      {hardwareInfo && (
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex items-center space-x-2 text-gray-600 dark:text-gray-400">
            <Cpu className="h-4 w-4" />
            <span className="truncate" title={hardwareInfo.cpu}>
              {hardwareInfo.cpu}
            </span>
          </div>

          <div className="flex items-center space-x-2 text-gray-600 dark:text-gray-400">
            <Monitor className="h-4 w-4" />
            <span className="truncate" title={hardwareInfo.gpu}>
              {hardwareInfo.gpu}
            </span>
          </div>

          <div className="flex items-center space-x-2 text-gray-600 dark:text-gray-400">
            <Zap className="h-4 w-4" />
            <span>{hardwareInfo.vram_gb}GB VRAM</span>
          </div>

          <div className="flex items-center space-x-2 text-gray-600 dark:text-gray-400">
            <HardDrive className="h-4 w-4" />
            <span>{hardwareInfo.ram_gb}GB RAM</span>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default DynamicETA;
