"use client";

import React from "react";
import { formatDuration, formatFileSize } from "../../../utils/formatters";

interface ProcessingStatisticsProps {
  processingTime: number;
  originalSize: number;
  fileSize: number;
  compressionRatio: number;
}

function ProcessingStatistics({
  processingTime,
  originalSize,
  fileSize,
  compressionRatio,
}: ProcessingStatisticsProps) {
  // Only render if we have some real data
  const hasData = processingTime > 0 || originalSize > 0;

  if (!hasData) return null;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
      <h3 className="mb-4 text-lg font-semibold text-gray-900 dark:text-white">
        Processing Statistics
      </h3>

      <div className="space-y-3">
        <div className="flex justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Processing Time
          </span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatDuration(processingTime)}
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Original Size
          </span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatFileSize(originalSize)}
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Final Size
          </span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatFileSize(fileSize)}
          </span>
        </div>

        {fileSize > 0 && originalSize > 0 && (
          <div className="flex justify-between">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Compression
            </span>
            <span
              className={`text-sm font-medium ${
                compressionRatio > 0
                  ? "text-green-600 dark:text-green-400"
                  : "text-red-600 dark:text-red-400"
              }`}
            >
              {compressionRatio > 0 ? "-" : "+"}
              {Math.abs(compressionRatio).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default ProcessingStatistics;




