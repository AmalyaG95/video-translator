"use client";

import React from "react";
import { Target, Volume2, Globe } from "lucide-react";

interface QualityMetricsProps {
  lipSyncAccuracy: number;
  voiceNaturalness: number;
  translationAccuracy: number;
}

function QualityMetrics({
  lipSyncAccuracy,
  voiceNaturalness,
  translationAccuracy,
}: QualityMetricsProps) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
      <h3 className="mb-4 text-lg font-semibold text-gray-900 dark:text-white">
        Quality Metrics
      </h3>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Target className="h-5 w-5 text-blue-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Lip-Sync Accuracy
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="h-2 w-20 rounded-full bg-gray-200 dark:bg-gray-700">
              <div
                className="h-2 rounded-full bg-green-500"
                style={{ width: `${lipSyncAccuracy}%` }}
              />
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {lipSyncAccuracy}%
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Volume2 className="h-5 w-5 text-green-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Voice Naturalness
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="h-2 w-20 rounded-full bg-gray-200 dark:bg-gray-700">
              <div
                className="h-2 rounded-full bg-green-500"
                style={{ width: `${voiceNaturalness}%` }}
              />
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {voiceNaturalness}%
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Globe className="h-5 w-5 text-purple-500" />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Translation Accuracy
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="h-2 w-20 rounded-full bg-gray-200 dark:bg-gray-700">
              <div
                className="h-2 rounded-full bg-green-500"
                style={{ width: `${translationAccuracy}%` }}
              />
            </div>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {translationAccuracy}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default QualityMetrics;










