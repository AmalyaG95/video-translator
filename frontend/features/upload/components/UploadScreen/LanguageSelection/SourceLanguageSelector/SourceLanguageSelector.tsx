"use client";

import React from "react";
import { SUPPORTED_LANGUAGES } from "@/constants";

interface SourceLanguageSelectorProps {
  sourceLanguage: string;
  isDetecting: boolean;
  detectedLanguage: string | null;
}

function SourceLanguageSelector({
  sourceLanguage,
  isDetecting,
  detectedLanguage,
}: SourceLanguageSelectorProps) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-2">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Source Language
        </label>
        <div className="flex w-full items-center justify-between rounded-lg border border-gray-300 bg-gray-50 px-4 py-3 dark:border-gray-600 dark:bg-gray-800">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">
              {SUPPORTED_LANGUAGES.find(lang => lang.code === sourceLanguage)
                ?.flag || "🌐"}
            </span>
            <span className="font-medium text-gray-900 dark:text-white">
              {SUPPORTED_LANGUAGES.find(lang => lang.code === sourceLanguage)
                ?.name || "Detecting..."}
            </span>
          </div>
          {isDetecting ? (
            <div className="flex items-center space-x-2 text-blue-600 dark:text-blue-400">
              <div className="h-4 w-4 animate-spin rounded-full border-b-2 border-blue-600"></div>
              <span className="text-sm">AI detecting...</span>
            </div>
          ) : detectedLanguage ? (
            <div className="flex items-center space-x-2 text-green-600 dark:text-green-400">
              <div className="h-2 w-2 rounded-full bg-green-500"></div>
              <span className="text-sm">Auto-detected</span>
            </div>
          ) : (
            <span className="text-sm text-gray-500 dark:text-gray-400">
              Upload video to detect
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default SourceLanguageSelector;
