"use client";

import React, { useState } from "react";
import { Download, RotateCcw, ArrowLeft, Loader2 } from "lucide-react";

interface ActionButtonsProps {
  onDownload: () => void | Promise<void>;
  onNewTranslation?: () => void | Promise<void>;
  onBack?: () => void | Promise<void>;
}

function ActionButtons({
  onDownload,
  onNewTranslation,
  onBack,
}: ActionButtonsProps) {
  const [isDownloading, setIsDownloading] = useState(false);
  const [isStartingNew, setIsStartingNew] = useState(false);

  const handleDownload = async () => {
    try {
      setIsDownloading(true);
      await onDownload();
    } finally {
      setIsDownloading(false);
    }
  };

  const handleNewTranslation = async () => {
    if (!onNewTranslation) return;
    try {
      setIsStartingNew(true);
      await onNewTranslation();
    } finally {
      setIsStartingNew(false);
    }
  };

  return (
    <div className="flex items-center space-x-3">
      {onBack && (
        <button
          onClick={onBack}
          className="flex items-center space-x-2 rounded-lg bg-gray-100 px-3 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>Back</span>
        </button>
      )}
      <button
        onClick={handleNewTranslation}
        disabled={isStartingNew}
        className="flex items-center space-x-2 rounded-lg bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
      >
        {isStartingNew ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <RotateCcw className="h-4 w-4" />
        )}
        <span>{isStartingNew ? "Starting..." : "New Translation"}</span>
      </button>
      <button
        onClick={handleDownload}
        disabled={isDownloading}
        className="flex items-center space-x-2 rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isDownloading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Download className="h-4 w-4" />
        )}
        <span>{isDownloading ? "Downloading..." : "Download MP4"}</span>
      </button>
    </div>
  );
}

export default ActionButtons;




