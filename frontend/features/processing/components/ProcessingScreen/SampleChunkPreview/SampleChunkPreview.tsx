"use client";

import React from "react";
import { FileVideo } from "lucide-react";
import { VideoPreview } from "@/shared/components/video/VideoPreview";

interface SampleChunkPreviewProps {
  earlyPreviewUrl: string | null;
  earlyPreviewAvailable: boolean;
  previewDuration: number;
  onDurationChange: (duration: number) => void;
}

function SampleChunkPreview({
  earlyPreviewUrl,
  earlyPreviewAvailable,
  previewDuration,
  onDurationChange,
}: SampleChunkPreviewProps) {
  return (
    <div className="card col-span-2 mx-auto flex max-w-4xl flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
          Sample Chunk
        </h3>
      </div>

      <div className="space-y-4">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Preview a 10-30 second sample of your translated video to check
          quality and sync.
        </p>

        {earlyPreviewUrl ? (
          <div className="flex flex-col gap-2">
            <VideoPreview
              src={earlyPreviewUrl}
              className="mx-auto w-full max-w-md rounded-lg shadow-lg"
              onDurationChange={onDurationChange}
            />
            <p className="text-center text-xs text-gray-500 dark:text-gray-400">
              {previewDuration > 0
                ? `This is a ${previewDuration}-second preview of your translation`
                : "Loading preview..."}
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 rounded-lg bg-gray-100 p-8 text-center dark:bg-gray-800">
            <FileVideo className="h-12 w-12 text-gray-400" />
            <p className="text-gray-600 dark:text-gray-400">
              {earlyPreviewAvailable
                ? "Loading preview..."
                : "Preview will be available when processing completes"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default SampleChunkPreview;
