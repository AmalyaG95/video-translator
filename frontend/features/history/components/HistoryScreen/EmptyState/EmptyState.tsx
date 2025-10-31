"use client";

import { FileVideo } from "lucide-react";

interface EmptyStateProps {
  filter: string;
}

export function EmptyState({ filter }: EmptyStateProps) {
  return (
    <div className="card flex flex-col items-center gap-4 py-12 text-center">
      <FileVideo className="h-16 w-16 text-gray-400" />
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
        No sessions found
      </h3>
      <p className="text-gray-600 dark:text-gray-400">
        {filter === "all"
          ? "Start by uploading a video to translate"
          : `No ${filter} sessions found`}
      </p>
    </div>
  );
}

