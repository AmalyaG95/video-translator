"use client";

import React from "react";

interface NoActiveSessionProps {
  onNavigateHome: () => void;
}

function NoActiveSession({ onNavigateHome }: NoActiveSessionProps) {
  return (
    <div className="space-y-6">
      <div className="py-12 text-center">
        <h2 className="mb-4 text-2xl font-semibold text-gray-900 dark:text-white">
          No Active Session
        </h2>
        <p className="mb-6 text-gray-600 dark:text-gray-400">
          No video translation session is currently active. Please upload a
          video to start translating.
        </p>
        <button
          onClick={onNavigateHome}
          className="rounded-lg bg-blue-600 px-6 py-3 text-white transition-colors hover:bg-blue-700"
        >
          Upload Video
        </button>
      </div>
    </div>
  );
}

export default NoActiveSession;
