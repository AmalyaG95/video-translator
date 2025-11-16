"use client";

import React from "react";
import { AlertCircle } from "lucide-react";

function NoResultsAvailable() {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-gray-200 bg-white p-12 dark:border-gray-700 dark:bg-gray-800">
      <AlertCircle className="h-16 w-16 text-gray-400 dark:text-gray-500" />
      <h3 className="mt-4 text-lg font-semibold text-gray-900 dark:text-white">
        No Results Available
      </h3>
      <p className="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
        The translation is not yet complete or no results are available for this
        session.
      </p>
    </div>
  );
}

export default NoResultsAvailable;










