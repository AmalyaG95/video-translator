"use client";

import { History } from "lucide-react";

const HistoryHeader = () => {
  return (
    <div className="flex items-center gap-3">
      <History className="h-6 w-6 text-blue-600 dark:text-blue-400" />
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
        History
      </h1>
    </div>
  );
};

export default HistoryHeader;
