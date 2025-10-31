"use client";

import { Eye, Download, Trash2, Loader2 } from "lucide-react";

interface SessionActionsProps {
  session: any;
  onView: () => void;
  onDownload: () => void;
  onDelete: () => void;
  isViewing: boolean;
}

 const SessionActions = ({
  session,
  onView,
  onDownload,
  onDelete,
  isViewing,
}: SessionActionsProps) => {
  return (
    <div className="flex items-center space-x-1">
      <button
        onClick={onView}
        disabled={isViewing}
        className="p-2 text-gray-600 transition-colors hover:text-blue-600 disabled:cursor-not-allowed disabled:opacity-50 dark:text-gray-400 dark:hover:text-blue-400"
      >
        {isViewing ? (
          <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
        ) : (
          <Eye className="h-4 w-4" />
        )}
      </button>

      {session.status === "completed" && (
        <button
          onClick={onDownload}
          className="p-2 text-gray-600 transition-colors hover:text-green-600 dark:text-gray-400 dark:hover:text-green-400"
        >
          <Download className="h-4 w-4" />
        </button>
      )}

      <button
        onClick={onDelete}
        className="p-2 text-gray-600 transition-colors hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400"
      >
        <Trash2 className="h-4 w-4" />
      </button>
    </div>
  );
}

export default SessionActions;




