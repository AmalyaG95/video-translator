"use client";

import { Calendar, Clock } from "lucide-react";
import SessionActions from "../SessionActions/SessionActions";

interface SessionCardProps {
  session: any;
  onView: (session: any) => void;
  onDownload: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
  getStatusIcon: (status: string) => React.ReactNode;
  getStatusColor: (status: string) => string;
  isViewing: boolean;
}

const SessionCard = ({
  session,
  onView,
  onDownload,
  onDelete,
  getStatusIcon,
  getStatusColor,
  isViewing,
}: SessionCardProps) => {
  return (
    <div className="card-hover">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            {getStatusIcon(session.status)}
            <span
              className={`rounded-full px-3 py-1 text-sm font-medium ${getStatusColor(session.status)}`}
            >
              {session.status}
            </span>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white">
              {session.sourceLang?.toUpperCase() || "N/A"} →{" "}
              {session.targetLang?.toUpperCase() || "N/A"}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Session ID: {session.sessionId}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {Math.round(session.progress || 0)}%
          </span>

          <SessionActions
            session={session}
            onView={() => onView(session)}
            onDownload={() => onDownload(session.sessionId)}
            onDelete={() => onDelete(session.sessionId)}
            isViewing={isViewing}
          />
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
            <div className="flex items-center space-x-1">
              <Calendar className="h-4 w-4" />
              <span>
                {session.createdAt
                  ? new Date(session.createdAt).toLocaleDateString()
                  : "N/A"}
              </span>
            </div>

            <div className="flex items-center space-x-1">
              <Clock className="h-4 w-4" />
              <span>
                {session.createdAt
                  ? new Date(session.createdAt).toLocaleTimeString()
                  : "N/A"}
              </span>
            </div>
          </div>

          <div className="text-sm text-gray-600 dark:text-gray-400">
            {session.currentStep || "N/A"}
          </div>
        </div>

        {/* Progress Bar */}
        <div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${session.progress || 0}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionCard;

