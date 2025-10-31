"use client";

import {
  useState,
  useEffect,
  useMemo,
  useCallback,
  useDeferredValue,
  startTransition,
} from "react";
import dynamic from "next/dynamic";
import {
  CheckCircle,
  AlertCircle,
  Pause,
  Square,
  Clock,
  FileVideo,
} from "lucide-react";
import { useTranslationStore } from "@/stores/translationStore";
import HistoryHeader from "./HistoryHeader/HistoryHeader";
import { HistoryFilters } from "./HistoryFilters/HistoryFilters";
import { EmptyState } from "./EmptyState/EmptyState";

// Dynamic imports for code splitting
const DynamicSessionsList = dynamic(
  () => import("./SessionsList/SessionsList"),
  {
    loading: () => <SessionsListSkeleton />,
    ssr: false,
  }
);

const SessionsListSkeleton = () => (
  <div className="grid grid-cols-1 gap-6">
    {[1, 2, 3].map(i => (
      <div key={i} className="card animate-pulse">
        <div className="h-24 rounded bg-gray-200 dark:bg-gray-700"></div>
      </div>
    ))}
  </div>
);

// Static content (can be pre-rendered)
const StaticContent = () => (
  <div className="flex items-center justify-between">
    <HistoryHeader />
    <div></div>
  </div>
);

const HistoryScreen = () => {
  const sessions = useTranslationStore(state => state.sessions);
  const [filter, setFilter] = useState<"all" | "completed" | "processing">(
    "all"
  );

  // Deferred values for better PPR
  const deferredFilter = useDeferredValue(filter);
  const deferredSessions = useDeferredValue(sessions);
  const [sessionExists, setSessionExists] = useState<Set<string>>(new Set());
  const [isValidating, setIsValidating] = useState(true);

  // Validate session existence on mount
  useEffect(() => {
    const validateSessions = async () => {
      setIsValidating(true);
      const validIds = new Set<string>();

      for (const session of deferredSessions) {
        // Skip cancelled or failed sessions - they shouldn't be shown or recovered
        if (session.status === "cancelled" || session.status === "failed") {
          continue;
        }

        // Only check non-completed sessions
        if (session.status !== "completed") {
          try {
            const response = await fetch(
              `http://localhost:3001/sessions/${session.sessionId}`
            );
            if (response.ok) {
              validIds.add(session.sessionId);
            }
          } catch {
            // Session doesn't exist on backend
          }
        } else {
          // Completed sessions are assumed to exist
          validIds.add(session.sessionId);
        }
      }

      setSessionExists(validIds);
      setIsValidating(false);
    };

    validateSessions();
  }, [deferredSessions]);

  // Memoized filtered sessions - hide expired sessions that are not completed
  const filteredSessions = useMemo(
    () =>
      deferredSessions.filter((session: any) => {
        // Don't show cancelled or failed sessions
        if (session.status === "cancelled" || session.status === "failed") {
          return false;
        }

        // Only show sessions that exist on backend
        if (!sessionExists.has(session.sessionId)) {
          return false;
        }

        if (deferredFilter === "all") return true;
        return session.status === deferredFilter;
      }),
    [deferredSessions, deferredFilter, sessionExists]
  );

  // Status utilities
  const getStatusIcon = useCallback((status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "processing":
        return <Clock className="h-5 w-5 animate-spin text-blue-500" />;
      case "failed":
      case "error":
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      case "paused":
        return <Pause className="h-5 w-5 text-yellow-500" />;
      case "cancelled":
        return <Square className="h-5 w-5 text-gray-500" />;
      default:
        return <FileVideo className="h-5 w-5 text-gray-500" />;
    }
  }, []);

  const getStatusColor = useCallback((status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "processing":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400";
      case "failed":
      case "error":
        return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      case "paused":
        return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400";
      case "cancelled":
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400";
    }
  }, []);

  const handleFilterChange = useCallback((newFilter: string) => {
    startTransition(() => {
      setFilter(newFilter as any);
    });
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <StaticContent />

      <div className="flex items-center justify-between">
        <div></div>
        <HistoryFilters filter={filter} onFilterChange={handleFilterChange} />
      </div>

      {isValidating ? (
        <SessionsListSkeleton />
      ) : filteredSessions.length === 0 ? (
        <EmptyState filter={deferredFilter} />
      ) : (
        <DynamicSessionsList
          sessions={filteredSessions}
          getStatusIcon={getStatusIcon}
          getStatusColor={getStatusColor}
        />
      )}
    </div>
  );
};

export default HistoryScreen;
