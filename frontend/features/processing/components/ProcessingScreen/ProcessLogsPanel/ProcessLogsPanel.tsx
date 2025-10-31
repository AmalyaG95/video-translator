"use client";

import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown,
  ChevronUp,
  Filter,
  Search,
  Clock,
  Cpu,
  Database,
  Zap,
  CheckCircle,
  AlertCircle,
  Info,
  Bug,
} from "lucide-react";

interface LogEntry {
  id: string;
  timestamp: string;
  level: "info" | "warning" | "error" | "debug" | "success";
  stage: string;
  message: string;
  chunkId?: string;
  duration?: number;
  sessionId?: string;
  data?: any;
  details?: string;
}

type ProcessLogsPanelProps = {
  logs: LogEntry[];
  error: string | null;
};

const STAGE_ICONS = {
  initialization: Database,
  video_analysis: Cpu,
  speech_to_text: Zap,
  translation: Database,
  text_to_speech: Zap,
  lip_sync: CheckCircle,
  finalization: CheckCircle,
  system: Cpu,
  unknown: Info,
};

// Removed LEVEL_COLORS and LEVEL_BG_COLORS - keeping colors only for icons

export function ProcessLogsPanel({ logs, error }: ProcessLogsPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [filter, setFilter] = useState<string>("all");
  const [searchTerm, setSearchTerm] = useState("");

  const filteredLogs = useMemo(() => {
    let filtered = logs;

    // Filter by level
    if (filter !== "all") {
      filtered = filtered.filter(log => log.level === filter);
    }

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(
        log =>
          log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
          log.stage.toLowerCase().includes(searchTerm.toLowerCase()) ||
          (log.data &&
            JSON.stringify(log.data)
              .toLowerCase()
              .includes(searchTerm.toLowerCase()))
      );
    }

    return filtered;
  }, [logs, filter, searchTerm]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getStageIcon = (stage: string) => {
    const Icon = STAGE_ICONS[stage as keyof typeof STAGE_ICONS] || Info;
    return <Icon className="h-4 w-4 text-gray-900 dark:text-white" />;
  };

  const getLevelIcon = (level: string) => {
    switch (level) {
      case "success":
        return (
          <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
        );
      case "error":
        return (
          <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
        );
      case "warning":
        return (
          <AlertCircle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
        );
      case "debug":
        return <Bug className="h-4 w-4 text-gray-600 dark:text-gray-400" />;
      default:
        return <Info className="h-4 w-4 text-blue-600 dark:text-blue-400" />;
    }
  };

  const logCounts = useMemo(() => {
    return logs.reduce(
      (acc, log) => {
        acc[log.level] = (acc[log.level] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );
  }, [logs]);

  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
      {/* Header */}
      <div
        className="cursor-pointer p-4 transition-colors hover:bg-gray-50 dark:hover:bg-gray-700/50"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Clock className="h-5 w-5 text-gray-600 dark:text-gray-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Process Logs
            </h3>
            <span className="rounded-full bg-blue-100 px-2 py-1 text-xs font-medium text-blue-700 dark:bg-blue-900 dark:text-blue-300">
              {filteredLogs.length} {filteredLogs.length === 1 ? "log" : "logs"}
            </span>
          </div>
          {isExpanded ? (
            <ChevronUp className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          ) : (
            <ChevronDown className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          )}
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="space-y-4 p-4">
              {/* Filters and Search */}
              <div className="flex flex-col gap-4 sm:flex-row">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 transform text-gray-400 dark:text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search logs..."
                      value={searchTerm}
                      onChange={e => setSearchTerm(e.target.value)}
                      className="input w-full border-red-400 pl-20"
                    />
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Filter className="h-4 w-4 text-gray-400 dark:text-gray-400" />
                  <select
                    value={filter}
                    onChange={e => setFilter(e.target.value)}
                    className="input"
                  >
                    <option value="all">All ({logs.length})</option>
                    <option value="info">Info ({logCounts.info || 0})</option>
                    <option value="success">
                      Success ({logCounts.success || 0})
                    </option>
                    <option value="warning">
                      Warning ({logCounts.warning || 0})
                    </option>
                    <option value="error">
                      Error ({logCounts.error || 0})
                    </option>
                    <option value="debug">
                      Debug ({logCounts.debug || 0})
                    </option>
                  </select>
                </div>
              </div>

              {/* Error Display */}
              {error && (
                <div className="rounded-lg border border-red-200 bg-red-50 p-3 dark:border-red-800 dark:bg-red-900/20">
                  <div className="flex items-center space-x-2 text-red-800 dark:text-red-200">
                    <AlertCircle className="h-4 w-4" />
                    <span className="font-medium">Connection Error:</span>
                    <span>{error}</span>
                  </div>
                </div>
              )}

              {/* Logs List */}
              <div className="max-h-96 space-y-2 overflow-y-auto">
                {filteredLogs.length === 0 ? (
                  <div className="py-8 text-center text-gray-500 dark:text-gray-400">
                    {logs.length === 0
                      ? "No logs available"
                      : "No logs match your filters"}
                  </div>
                ) : (
                  filteredLogs.map((log, index) => (
                    <motion.div
                      key={log.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="rounded-lg border border-gray-200 bg-white p-3 transition-all hover:shadow-md dark:border-gray-700 dark:bg-gray-800"
                    >
                      <div className="flex items-start space-x-3">
                        <div className="mt-0.5 flex items-center space-x-2">
                          {getStageIcon(log.stage)}
                          {getLevelIcon(log.level)}
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-900 dark:text-white">
                              {log.stage.replace(/_/g, " ").toUpperCase()}
                            </span>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {formatTimestamp(log.timestamp)}
                            </span>
                          </div>
                          <p className="mt-1 text-sm text-gray-700 dark:text-gray-300">
                            {log.message}
                          </p>
                          {log.chunkId && (
                            <div className="mt-1">
                              <span className="inline-block rounded bg-gray-100 px-2 py-1 text-xs dark:bg-gray-800">
                                {log.chunkId}
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default ProcessLogsPanel;
