"use client";

import React, { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  ChevronDown,
  ChevronUp,
  Filter,
  Info,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
} from "lucide-react";

interface AIReasoningLog {
  id: string;
  timestamp: string;
  stage: string;
  type: "info" | "warning" | "decision" | "error";
  message: string;
  data?: any;
}

interface AIReasoningPanelProps {
  logs: AIReasoningLog[];
  className?: string;
}

type FilterType = "all" | "decisions" | "errors" | "warnings";

export function AIReasoningPanel({
  logs,
  className = "",
}: AIReasoningPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [filter, setFilter] = useState<FilterType>("all");
  const [filteredLogs, setFilteredLogs] = useState<AIReasoningLog[]>([]);

  // Filter logs based on selected filter - use useMemo to prevent infinite loops
  const filteredLogsMemo = useMemo(() => {
    switch (filter) {
      case "decisions":
        return logs.filter(log => log.type === "decision");
      case "errors":
        return logs.filter(log => log.type === "error");
      case "warnings":
        return logs.filter(log => log.type === "warning");
      default:
        return logs;
    }
  }, [logs, filter]);

  // Sync memoized filtered logs to state
  useEffect(() => {
    setFilteredLogs(filteredLogsMemo);
  }, [filteredLogsMemo]);

  const getLogIcon = (type: string) => {
    switch (type) {
      case "decision":
        return <Brain className="h-4 w-4 text-blue-500" />;
      case "error":
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case "warning":
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default:
        return <Info className="h-4 w-4 text-gray-500" />;
    }
  };

  const getLogColor = (type: string) => {
    switch (type) {
      case "decision":
        return "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800";
      case "error":
        return "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800";
      case "warning":
        return "bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800";
      default:
        return "bg-gray-50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700";
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString();
    } catch {
      return timestamp;
    }
  };

  const formatReasoningData = (data: any) => {
    if (!data) return null;

    if (typeof data === "string") {
      try {
        data = JSON.parse(data);
      } catch {
        return data;
      }
    }

    if (typeof data === "object") {
      return Object.entries(data).map(([key, value]) => (
        <div key={key} className="text-xs">
          <span className="font-medium text-gray-600 dark:text-gray-400">
            {key}:
          </span>{" "}
          <span className="text-gray-800 dark:text-gray-200">
            {String(value)}
          </span>
        </div>
      ));
    }

    return String(data);
  };

  // Memoize filterOptions to prevent infinite loops
  const filterOptions: { value: FilterType; label: string; count: number }[] = useMemo(() => [
    { value: "all", label: "All", count: logs.length },
    {
      value: "decisions",
      label: "Decisions",
      count: logs.filter(l => l.type === "decision").length,
    },
    {
      value: "warnings",
      label: "Warnings",
      count: logs.filter(l => l.type === "warning").length,
    },
    {
      value: "errors",
      label: "Errors",
      count: logs.filter(l => l.type === "error").length,
    },
  ], [logs]);

  return (
    <div
      className={`rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800 ${className}`}
    >
      {/* Header */}
      <div
        className="cursor-pointer p-4 transition-colors hover:bg-gray-50 dark:hover:bg-gray-700/50"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Brain className="h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              AI Reasoning
            </h3>
            <span className="rounded-full bg-blue-100 px-2 py-1 text-xs font-medium text-blue-800 dark:bg-blue-900/20 dark:text-blue-400">
              {logs.length} insights
            </span>
          </div>
          {isExpanded ? (
            <ChevronUp className="h-5 w-5 text-gray-500" />
          ) : (
            <ChevronDown className="h-5 w-5 text-gray-500" />
          )}
        </div>
      </div>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            {/* Filters */}
            <div className="border-b border-gray-200 px-4 pb-3 dark:border-gray-700">
              <div className="mb-3 flex items-center space-x-2">
                <Filter className="h-4 w-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Filter:
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {filterOptions.map(option => (
                  <button
                    key={option.value}
                    onClick={() => setFilter(option.value)}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                      filter === option.value
                        ? "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600"
                    }`}
                  >
                    {option.label} ({option.count})
                  </button>
                ))}
              </div>
            </div>

            {/* Logs */}
            <div className="max-h-96 overflow-y-auto">
              {filteredLogs.length === 0 ? (
                <div className="p-8 text-center text-gray-500 dark:text-gray-400">
                  <Brain className="mx-auto mb-3 h-12 w-12 opacity-50" />
                  <p>No {filter === "all" ? "" : filter} insights available</p>
                </div>
              ) : (
                <div className="space-y-3 p-4">
                  {filteredLogs.map((log, index) => (
                    <motion.div
                      key={log.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className={`rounded-lg border p-3 ${getLogColor(log.type)}`}
                    >
                      <div className="flex items-start space-x-3">
                        {getLogIcon(log.type)}
                        <div className="min-w-0 flex-1">
                          <div className="mb-1 flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-900 dark:text-white">
                              {log.stage}
                            </span>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {formatTimestamp(log.timestamp)}
                            </span>
                          </div>

                          <p className="mb-2 text-sm text-gray-700 dark:text-gray-300">
                            {log.message}
                          </p>

                          {log.data && (
                            <div className="mt-2 rounded border bg-white p-2 dark:bg-gray-800">
                              <div className="mb-1 text-xs text-gray-600 dark:text-gray-400">
                                Additional data:
                              </div>
                              {formatReasoningData(log.data)}
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default AIReasoningPanel;
