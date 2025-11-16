"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useTranslationStore } from "@/stores/translationStore";

// Helper to schedule idle work
const scheduleIdleWork = (callback: () => void) => {
  if (typeof window !== "undefined" && "requestIdleCallback" in window) {
    requestIdleCallback(callback, { timeout: 2000 });
  } else {
    setTimeout(callback, 1);
  }
};

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

interface UseLogsStreamReturn {
  logs: LogEntry[];
  isConnected: boolean;
  error: string | null;
  clearLogs: () => void;
  addLog: (log: LogEntry) => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

export function useLogsStream(sessionId: string | null): UseLogsStreamReturn {
  // Use selector functions to ensure stable references and prevent infinite loops
  const processLogs = useTranslationStore(state => state.processLogs);
  const setProcessLogs = useTranslationStore(state => state.setProcessLogs);
  const addProcessLog = useTranslationStore(state => state.addProcessLog);
  
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const hasCheckedStatusRef = useRef<boolean>(false);
  
  // Use refs to store stable function references to prevent infinite loops
  // Initialize refs once - Zustand selectors should be stable
  const addProcessLogRef = useRef(addProcessLog);
  const setProcessLogsRef = useRef(setProcessLogs);
  
  // Track last log ID to prevent duplicate rapid additions
  const lastLogIdRef = useRef<string | null>(null);
  const lastLogTimeRef = useRef<number>(0);
  
  // Batch logs to prevent rapid store updates
  const logBatchRef = useRef<LogEntry[]>([]);
  const batchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Update refs only if functions actually changed (defensive check)
  useEffect(() => {
    if (addProcessLogRef.current !== addProcessLog) {
      addProcessLogRef.current = addProcessLog;
    }
    if (setProcessLogsRef.current !== setProcessLogs) {
      setProcessLogsRef.current = setProcessLogs;
    }
  }, [addProcessLog, setProcessLogs]);
  
  // Flush batched logs - use ref to avoid dependency issues
  const flushLogBatchRef = useRef(() => {
    if (logBatchRef.current.length === 0) return;
    
    const logsToAdd = [...logBatchRef.current];
    logBatchRef.current = [];
    
    // Add logs one by one to ensure duplicate checking works
    logsToAdd.forEach(log => {
      try {
        addProcessLogRef.current(log);
      } catch (error) {
        console.error("Error adding batched log:", error);
      }
    });
  });
  
  // Update flush function when addProcessLogRef changes
  useEffect(() => {
    flushLogBatchRef.current = () => {
      if (logBatchRef.current.length === 0) return;
      
      const logsToAdd = [...logBatchRef.current];
      logBatchRef.current = [];
      
      logsToAdd.forEach(log => {
        try {
          addProcessLogRef.current(log);
        } catch (error) {
          console.error("Error adding batched log:", error);
        }
      });
    };
  }, []);

  const clearLogs = useCallback(() => {
    // Don't clear all logs - just clear logs for current session
    // Keep historical logs from previous sessions
    if (sessionId) {
      const currentLogs = useTranslationStore.getState().processLogs;
      const filteredLogs = currentLogs.filter(
        log => log.sessionId !== sessionId
      );
      setProcessLogsRef.current(filteredLogs);
    }
  }, [sessionId]);

  const addLog = useCallback(
    (log: LogEntry) => {
      // Prevent rapid duplicate additions that could cause infinite loops
      const now = Date.now();
      const logId = log.id || `${log.timestamp}-${log.message}`;
      
      // Skip if same log was added very recently (< 50ms ago) - increased threshold
      if (
        lastLogIdRef.current === logId &&
        now - lastLogTimeRef.current < 50
      ) {
        return; // Skip duplicate rapid additions
      }
      
      lastLogIdRef.current = logId;
      lastLogTimeRef.current = now;
      
      // Ensure log is tagged with current sessionId
      const logWithSession = {
        ...log,
        sessionId: sessionId || log.sessionId,
      };
      
      // Batch logs to prevent rapid store updates that could cause infinite loops
      logBatchRef.current.push(logWithSession);
      
      // Clear existing timeout
      if (batchTimeoutRef.current) {
        clearTimeout(batchTimeoutRef.current);
      }
      
      // Flush batch after a short delay (debounce)
      batchTimeoutRef.current = setTimeout(() => {
        flushLogBatchRef.current();
        batchTimeoutRef.current = null;
      }, 100); // 100ms debounce
    },
    [sessionId]
  );

  useEffect(() => {
    // Close existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    // Reset status check flag when session changes
    hasCheckedStatusRef.current = false;

    if (!sessionId || sessionId === "null" || sessionId === "undefined") {
      setIsConnected(false);
      setError(null);
      // Don't clear logs when no sessionId - show all historical logs
      return;
    }

    // Check if session is already completed to avoid unnecessary connections
    const checkSessionStatus = async () => {
      if (hasCheckedStatusRef.current) {
        return true; // Already checked, proceed with connection
      }

      try {
        // First check if backend is running
        const healthResponse = await fetch(`${API_URL}/health`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (!healthResponse.ok) {
          console.warn("Backend health check failed, skipping logs stream");
          setIsConnected(false);
          setError("Backend not available");
          return false;
        }

        const response = await fetch(`${API_URL}/sessions/${sessionId}`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (response.ok) {
          const session = await response.json();
          hasCheckedStatusRef.current = true;
          if (session.status === "completed" || session.status === "failed") {
            setIsConnected(false);
            setError(null);
            return false;
          }
        } else if (response.status === 404) {
          // Session doesn't exist - mark as checked and stop retrying
          // Don't log 404s as they're expected for missing/deleted sessions
          hasCheckedStatusRef.current = true;
          setIsConnected(false);
          setError(null);
          return false;
        } else {
          // Other errors - mark as checked to avoid infinite retries
          // Only log non-404 errors
          console.warn(`Session check failed: ${response.status}`);
          hasCheckedStatusRef.current = true;
          setIsConnected(false);
          setError("Session check failed");
          return false;
        }
      } catch (err) {
        // Silently handle connection errors - don't spam console
        // These are common when backend isn't running
        setIsConnected(false);
        setError("Backend not available");
        return false;
      }
      return true;
    };

    // Only create EventSource if session is not completed
    checkSessionStatus()
      .then(shouldConnect => {
        if (!shouldConnect) {
          // Session is completed, don't create EventSource
          return;
        }

        // Double-check sessionId is valid before creating EventSource
        if (!sessionId || sessionId === "null" || sessionId === "undefined") {
          return;
        }

        // Add a small delay and re-check session status to avoid race conditions
        setTimeout(() => {
          // Re-check session status before creating EventSource
          fetch(`${API_URL}/sessions/${sessionId}`)
            .then(response => {
              if (!response.ok) {
                return;
              }
              return response.json();
            })
            .then(session => {
              if (!session) return;

              if (
                session.status === "completed" ||
                session.status === "failed"
              ) {
                return;
              }

              // Create EventSource for Server-Sent Events
              const eventSource = new EventSource(
                `${API_URL}/logs/stream/${sessionId}`
              );
              eventSourceRef.current = eventSource;

              // Set a timeout to close connection if it doesn't open within 5 seconds
              const connectionTimeout = setTimeout(() => {
                if (eventSource.readyState === EventSource.CONNECTING) {
                  console.warn("Logs stream connection timeout, closing...");
                  eventSource.close();
                  eventSourceRef.current = null;
                  setIsConnected(false);
                  setError(null); // Don't show error for timeout
                }
              }, 5000);

              eventSource.onopen = () => {
                clearTimeout(connectionTimeout);
                setIsConnected(true);
                setError(null);
              };

              eventSource.onmessage = event => {
                try {
                  const logData = JSON.parse(event.data);
                  
                  // Skip heartbeat messages - they're just to keep connection alive
                  if (logData.type === 'heartbeat') {
                    return;
                  }
                  
                  // Skip if message is empty (heartbeat)
                  if (!logData.message || logData.message.trim() === '') {
                    return;
                  }
                  
                  const logEntry: LogEntry = {
                    id:
                      logData.id ||
                      `${logData.timestamp || new Date().toISOString()}-${logData.message}`,
                    timestamp: logData.timestamp || new Date().toISOString(),
                    level: logData.level || "info",
                    stage: logData.stage || "unknown",
                    message: logData.message || "",
                    chunkId: logData.chunkId,
                    duration: logData.duration,
                    sessionId: logData.sessionId,
                    data: logData.data,
                    details: logData.details,
                  };

                  // Use idle callback to prevent blocking UI
                  scheduleIdleWork(() => {
                    addLog(logEntry);
                  });
                } catch (err) {
                  console.error("Failed to parse log data:", err);
                }
              };

              eventSource.onerror = err => {
                clearTimeout(connectionTimeout);
                setIsConnected(false);

                // Close the connection on error to prevent reconnection attempts
                eventSource.close();
                eventSourceRef.current = null;

                // Check if this is a completed session error (expected behavior)
                if (eventSource.readyState === EventSource.CLOSED) {
                  // Check session status to determine if this is expected
                  fetch(`${API_URL}/sessions/${sessionId}`)
                    .then(response => {
                      if (response.ok) {
                        return response.json();
                      }
                      throw new Error("Session not found");
                    })
                    .then(session => {
                      if (
                        session.status === "completed" ||
                        session.status === "failed"
                      ) {
                        setError(null); // Clear error for completed sessions
                        setIsConnected(false); // Ensure connection state is updated
                      } else {
                        setError("Logs stream connection lost");
                      }
                    })
                    .catch(() => {
                      setError(null);
                      setIsConnected(false);
                    });
                } else {
                  setError("Failed to connect to logs stream");
                }
              };
            })
            .catch(error => {
              console.error("Re-check failed:", error);
            });
        }, 100); // Small delay to avoid race conditions
      })
      .catch(error => {
        console.error("Session check failed:", error);
      });

    // Cleanup on unmount or session change
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      // Flush any pending logs before cleanup
      if (batchTimeoutRef.current) {
        clearTimeout(batchTimeoutRef.current);
        batchTimeoutRef.current = null;
      }
      flushLogBatchRef.current();
      setIsConnected(false);
      // Reset status check flag on cleanup
      hasCheckedStatusRef.current = false;
    };
  }, [sessionId]); // Removed addLog from dependencies to prevent unnecessary re-runs

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  // Get current session to extract logs
  const currentSession = useTranslationStore(state => state.currentSession);
  
  // Combine logs from session and processLogs store
  const sessionLogs = currentSession?.logs || [];
  const storeLogs = sessionId
    ? processLogs.filter(log => log.sessionId === sessionId)
    : processLogs;
  
  // Merge logs, avoiding duplicates by id
  const logMap = new Map<string, LogEntry>();
  
  // Add store logs first
  storeLogs.forEach(log => {
    logMap.set(log.id || `${log.timestamp}-${log.message}`, log);
  });
  
  // Add session logs (they override store logs if same id)
  sessionLogs.forEach(log => {
    logMap.set(log.id || `${log.timestamp}-${log.message}`, log);
  });
  
  // Convert to array and sort by timestamp
  const filteredLogs = Array.from(logMap.values())
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));

  return {
    logs: filteredLogs,
    isConnected,
    error,
    clearLogs,
    addLog,
  };
}
