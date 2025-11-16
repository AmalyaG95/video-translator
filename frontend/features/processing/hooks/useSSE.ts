"use client";

import {
  useCallback,
  useEffect,
  useState,
  useRef,
  startTransition,
} from "react";
import { useTranslationStore } from "@/stores/translationStore";
import toast from "react-hot-toast";
import throttle from "lodash.throttle";

// Use nullish coalescing for default API URL (ES2020)
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

export function useSSE() {
  const [isConnected, setIsConnected] = useState(false);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const { updateSessionProgress, setCurrentSession } = useTranslationStore();

  // Use refs to avoid dependency issues
  const eventSourceRef = useRef<EventSource | null>(null);
  const updateSessionProgressRef = useRef(updateSessionProgress);
  const setCurrentSessionRef = useRef(setCurrentSession);
  const lastSessionIdRef = useRef<string | null>(null);
  const connectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Reduced throttling for real-time updates (100ms instead of 500ms)
  const throttledUpdateProgress = useRef(
    throttle(
      (sessionId: string, data: any) => {
        updateSessionProgressRef.current(sessionId, data);
      },
      100, // Reduced from 500ms to 100ms for real-time feel
      { leading: true, trailing: true }
    )
  );

  const throttledUpdateSession = useRef(
    throttle(
      (session: any) => {
        setCurrentSessionRef.current(session);
      },
      100, // Reduced from 500ms to 100ms for real-time feel
      { leading: true, trailing: true }
    )
  );

  // Update refs when values change
  useEffect(() => {
    updateSessionProgressRef.current = updateSessionProgress;
    setCurrentSessionRef.current = setCurrentSession;
  }, [updateSessionProgress, setCurrentSession]);

  const connectToSession = useCallback((sessionId: string) => {
    // Don't connect if sessionId is empty or invalid
    if (!sessionId || sessionId.trim() === "") {
      return;
    }

    // Don't connect if already connected to the same session
    if (
      lastSessionIdRef.current === sessionId &&
      eventSourceRef.current?.readyState === EventSource.OPEN
    ) {
      return;
    }

    // Clear any pending connection timeout
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
    }

    // Close existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = new EventSource(`${API_URL}/progress/${sessionId}`);
    eventSourceRef.current = es;
    setEventSource(es);
    lastSessionIdRef.current = sessionId;

    es.onopen = () => {
      setIsConnected(true);
    };

    es.onmessage = event => {
      try {
        const data = JSON.parse(event.data);

        // Use optional chaining and nullish coalescing (ES2020)
        const sessionId = data.sessionId ?? data.session_id;

        // Prepare progress data
        const progressData = {
          progress: data.progress ?? 0,
          currentStep: data.currentStep ?? data.current_step ?? "",
          status: data.status,
          early_preview_available:
            data.earlyPreviewAvailable ?? data.early_preview_available ?? false,
          early_preview_path:
            data.earlyPreviewPath ?? data.early_preview_path ?? "",
          // Enhanced fields
          etaSeconds: data.etaSeconds,
          hardwareInfo: data.hardwareInfo,
          // Detailed progress fields (from backend kwargs)
          segments_processed: data.segments_processed,
          current_time: data.current_time,
          current_time_formatted: data.current_time_formatted,
          total_duration: data.total_duration,
          total_duration_formatted: data.total_duration_formatted,
          progress_percent: data.progress_percent,
          elapsed_time: data.elapsed_time,
          stage: data.stage,
          stage_number: data.stage_number,
          total_stages: data.total_stages,
          stage_progress_percent: data.stage_progress_percent,
          availableSegments: data.availableSegments,
          processingSpeed: data.processingSpeed,
          currentChunk: data.currentChunk,
          totalChunks: data.totalChunks,
          logs: data.logs || [], // Include logs from SSE
        };

        // Check if this is a critical status change
        const isCritical =
          data.status === "completed" || data.status === "failed";

        if (isCritical) {
          // For completed sessions, fetch full session data from API to get result object
          if (data.status === "completed" && sessionId) {
            fetch(`${API_URL}/sessions/${sessionId}`)
              .then(response => {
                if (response.ok) {
                  return response.json();
                }
                throw new Error("Failed to fetch session");
              })
              .then(fullSessionData => {
                // Include result object in updates
                const updatesWithResult = {
                  ...progressData,
                  result: fullSessionData.result,
                  completedAt: fullSessionData.completedAt,
                  startedAt: fullSessionData.startedAt,
                };
                
                updateSessionProgressRef.current(sessionId, updatesWithResult);

                // Update current session with full data including result
                const currentSession = useTranslationStore.getState().currentSession;
                if (currentSession && sessionId) {
                  const updatedSession: typeof currentSession = {
                    ...currentSession,
                    ...fullSessionData,
                    sessionId,
                  };

                  setCurrentSessionRef.current(updatedSession);
                }
              })
              .catch(error => {
                console.error("Failed to fetch full session data:", error);
                // Fallback to basic update if fetch fails
                updateSessionProgressRef.current(sessionId, progressData);
              });
          } else {
            // For failed or other critical statuses, use basic update
            updateSessionProgressRef.current(sessionId, progressData);

            // Update current session immediately
            const currentSession = useTranslationStore.getState().currentSession;
            if (currentSession && sessionId) {
              // Convert logs from backend format to frontend format
              const backendLogs = data.logs || [];
              const frontendLogs = backendLogs.map((log: any) => ({
                id: `${log.timestamp}-${log.message}`,
                timestamp: log.timestamp,
                level: log.level as "info" | "warning" | "error" | "debug" | "success",
                stage: log.stage,
                message: log.message,
                chunkId: log.chunkId || log.chunk_id,
                sessionId: log.sessionId || log.session_id,
                data: log.extraData || log.extra_data,
              }));
              
              const updatedSession: typeof currentSession = {
                ...currentSession,
                sessionId,
                progress: data.progress ?? currentSession.progress,
                currentStep:
                  data.currentStep ??
                  data.current_step ??
                  currentSession.currentStep,
                status: data.status ?? currentSession.status,
                message: data.message ?? currentSession.message,
                early_preview_available:
                  data.earlyPreviewAvailable ??
                  data.early_preview_available ??
                  currentSession.early_preview_available,
                early_preview_path:
                  data.earlyPreviewPath ??
                  data.early_preview_path ??
                  currentSession.early_preview_path,
                etaSeconds: data.etaSeconds ?? currentSession.etaSeconds,
                hardwareInfo: data.hardwareInfo ?? currentSession.hardwareInfo,
                availableSegments:
                  data.availableSegments ?? currentSession.availableSegments,
                processingSpeed:
                  data.processingSpeed ?? currentSession.processingSpeed,
                currentChunk: data.currentChunk ?? currentSession.currentChunk,
                totalChunks:
                  data.totalChunks !== undefined && data.totalChunks > 0
                    ? data.totalChunks
                    : currentSession.totalChunks !== undefined &&
                        currentSession.totalChunks > 0
                      ? currentSession.totalChunks
                      : (data.totalChunks ?? currentSession.totalChunks),
                logs: frontendLogs.length > 0 ? frontendLogs : (currentSession.logs || []),
                // Detailed progress fields (from backend)
                stage: data.stage ?? currentSession.stage,
                stage_number: data.stage_number ?? currentSession.stage_number,
                total_stages: data.total_stages ?? currentSession.total_stages,
                segments_processed: data.segments_processed ?? currentSession.segments_processed,
                current_time: data.current_time ?? currentSession.current_time,
                current_time_formatted: data.current_time_formatted ?? currentSession.current_time_formatted,
                total_duration: data.total_duration ?? currentSession.total_duration,
                total_duration_formatted: data.total_duration_formatted ?? currentSession.total_duration_formatted,
                progress_percent: data.progress_percent ?? currentSession.progress_percent,
                elapsed_time: data.elapsed_time ?? currentSession.elapsed_time,
              };

              setCurrentSessionRef.current(updatedSession);
            }
          }

          // Show toast notifications
          if (data.status === "completed") {
            // Ensure totalChunks is preserved when completing
            const completedSession =
              useTranslationStore.getState().currentSession;
            if (completedSession && completedSession.sessionId === sessionId) {
              const finalTotalChunks =
                completedSession.totalChunks && completedSession.totalChunks > 0
                  ? completedSession.totalChunks
                  : completedSession.currentChunk &&
                      completedSession.currentChunk > 0
                    ? completedSession.currentChunk
                    : 0;

              // Update with preserved totalChunks if we have a valid value
              if (
                finalTotalChunks > 0 &&
                (!completedSession.totalChunks ||
                  completedSession.totalChunks === 0)
              ) {
                setCurrentSessionRef.current({
                  ...completedSession,
                  totalChunks: finalTotalChunks,
                });
              }
            }
            toast.success("Translation completed!");
            es.close();
          } else if (data.status === "failed") {
            toast.error(
              `Translation failed: ${data.message ?? "Unknown error"}`
            );
            es.close();
          }
        } else {
          // Non-critical updates - use throttled updates with startTransition
          startTransition(() => {
            throttledUpdateProgress.current(sessionId, progressData);

            // Update current session if it matches
            const currentSession =
              useTranslationStore.getState().currentSession;
            if (currentSession && sessionId) {
              // Convert logs from backend format to frontend format
              const backendLogs = data.logs || [];
              
              // Debug: Log if we receive logs
              if (backendLogs.length > 0) {
                console.log(`📥 [FRONTEND SSE] Received ${backendLogs.length} logs from progress stream for session ${sessionId}`);
              }
              
              const frontendLogs = backendLogs.map((log: any) => ({
                id: `${log.timestamp}-${log.message}`,
                timestamp: log.timestamp,
                level: log.level as "info" | "warning" | "error" | "debug" | "success",
                stage: log.stage,
                message: log.message,
                chunkId: log.chunkId || log.chunk_id,
                sessionId: log.sessionId || log.session_id,
                data: log.extraData || log.extra_data,
              }));
              
              // Merge logs properly to avoid duplicates
              const mergeLogs = (existing: any[] | undefined, incoming: any[] | undefined) => {
                if (!incoming || incoming.length === 0) {
                  return existing || [];
                }
                
                const logMap = new Map<string, any>();
                
                // Add existing logs
                (existing || []).forEach(log => {
                  const key = log.id || `${log.timestamp}-${log.message}`;
                  logMap.set(key, log);
                });
                
                // Add new logs (override if same id, but won't create duplicates)
                incoming.forEach(log => {
                  const key = log.id || `${log.timestamp}-${log.message}`;
                  logMap.set(key, log);
                });
                
                // Convert back to array, sort by timestamp, limit to 200
                return Array.from(logMap.values())
                  .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
                  .slice(-200);
              };
              
              const updatedSession: typeof currentSession = {
                ...currentSession,
                sessionId,
                progress: data.progress ?? currentSession.progress,
                currentStep:
                  data.currentStep ??
                  data.current_step ??
                  currentSession.currentStep,
                status: data.status ?? currentSession.status,
                message: data.message ?? currentSession.message,
                early_preview_available:
                  data.earlyPreviewAvailable ??
                  data.early_preview_available ??
                  currentSession.early_preview_available,
                early_preview_path:
                  data.earlyPreviewPath ??
                  data.early_preview_path ??
                  currentSession.early_preview_path,
                etaSeconds: data.etaSeconds ?? currentSession.etaSeconds,
                hardwareInfo: data.hardwareInfo ?? currentSession.hardwareInfo,
                availableSegments:
                  data.availableSegments ?? currentSession.availableSegments,
                processingSpeed:
                  data.processingSpeed ?? currentSession.processingSpeed,
                currentChunk: data.currentChunk ?? currentSession.currentChunk,
                totalChunks:
                  data.totalChunks !== undefined && data.totalChunks > 0
                    ? data.totalChunks
                    : currentSession.totalChunks !== undefined &&
                        currentSession.totalChunks > 0
                      ? currentSession.totalChunks
                      : (data.totalChunks ?? currentSession.totalChunks),
                logs: mergeLogs(currentSession.logs, frontendLogs),
                // Detailed progress fields (from backend)
                stage: data.stage ?? currentSession.stage,
                stage_number: data.stage_number ?? currentSession.stage_number,
                total_stages: data.total_stages ?? currentSession.total_stages,
                segments_processed: data.segments_processed ?? currentSession.segments_processed,
                current_time: data.current_time ?? currentSession.current_time,
                current_time_formatted: data.current_time_formatted ?? currentSession.current_time_formatted,
                total_duration: data.total_duration ?? currentSession.total_duration,
                total_duration_formatted: data.total_duration_formatted ?? currentSession.total_duration_formatted,
                progress_percent: data.progress_percent ?? currentSession.progress_percent,
                elapsed_time: data.elapsed_time ?? currentSession.elapsed_time,
              };

              throttledUpdateSession.current(updatedSession);
            }
          });
        }
      } catch (error) {
        console.error("Error parsing SSE message:", error);
      }
    };

    es.onerror = error => {
      // Silently handle SSE connection errors (expected when no active session)
      try {
        setIsConnected(false);
        es.close();
      } catch (e) {
        // Ignore cleanup errors
      }
    };
  }, []); // Empty dependency array to prevent infinite loops

  const disconnect = useCallback(() => {
    // Clear any pending connection timeout
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = null;
    }

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setEventSource(null);
      setIsConnected(false);
    }

    lastSessionIdRef.current = null;
  }, []); // Empty dependency array to prevent infinite loops

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Cancel any pending throttled updates
      throttledUpdateProgress.current.cancel();
      throttledUpdateSession.current.cancel();

      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []); // Empty dependency array

  return {
    isConnected,
    connectToSession,
    disconnect,
  };
}
