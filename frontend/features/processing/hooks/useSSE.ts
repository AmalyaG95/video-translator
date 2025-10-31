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

  // Throttled update functions to prevent excessive re-renders
  const throttledUpdateProgress = useRef(
    throttle(
      (sessionId: string, data: any) => {
        updateSessionProgressRef.current(sessionId, data);
      },
      500,
      { leading: true, trailing: true }
    )
  );

  const throttledUpdateSession = useRef(
    throttle(
      (session: any) => {
        setCurrentSessionRef.current(session);
      },
      500,
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
          availableSegments: data.availableSegments,
          processingSpeed: data.processingSpeed,
          currentChunk: data.currentChunk,
          totalChunks: data.totalChunks,
        };

        // Check if this is a critical status change
        const isCritical =
          data.status === "completed" || data.status === "failed";

        if (isCritical) {
          // Critical updates - immediate, no throttle
          updateSessionProgressRef.current(sessionId, progressData);

          // Update current session immediately
          const currentSession = useTranslationStore.getState().currentSession;
          if (currentSession && sessionId) {
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
            };

            setCurrentSessionRef.current(updatedSession);
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
