"use client";

import React, { useEffect } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useTranslationStore } from "@/stores/translationStore";
import { parseAIReasoningLogs } from "../../utils/aiLogsParser";
import { SESSION_REFRESH_INTERVAL } from "../../constants";

// Custom hooks
import {
  useSessionValidation,
  useEarlyPreview,
  useSessionRefresh,
  useProcessingActions,
  useSSE,
  useLogsStream,
} from "../../hooks";

// Components
import NoActiveSession from "./NoActiveSession/NoActiveSession";
import SampleChunkPreview from "./SampleChunkPreview/SampleChunkPreview";
import ProcessingTimeline from "./ProcessingTimeline/ProcessingTimeline";
import DynamicETA from "./DynamicETA/DynamicETA";

// Dynamic imports for heavy components
const DynamicProcessLogsPanel = dynamic(
  () => import("./ProcessLogsPanel/ProcessLogsPanel"),
  {
    loading: () => (
      <div className="flex items-center justify-center p-4">
        <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-blue-600"></div>
      </div>
    ),
    ssr: false,
  }
);

const DynamicAIReasoningPanel = dynamic(
  () => import("./AIReasoningPanel/AIReasoningPanel"),
  {
    loading: () => (
      <div className="flex items-center justify-center p-4">
        <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-blue-600"></div>
      </div>
    ),
    ssr: false,
  }
);

function ProcessingScreen() {
  const router = useRouter();
  const { currentSession } = useTranslationStore();

  // Custom hooks
  const { isValid } = useSessionValidation(currentSession?.sessionId);
  const {
    earlyPreviewAvailable,
    setEarlyPreviewAvailable,
    earlyPreviewUrl,
    previewDuration,
    setPreviewDuration,
    handleEarlyPreview,
  } = useEarlyPreview(currentSession?.sessionId);
  const { handleForceRefresh } = useSessionRefresh(currentSession?.sessionId);
  const {
    handleStartTranslation,
    handleViewResults,
    handlePause,
    handleResume,
    handleCancel,
    isViewingResults,
  } = useProcessingActions(currentSession);

  // Existing hooks
  const { connectToSession, disconnect } = useSSE();
  const { logs, error } = useLogsStream(currentSession?.sessionId || null);

  // Enable SSE connection for real-time updates
  useEffect(() => {
    if (currentSession?.sessionId) {
      connectToSession(currentSession.sessionId);
    }

    return () => {
      disconnect();
    };
  }, [currentSession?.sessionId, connectToSession, disconnect]);

  // Enable error handling
  useEffect(() => {
    if (currentSession?.sessionId && error) {
      useTranslationStore.getState().setCurrentSession(null);
      router.push("/");
    }
  }, [currentSession?.sessionId, error, router]);

  // Monitor for early preview availability and auto-fetch
  useEffect(() => {
    if (currentSession?.early_preview_available && !earlyPreviewAvailable) {
      setEarlyPreviewAvailable(true);
      // Automatically fetch and show preview
      handleEarlyPreview();
    }
  }, [
    currentSession?.early_preview_available,
    earlyPreviewAvailable,
    handleEarlyPreview,
  ]);

  // Auto-start translation when session is uploaded (but not if cancelled)
  useEffect(() => {
    if (
      currentSession?.status === "uploaded" &&
      !currentSession.isPaused &&
      currentSession.status !== "cancelled" &&
      currentSession.status !== "failed"
    ) {
      handleStartTranslation();
    }
  }, [
    currentSession?.status,
    currentSession?.isPaused,
    handleStartTranslation,
  ]);

  // Periodic session refresh for completed sessions
  useEffect(() => {
    if (!currentSession?.sessionId) return;

    let intervalId: NodeJS.Timeout | null = null;

    const startPolling = () => {
      intervalId = setInterval(async () => {
        try {
          const response = await fetch(
            `http://localhost:3001/sessions/${currentSession.sessionId}`
          );

          // Session doesn't exist (404) - stop polling and clear current session
          if (response.status === 404) {
            if (intervalId) {
              clearInterval(intervalId);
              intervalId = null;
            }
            // Clear the current session from store
            useTranslationStore.getState().setCurrentSession(null);
            return;
          }

          // Only update if session exists (200 OK)
          if (response.ok) {
            const sessionData = await response.json();

            useTranslationStore
              .getState()
              .updateSessionProgress(currentSession.sessionId, {
                status: sessionData.status,
                progress: sessionData.progress,
                currentStep: sessionData.currentStep,
                currentChunk: sessionData.currentChunk,
                totalChunks: sessionData.totalChunks,
                message: sessionData.message,
              });
          }
        } catch (error) {
          // Connection refused or other network errors - stop polling
          if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
          }
          // Clear the current session from store on connection errors
          useTranslationStore.getState().setCurrentSession(null);
        }
      }, SESSION_REFRESH_INTERVAL);
    };

    startPolling();

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [currentSession?.sessionId, currentSession?.status]);

  // Try to restore current session from recent sessions if none exists
  useEffect(() => {
    if (
      !currentSession &&
      useTranslationStore.getState().recentSessions.length > 0
    ) {
      const mostRecentSession =
        useTranslationStore.getState().recentSessions[0];
      if (mostRecentSession) {
        useTranslationStore.getState().setCurrentSession(mostRecentSession);
      }
    }
  }, [currentSession]);

  // Use useEffect to handle navigation to avoid setState during render
  useEffect(() => {
    if (!currentSession) {
      router.push("/");
    }
  }, [currentSession, router]);

  // Parse AI reasoning logs from the logs stream
  const aiReasoningLogs = parseAIReasoningLogs(logs);

  // Handle case when no current session exists - use conditional rendering instead of early return
  if (!currentSession) {
    return <NoActiveSession onNavigateHome={() => router.push("/")} />;
  }

  return (
    <div className="space-y-6">
      {/* Unified Processing Timeline */}
      <ProcessingTimeline
        session={currentSession}
        onStart={handleStartTranslation}
        onPause={handlePause}
        onResume={handleResume}
        onCancel={handleCancel}
        onViewResults={handleViewResults}
      />

      {/*ETA and Hardware Info + Main Content: Sidebar*/}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/*ETA and Hardware Info */}
        <DynamicETA
          etaSeconds={currentSession.etaSeconds}
          hardwareInfo={currentSession.hardwareInfo}
          processingSpeed={currentSession.processingSpeed}
          currentChunk={currentSession.currentChunk}
          totalChunks={currentSession.totalChunks}
          isPaused={currentSession.isPaused}
          status={currentSession.status}
          progress={currentSession.progress}
        />

        {/* Sample Chunk Preview Section */}
        <SampleChunkPreview
          earlyPreviewUrl={earlyPreviewUrl}
          earlyPreviewAvailable={earlyPreviewAvailable}
          previewDuration={previewDuration}
          onDurationChange={setPreviewDuration}
        />
      </div>

      {/* Process Logs Panel */}
      <DynamicProcessLogsPanel logs={logs} error={error} />

      {/* AI Reasoning Panel */}
      <div className="lg:col-span-2">
        <DynamicAIReasoningPanel logs={aiReasoningLogs} />
      </div>
    </div>
  );
}

export default ProcessingScreen;
