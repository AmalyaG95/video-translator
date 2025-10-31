"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import { ProcessingService } from "../services/processingService";
import { useTranslationStore } from "@/stores/translationStore";

export function useProcessingActions(session: any) {
  const router = useRouter();
  const [isViewingResults, setIsViewingResults] = useState(false);

  const handleStartTranslation = useCallback(async () => {
    if (!session?.sessionId) return;

    try {
      const result = await ProcessingService.startTranslation(
        session.sessionId
      );

      if (result.success) {
        // Update session status to processing
        useTranslationStore.getState().setCurrentSession({
          ...session,
          status: "processing",
          progress: 0,
          currentStep: "Starting translation...",
        });
        toast.success("Translation started!");
      } else {
        toast.error(result.message || "Failed to start translation");
      }
    } catch (error) {
      console.error("Translation error:", error);
      toast.error("Failed to start translation");
    }
  }, [session]);

  const handleViewResults = useCallback(() => {
    if (!session?.sessionId || isViewingResults) return;
    setIsViewingResults(true);
    router.push(`/results/${session.sessionId}`);
  }, [session, router, isViewingResults]);

  const handlePause = useCallback(async () => {
    if (!session?.sessionId) return;

    try {
      const result = await ProcessingService.pauseTranslation(
        session.sessionId
      );

      if (result.success) {
        // Update session state
        useTranslationStore.getState().setCurrentSession({
          ...session,
          isPaused: true,
          status: "paused",
        });
        toast.success("Translation paused");
      } else {
        toast.error(result.message || "Failed to pause translation");
      }
    } catch (error) {
      console.error("Pause error:", error);
      toast.error("Failed to pause translation");
    }
  }, [session]);

  const handleResume = useCallback(async () => {
    if (!session?.sessionId) return;

    try {
      const result = await ProcessingService.resumeTranslation(
        session.sessionId
      );

      if (result.success) {
        // Update session state
        useTranslationStore.getState().setCurrentSession({
          ...session,
          isPaused: false,
          status: "processing",
        });
        toast.success("Translation resumed");
      } else {
        toast.error(result.message || "Failed to resume translation");
      }
    } catch (error) {
      console.error("Resume error:", error);
      toast.error("Failed to resume translation");
    }
  }, [session]);

  const handleCancel = useCallback(async () => {
    if (!session?.sessionId) return;

    try {
      const result = await ProcessingService.cancelTranslation(
        session.sessionId
      );

      if (result.success) {
        // Remove session from history
        useTranslationStore.getState().removeSession(session.sessionId);

        // Clear current session and navigate back
        useTranslationStore.getState().setCurrentSession(null);
        toast.success("Translation cancelled");
        router.push("/");
      } else {
        toast.error(result.message || "Failed to cancel translation");
      }
    } catch (error) {
      console.error("Cancel error:", error);
      toast.error("Failed to cancel translation");
    }
  }, [session, router]);

  return {
    handleStartTranslation,
    handleViewResults,
    handlePause,
    handleResume,
    handleCancel,
    isViewingResults,
  };
}
