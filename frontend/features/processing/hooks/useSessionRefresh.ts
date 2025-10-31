import { useCallback } from "react";
import toast from "react-hot-toast";
import { ProcessingService } from "../services/processingService";
import { useTranslationStore } from "@/stores/translationStore";

export function useSessionRefresh(sessionId: string | undefined) {
  const handleForceRefresh = useCallback(async () => {
    if (!sessionId) return;

    try {
      // Clear sessionStorage cache to prevent stale data
      sessionStorage.removeItem("translation-store");

      const sessionData = await ProcessingService.getSession(sessionId);

      // Force update the store with backend data
      const store = useTranslationStore.getState();
      const currentSession = store.currentSession;

      if (currentSession) {
        // Preserve totalChunks if session is completed and backend doesn't have it
        const preservedTotalChunks =
          sessionData.totalChunks !== undefined && sessionData.totalChunks > 0
            ? sessionData.totalChunks
            : currentSession.totalChunks !== undefined &&
                currentSession.totalChunks > 0
              ? currentSession.totalChunks
              : (sessionData.totalChunks ?? currentSession.totalChunks);

        store.setCurrentSession({
          ...currentSession,
          status: sessionData.status,
          progress: sessionData.progress,
          currentStep: sessionData.currentStep,
          currentChunk: sessionData.currentChunk ?? currentSession.currentChunk,
          totalChunks: preservedTotalChunks,
          message: sessionData.message,
        });
      }

      toast.success("Session data refreshed and cache cleared!");
    } catch (error) {
      console.error("Failed to force refresh session:", error);
      toast.error("Failed to refresh session data");
    }
  }, [sessionId]);

  return { handleForceRefresh };
}
