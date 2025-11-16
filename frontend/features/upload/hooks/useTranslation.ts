"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useTranslationStore } from "@/stores/translationStore";
import { apiClient } from "@/lib/api-client";
import { API_ENDPOINTS } from "@/constants";
import toast from "react-hot-toast";

export function useTranslation() {
  const router = useRouter();
  const { currentSession } = useTranslationStore();
  const [isStartingTranslation, setIsStartingTranslation] = useState(false);

  const handleStartTranslation = useCallback(async () => {
    console.log("🟢🟢🟢 [FRONTEND] handleStartTranslation CALLED");
    console.log("   Current session:", currentSession);
    
    if (!currentSession?.sessionId) {
      console.error("❌ [FRONTEND] No session found!");
      toast.error("No session found. Please upload a file first.");
      return;
    }

    const sessionId = currentSession.sessionId;
    const endpoint = API_ENDPOINTS.TRANSLATE(sessionId);
    
    console.log("🟢 [FRONTEND] Starting translation request");
    console.log("   Session ID:", sessionId);
    console.log("   Endpoint:", endpoint);
    console.log("   Full URL:", endpoint);

    try {
      setIsStartingTranslation(true);
      console.log("🟢 [FRONTEND] Making POST request to:", endpoint);
      const startTime = Date.now();
      
      const response = await apiClient.post(
        endpoint,
        {}
      );
      
      const duration = Date.now() - startTime;
      console.log("🟢 [FRONTEND] POST request completed");
      console.log("   Response:", response);
      console.log("   Duration:", duration, "ms");
      
      toast.success("Translation started!");
      console.log("🟢 [FRONTEND] Navigating to /processing");
      router.push("/processing");
    } catch (error) {
      console.error("❌❌❌ [FRONTEND] Translation error:", error);
      console.error("   Error type:", error instanceof Error ? error.constructor.name : typeof error);
      console.error("   Error message:", error instanceof Error ? error.message : String(error));
      if (error instanceof Error && error.stack) {
        console.error("   Stack trace:", error.stack);
      }
      const errorMessage =
        error instanceof Error ? error.message : "Failed to start translation";
      toast.error(errorMessage);
      // Only reset loading state on error, navigation will clear it on success
      setIsStartingTranslation(false);
    }
  }, [currentSession, router]);

  return {
    handleStartTranslation,
    isStartingTranslation,
  };
}
