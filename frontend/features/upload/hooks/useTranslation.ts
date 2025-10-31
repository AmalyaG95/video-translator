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
    if (!currentSession?.sessionId) {
      toast.error("No session found. Please upload a file first.");
      return;
    }

    try {
      setIsStartingTranslation(true);
      await apiClient.post(
        API_ENDPOINTS.TRANSLATE(currentSession.sessionId),
        {}
      );
      toast.success("Translation started!");
      router.push("/processing");
    } catch (error) {
      console.error("Translation error:", error);
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
