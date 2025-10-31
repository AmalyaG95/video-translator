"use client";

import { useState, useCallback } from "react";
import { useTranslationStore } from "@/stores/translationStore";
import { apiClient } from "@/lib/api-client";
import { API_ENDPOINTS } from "@/constants";
import toast from "react-hot-toast";

export function useLanguageDetection() {
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null);
  const { setSourceLanguage, setCurrentSession, currentSession } =
    useTranslationStore();

  const handleLanguageDetected = useCallback(
    (language: string, _confidence: number) => {
      setSourceLanguage(language);
      useTranslationStore.getState().setSourceLanguage(language);
    },
    [setSourceLanguage]
  );

  const detectLanguage = useCallback(
    async (sessionId: string) => {
      console.log(
        `[Language Detection] Starting detection for session: ${sessionId}`
      );

      const maxTimeout = setTimeout(() => {
        setIsDetecting(false);
        setSourceLanguage("en");
        console.log(
          `[Language Detection] Timeout reached, defaulting to English`
        );
      }, 35_000); // Increased to 35 seconds

      try {
        setIsDetecting(true);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
          controller.abort();
        }, 30_000); // Increased to 30 seconds

        console.log(
          `[Language Detection] Calling endpoint: ${API_ENDPOINTS.LANGUAGE_DETECTION_SESSION(sessionId)}`
        );

        const result = await apiClient.post<{
          detected_language: string;
          confidence: number;
          success: boolean;
          message: string;
        }>(
          API_ENDPOINTS.LANGUAGE_DETECTION_SESSION(sessionId),
          {}, // No payload needed - backend gets file from session
          {
            signal: controller.signal,
          }
        );

        console.log(`[Language Detection] Response received:`, result);

        clearTimeout(timeoutId);
        clearTimeout(maxTimeout);

        if (result?.success && result?.detected_language) {
          setDetectedLanguage(result.detected_language);
          setSourceLanguage(result.detected_language);

          // Update current session with detected language
          if (currentSession) {
            setCurrentSession({
              ...currentSession,
              sourceLang: result.detected_language,
            });
          }

          console.log(
            `[Language Detection] Success: ${result.detected_language} (${Math.round(result.confidence * 100)}% confidence)`
          );
          toast.success(
            `Language detected: ${result.detected_language} (${Math.round(result.confidence * 100)}% confidence)`
          );
        } else {
          console.log(
            `[Language Detection] Failed or no language detected:`,
            result?.message
          );
          setSourceLanguage("en");

          // Update current session with default language
          if (currentSession) {
            setCurrentSession({
              ...currentSession,
              sourceLang: "en",
            });
          }
          toast(
            `Language detection unavailable: ${result?.message || "Using English as default"}`,
            {
              icon: "ℹ️",
            }
          );
        }
      } catch (error) {
        clearTimeout(maxTimeout);
        console.error(`[Language Detection] Error:`, error);

        if (error instanceof Error && error.name === "AbortError") {
          console.warn("Language detection was aborted (timeout)");
          toast("Language detection timed out, using English as default", {
            icon: "⏰",
          });
        } else {
          console.warn("Language detection failed:", error);
          toast("Language detection failed, using English as default", {
            icon: "⚠️",
          });
        }
        setSourceLanguage("en");
      } finally {
        setIsDetecting(false);
      }
    },
    [setSourceLanguage, currentSession, setCurrentSession]
  );

  return {
    isDetecting,
    detectedLanguage,
    setDetectedLanguage,
    handleLanguageDetected,
    detectLanguage,
  };
}
