"use client";

import { useState, useEffect, useCallback } from "react";
import toast from "react-hot-toast";
import { ProcessingService } from "../services/processingService";

export function useEarlyPreview(sessionId: string | undefined) {
  const [earlyPreviewAvailable, setEarlyPreviewAvailable] = useState(false);
  const [earlyPreviewUrl, setEarlyPreviewUrl] = useState<string | null>(null);
  const [previewDuration, setPreviewDuration] = useState<number>(0);

  const handleEarlyPreview = useCallback(async () => {
    if (!sessionId) return;

    try {
      const blob = await ProcessingService.getEarlyPreview(sessionId);

      // Clean up old URL if exists
      if (earlyPreviewUrl) {
        URL.revokeObjectURL(earlyPreviewUrl);
      }

      const url = URL.createObjectURL(blob);
      setEarlyPreviewUrl(url);
      toast.success("Early preview available!");
    } catch (error) {
      console.error("Failed to load early preview:", error);
      toast.error("Failed to load early preview");
    }
  }, [sessionId, earlyPreviewUrl]);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      if (earlyPreviewUrl) {
        URL.revokeObjectURL(earlyPreviewUrl);
      }
    };
  }, [earlyPreviewUrl]);

  return {
    earlyPreviewAvailable,
    setEarlyPreviewAvailable,
    earlyPreviewUrl,
    previewDuration,
    setPreviewDuration,
    handleEarlyPreview,
  };
}
