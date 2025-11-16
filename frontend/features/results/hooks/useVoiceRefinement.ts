"use client";

import { useCallback } from "react";
import toast from "react-hot-toast";
import { resultsService } from "../services/resultsService";

export function useVoiceRefinement() {
  const handleRefineVoice = useCallback(async (sessionId: string) => {
    try {
      const result = await resultsService.refineVoice(sessionId);
      toast.success(result.message);
    } catch (error) {
      console.error("Voice refinement error:", error);
      toast.error("Failed to start voice refinement");
    }
  }, []);

  return { handleRefineVoice };
}










