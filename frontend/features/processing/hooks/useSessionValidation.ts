"use client";

import { useEffect, useState } from "react";
import { ProcessingService } from "../services/processingService";
import { useTranslationStore } from "@/stores/translationStore";

export function useSessionValidation(sessionId: string | undefined) {
  const [isValid, setIsValid] = useState<boolean | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [hasFailed, setHasFailed] = useState(false);

  useEffect(() => {
    // Reset failed state when sessionId changes
    setHasFailed(false);
    
    if (!sessionId) {
      setIsValid(null);
      return;
    }

    // Don't validate if we've already determined the session doesn't exist
    // (This check is now handled by the cleanup and reset above)

    let isMounted = true;

    const validateSession = async () => {
      setIsValidating(true);
      try {
        const valid = await ProcessingService.validateSession(sessionId);
        
        // Only update state if component is still mounted
        if (!isMounted) return;

        setIsValid(valid);

        // If session is invalid, mark as failed and clear it from the store
        if (!valid) {
          setHasFailed(true);
          useTranslationStore.getState().setCurrentSession(null);
        }
      } catch (error) {
        // Silently handle validation errors (404, etc.)
        if (!isMounted) return;
        setHasFailed(true);
        setIsValid(false);
        useTranslationStore.getState().setCurrentSession(null);
      } finally {
        if (isMounted) {
          setIsValidating(false);
        }
      }
    };

    validateSession();

    // Cleanup function
    return () => {
      isMounted = false;
    };
  }, [sessionId]); // Removed hasFailed from dependencies to prevent infinite loop

  return { isValid, isValidating };
}
