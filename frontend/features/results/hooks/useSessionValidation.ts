"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useTranslationStore } from "@/stores/translationStore";
import { isSessionComplete } from "../utils/validators";

export function useSessionValidation() {
  const router = useRouter();
  const currentSession = useTranslationStore(state => state.currentSession);

  useEffect(() => {
    if (!isSessionComplete(currentSession)) {
      router.push("/");
    }
  }, [currentSession, router]);

  return {
    currentSession,
    isValid: isSessionComplete(currentSession),
  };
}






