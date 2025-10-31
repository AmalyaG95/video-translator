"use client";

import { useState, useEffect, useCallback } from "react";
import { useTranslationStore } from "@/stores/translationStore";

interface AIInsight {
  id: string;
  type: "decision" | "optimization" | "warning" | "success";
  title: string;
  description: string;
  impact: "high" | "medium" | "low";
  timestamp: string;
  sessionId?: string;
  data?: any;
}

interface UseAIInsightsReturn {
  insights: AIInsight[];
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

export function useAIInsights(sessionId: string | null): UseAIInsightsReturn {
  const { aiInsights, setAIInsights } = useTranslationStore();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchInsights = useCallback(async () => {
    if (!sessionId) {
      // Don't clear insights when no session - keep historical data
      // Return existing insights from store
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/ai-insights/${sessionId}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch AI insights: ${response.status}`);
      }

      const data = await response.json();
      const newInsights = data.insights || [];

      // Append new insights to existing ones, avoiding duplicates
      const existingIds = new Set(aiInsights.map((insight: any) => insight.id));
      const uniqueNewInsights = newInsights
        .filter((insight: any) => !existingIds.has(insight.id))
        .map((insight: any) => ({ ...insight, sessionId: sessionId }));
      setAIInsights([...aiInsights, ...uniqueNewInsights]);
    } catch (err) {
      console.error("Error fetching AI insights:", err);
      setError(
        err instanceof Error ? err.message : "Failed to fetch AI insights"
      );
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, setAIInsights]);

  useEffect(() => {
    fetchInsights();
  }, [fetchInsights]);

  // Filter insights by sessionId if provided
  const filteredInsights = sessionId
    ? aiInsights.filter(insight => insight.sessionId === sessionId)
    : aiInsights;

  return {
    insights: filteredInsights,
    isLoading,
    error,
    refetch: fetchInsights,
  };
}
