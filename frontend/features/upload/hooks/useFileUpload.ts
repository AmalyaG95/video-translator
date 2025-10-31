"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslationStore } from "@/stores/translationStore";
import { apiClient } from "@/lib/api-client";
import { API_ENDPOINTS } from "@/constants";
import { UploadResponse } from "../types";
import toast from "react-hot-toast";

export function useFileUpload() {
  const router = useRouter();
  const { setCurrentSession, addSession } = useTranslationStore();
  const [videoMetadata, setVideoMetadata] = useState<any>(null);
  const [chunkingInfo, setChunkingInfo] = useState<any>(null);

  const handleFileUpload = useCallback(
    async (file: File, sourceLanguage: string, targetLanguage: string) => {
      try {
        // Clear stale data
        setChunkingInfo(null);

        if (!sourceLanguage || !targetLanguage) {
          toast.error("Please select both source and target languages");
          return;
        }

        console.log("[Upload] Sending languages:", {
          sourceLanguage,
          targetLanguage,
        });

        const data = await apiClient.uploadFile<UploadResponse>(
          "/upload",
          file,
          {
            sourceLang: sourceLanguage,
            targetLang: targetLanguage,
          }
        );

        console.log("[Upload] Backend returned:", {
          sourceLang: data.sourceLang,
          targetLang: data.targetLang,
        });

        const newSession = {
          sessionId: data.sessionId,
          status: (data.status ?? "uploaded") as
            | "uploaded"
            | "processing"
            | "completed"
            | "failed"
            | "paused"
            | "cancelled",
          progress: data.progress ?? 0,
          currentStep: data.currentStep ?? "Upload complete",
          message: data.message ?? "",
          sourceLang: data.sourceLang ?? sourceLanguage,
          targetLang: data.targetLang ?? targetLanguage,
          filePath: data.filePath,
          fileName: data.fileName,
          fileSize: data.fileSize,
          createdAt: new Date(data.createdAt ?? Date.now()),
          updatedAt: new Date(data.updatedAt ?? Date.now()),
          totalChunks: 0, // Will be set by Python ML service via gRPC
          currentChunk: data.currentChunk ?? 0,
          etaSeconds: data.etaSeconds ?? 0,
          processingSpeed: data.processingSpeed ?? 0,
          hardwareInfo: data.hardwareInfo ?? {
            cpu: "Unknown",
            gpu: "Unknown",
            vram_gb: 0,
            ram_gb: 0,
          },
          availableSegments: data.availableSegments ?? [],
          early_preview_available: data.early_preview_available ?? false,
          early_preview_path: data.early_preview_path ?? "",
          isPaused: data.isPaused ?? false,
        };

        console.log("[Upload] Created session with languages:", {
          sourceLang: newSession.sourceLang,
          targetLang: newSession.targetLang,
          backendReturned: { source: data.sourceLang, target: data.targetLang },
          frontendSent: { source: sourceLanguage, target: targetLanguage },
        });

        setCurrentSession(newSession);
        addSession(newSession);

        if (videoMetadata) {
          try {
            await apiClient.patch(
              `/sessions/${newSession.sessionId}/metadata`,
              videoMetadata
            );
          } catch (error) {
            console.warn("Failed to send metadata to backend:", error);
          }
        }

        return newSession;
      } catch (error) {
        console.error("Upload error:", error);
        const errorMessage =
          error instanceof Error ? error.message : "Failed to upload file";
        toast.error(`Upload failed: ${errorMessage}`);
        throw error;
      }
    },
    [videoMetadata, setCurrentSession, addSession]
  );

  const handleMetadataDetected = useCallback(
    async (metadata: any, currentSession: any) => {
      setVideoMetadata(metadata);

      if (currentSession) {
        setCurrentSession({
          ...currentSession,
          totalChunks:
            currentSession.totalChunks ||
            (metadata.duration ? Math.ceil(metadata.duration / 30) : 0),
        });

        try {
          await apiClient.patch(
            `/sessions/${currentSession.sessionId}/metadata`,
            metadata
          );
        } catch (error) {
          console.warn("Failed to send metadata to backend:", error);
        }
      }
    },
    [setCurrentSession]
  );

  const handleFileUnload = useCallback(() => {
    setVideoMetadata(null);
    setChunkingInfo(null);
    setCurrentSession(null);
  }, [setCurrentSession]);

  return {
    videoMetadata,
    chunkingInfo,
    setChunkingInfo,
    handleFileUpload,
    handleMetadataDetected,
    handleFileUnload,
  };
}
