"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useTranslationStore } from "@/stores/translationStore";
import SessionCard from "../SessionCard/SessionCard";
import type { ProcessingSession, SessionStatus } from "@/shared/types/session.types";

interface Session {
  sessionId: string;
  status: string;
  sourceLang?: string;
  targetLang?: string;
  progress?: number;
  createdAt?: Date | string;
  updatedAt?: Date | string;
  currentStep?: string;
  message?: string;
  fileName?: string;
  fileSize?: number;
  duration?: number;
}

interface SessionsListProps {
  sessions: Session[];
  getStatusIcon: (status: string) => React.ReactNode;
  getStatusColor: (status: string) => string;
}

export default function SessionsList({
  sessions,
  getStatusIcon,
  getStatusColor,
}: SessionsListProps) {
  const router = useRouter();
  const { removeSession, setCurrentSession } = useTranslationStore();
  const [viewingSessions, setViewingSessions] = useState<Set<string>>(
    new Set()
  );

  const handleViewSession = async (session: Session) => {
    setViewingSessions(prev => new Set(prev).add(session.sessionId));
    try {
      // Convert Session to ProcessingSession with required fields
      const processingSession: ProcessingSession = {
        sessionId: session.sessionId,
        status: session.status as SessionStatus,
        progress: session.progress ?? 0,
        currentStep: session.currentStep ?? "initialization",
        message: session.message ?? "",
        sourceLang: session.sourceLang ?? "en",
        targetLang: session.targetLang ?? "en",
        fileName: session.fileName,
        fileSize: session.fileSize,
        duration: session.duration,
        createdAt: session.createdAt ? (typeof session.createdAt === 'string' ? new Date(session.createdAt) : session.createdAt) : undefined,
        updatedAt: session.updatedAt ? (typeof session.updatedAt === 'string' ? new Date(session.updatedAt) : session.updatedAt) : undefined,
      };
      setCurrentSession(processingSession);
      // Navigate based on session status
      if (session.status === "completed") {
        router.push(`/results/${session.sessionId}`);
      } else {
        router.push("/processing");
      }
    } finally {
      // State will be reset on unmount when navigating
      setTimeout(() => {
        setViewingSessions(prev => {
          const newSet = new Set(prev);
          newSet.delete(session.sessionId);
          return newSet;
        });
      }, 500);
    }
  };

  const handleDownload = async (sessionId: string) => {
    try {
      const response = await fetch(
        `http://localhost:3001/download/${sessionId}`
      );
      if (!response.ok) {
        throw new Error(`Failed to download video: ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);

      const downloadWindow = window.open(
        "",
        "_blank",
        "width=1,height=1,left=0,top=0"
      );
      if (downloadWindow) {
        downloadWindow.document.write(`
          <html>
            <head><title>Download</title></head>
            <body>
              <script>
                const a = document.createElement('a');
                a.href = '${url}';
                a.download = 'translated_video_${sessionId}.mp4';
                document.body.appendChild(a);
                a.click();
                setTimeout(() => {
                  window.close();
                }, 100);
              </script>
            </body>
          </html>
        `);
        downloadWindow.document.close();
      } else {
        const a = document.createElement("a");
        a.href = url;
        a.download = `translated_video_${sessionId}.mp4`;
        a.target = "_blank";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }

      setTimeout(() => {
        window.URL.revokeObjectURL(url);
      }, 1000);

      console.log("Download completed successfully");
    } catch (error) {
      console.error("Download failed:", error);
    }
  };

  const handleDelete = async (sessionId: string) => {
    try {
      await new Promise(resolve => setTimeout(resolve, 300));
      removeSession(sessionId);
    } catch (error) {
      console.error("Delete failed:", error);
    }
  };

  return (
    <div className="grid grid-cols-1 gap-6">
      {sessions.map(session => (
        <SessionCard
          key={session.sessionId}
          session={session}
          onView={handleViewSession}
          onDownload={handleDownload}
          onDelete={handleDelete}
          getStatusIcon={getStatusIcon}
          getStatusColor={getStatusColor}
          isViewing={viewingSessions.has(session.sessionId)}
        />
      ))}
    </div>
  );
}
