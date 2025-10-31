
import type { ProcessingSession } from "@/shared/types";

export type HistoryScreenProps = {
  sessions?: ProcessingSession[];
  onSessionSelect?: (session: ProcessingSession) => void;
  onSessionDelete?: (sessionId: string) => void;
  onRefresh?: () => void;
};

export type SessionCardProps = {
  session: ProcessingSession;
  onSelect?: (session: ProcessingSession) => void;
  onDelete?: (sessionId: string) => void;
  isSelected?: boolean;
};

