interface LogEntry {
  timestamp: string;
  level: string;
  stage: string;
  message: string;
  data?: any;
  chunkId?: string;
}

interface AIReasoningLog {
  id: string;
  timestamp: string;
  stage: string;
  type: "info" | "warning" | "decision" | "error";
  message: string;
  data?: any;
}

/**
 * Parse logs to extract AI reasoning insights
 * Identifies decisions, warnings, errors, and important information
 */
export function parseAIReasoningLogs(logs: LogEntry[]): AIReasoningLog[] {
  if (!logs || logs.length === 0) {
    return [];
  }

  const aiLogs: AIReasoningLog[] = [];

  logs.forEach((log, index) => {
    const message = log.message || "";
    const level = log.level?.toLowerCase() || "";
    const stage = log.stage || "unknown";

    // Determine log type based on level and message content
    let type: "info" | "warning" | "decision" | "error" = "info";

    // Errors
    if (level === "error" || message.toLowerCase().includes("error") || message.toLowerCase().includes("failed")) {
      type = "error";
    }
    // Warnings
    else if (level === "warning" || message.toLowerCase().includes("warning") || message.toLowerCase().includes("caution")) {
      type = "warning";
    }
    // Decisions - look for decision-making keywords
    else if (
      message.toLowerCase().includes("decided") ||
      message.toLowerCase().includes("chose") ||
      message.toLowerCase().includes("selected") ||
      message.toLowerCase().includes("using") ||
      message.toLowerCase().includes("switched") ||
      message.toLowerCase().includes("fallback") ||
      message.toLowerCase().includes("recovery") ||
      message.toLowerCase().includes("detected") ||
      message.toLowerCase().includes("language mismatch") ||
      message.toLowerCase().includes("voice") ||
      message.toLowerCase().includes("model") ||
      message.toLowerCase().includes("strategy")
    ) {
      type = "decision";
    }

    // Filter for meaningful insights (skip routine progress messages)
    const isRoutineProgress = 
      message.toLowerCase().includes("progress") ||
      message.toLowerCase().includes("processing") ||
      message.toLowerCase().includes("completed") ||
      message.toLowerCase().includes("started") ||
      message.toLowerCase().includes("stage") ||
      message.toLowerCase().includes("chunk") ||
      message.toLowerCase().includes("segment") ||
      message.toLowerCase().includes("yielding") ||
      message.toLowerCase().includes("grpc stream");

    // Include if it's an error, warning, decision, or important info (not routine progress)
    if (type !== "info" || (!isRoutineProgress && message.length > 20)) {
      aiLogs.push({
        id: `ai-log-${log.timestamp}-${index}`,
        timestamp: log.timestamp,
        stage: stage,
        type: type,
        message: message,
        data: log.data || null,
      });
    }
  });

  // Sort by timestamp (newest first)
  return aiLogs.sort((a, b) => {
    const timeA = new Date(a.timestamp).getTime();
    const timeB = new Date(b.timestamp).getTime();
    return timeB - timeA; // Descending order
  });
}
