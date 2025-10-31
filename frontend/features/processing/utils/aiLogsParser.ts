export function parseAIReasoningLogs(logs: any[]) {
  return logs
    .filter(log => log.message && log.message.includes("AI"))
    .map((log, index) => {
      return {
        id: `ai-log-${index}`,
        timestamp: log.timestamp,
        stage: log.stage || "unknown",
        type: "info" as const,
        message: log.message || "No message",
        data: log.data || null,
      };
    });
}
