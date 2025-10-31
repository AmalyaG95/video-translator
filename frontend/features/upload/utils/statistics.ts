import { useMemo } from "react";

export function useStatistics(sessions: any[]) {
  const completedSessions = sessions.filter(
    (s: any) => s.status === "completed"
  );
  const totalSessions = sessions.length;
  const successRate =
    totalSessions > 0
      ? Math.round((completedSessions.length / totalSessions) * 100)
      : 0;

  const averageProcessingTime = useMemo(() => {
    if (completedSessions.length === 0) return "0 min";

    const totalTime = completedSessions.reduce((acc: number, session: any) => {
      if (session.processingTime) {
        return acc + session.processingTime;
      }
      return acc;
    }, 0);

    const avgMinutes = totalTime / completedSessions.length;
    if (avgMinutes < 1) {
      return `${Math.round(avgMinutes * 60)}s`;
    } else if (avgMinutes < 60) {
      return `${Math.round(avgMinutes)}m`;
    } else {
      const hours = Math.floor(avgMinutes / 60);
      const minutes = Math.round(avgMinutes % 60);
      return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
    }
  }, [completedSessions]);

  return {
    completedSessions,
    totalSessions,
    successRate,
    averageProcessingTime,
  };
}
