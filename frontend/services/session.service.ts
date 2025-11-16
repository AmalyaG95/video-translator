
import { apiClient } from "@/lib/api-client";
import { API_ENDPOINTS } from "@/constants";
import type { ProcessingSession, ApiResponse } from "@/shared/types";

export class SessionService {
  async getSession(sessionId: string): Promise<ProcessingSession> {
    const response = await apiClient.get<ProcessingSession>(
      API_ENDPOINTS.SESSION(sessionId)
    );

    return response;
  }

  async getSessions(): Promise<ProcessingSession[]> {
    const response = await apiClient.get<{ sessions: ProcessingSession[] } | ProcessingSession[]>(
      API_ENDPOINTS.SESSIONS
    );

    // Backend returns { sessions: [...] }, extract the array
    if (Array.isArray(response)) {
      return response;
    }
    return (response as { sessions: ProcessingSession[] }).sessions || [];
  }

  async getSessionProgress(sessionId: string): Promise<ProcessingSession> {
    const response = await apiClient.get<ProcessingSession>(
      API_ENDPOINTS.SESSION_PROGRESS(sessionId)
    );

    return response;
  }

  async getSessionLogs(sessionId: string): Promise<unknown[]> {
    const response = await apiClient.get<unknown[]>(
      API_ENDPOINTS.SESSION_LOGS(sessionId)
    );

    return response;
  }

  async deleteSession(sessionId: string): Promise<{ success: boolean }> {
    const response = await apiClient.delete<{ success: boolean }>(
      API_ENDPOINTS.SESSION(sessionId)
    );

    return response;
  }

  async getRecentSessions(): Promise<ProcessingSession[]> {
    const sessions = await this.getSessions();

    // Sort by creation date (most recent first) and take first 10
    return sessions
      .sort((a, b) => {
        const dateA = new Date(a.createdAt ?? 0);
        const dateB = new Date(b.createdAt ?? 0);
        return dateB.getTime() - dateA.getTime();
      })
      .slice(0, 10);
  }

  async getSessionsByStatus(
    status: ProcessingSession["status"]
  ): Promise<ProcessingSession[]> {
    const sessions = await this.getSessions();
    return sessions.filter(session => session.status === status);
  }

  async getCompletedSessions(): Promise<ProcessingSession[]> {
    return this.getSessionsByStatus("completed");
  }

  async getProcessingSessions(): Promise<ProcessingSession[]> {
    return this.getSessionsByStatus("processing");
  }

  async getFailedSessions(): Promise<ProcessingSession[]> {
    return this.getSessionsByStatus("failed");
  }
}

// Create singleton instance
export const sessionService = new SessionService();
