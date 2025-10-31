import { apiClient } from "@/lib/api-client";
import { API_ENDPOINTS } from "@/constants";

export const resultsService = {
  async downloadVideo(sessionId: string): Promise<Blob> {
    try {
      const response = await fetch(API_ENDPOINTS.DOWNLOAD(sessionId));
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }
      const blob = await response.blob();
      return blob;
    } catch (error) {
      console.error("Download error:", error);
      throw error;
    }
  },

  async getVideoUrl(sessionId: string): Promise<string | null> {
    try {
      const blob = await this.downloadVideo(sessionId);
      const url = window.URL.createObjectURL(blob);
      return url;
    } catch (error) {
      console.error("Failed to get video URL:", error);
      return null;
    }
  },

  async refineVoice(
    sessionId: string
  ): Promise<{ success: boolean; message: string }> {
    // TODO: Implement voice refinement API call
    return {
      success: true,
      message: "Voice refinement started!",
    };
  },

  async getSessionResult(sessionId: string): Promise<any> {
    const result = await apiClient.get(API_ENDPOINTS.SESSION(sessionId));
    return result;
  },
};
