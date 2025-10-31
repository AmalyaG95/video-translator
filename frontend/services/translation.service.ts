
import { apiClient } from "@/lib/api-client";
import { API_ENDPOINTS } from "@/constants";
import type {
  ProcessingSession,
  TranslationRequest,
  ControlTranslationRequest,
  ControlTranslationResponse,
  ApiResponse,
} from "@/shared/types";

export class TranslationService {
  async startTranslation(
    request: TranslationRequest
  ): Promise<ProcessingSession> {
    const formData = new FormData();
    formData.append("file", request.file);
    formData.append("sourceLang", request.sourceLang);
    formData.append("targetLang", request.targetLang);

    if (request.qualityPreset) {
      formData.append("qualityPreset", request.qualityPreset);
    }
    if (request.preserveTiming !== undefined) {
      formData.append("preserveTiming", request.preserveTiming.toString());
    }
    if (request.lipSyncEnabled !== undefined) {
      formData.append("lipSyncEnabled", request.lipSyncEnabled.toString());
    }

    const response = await apiClient.post<ProcessingSession>(
      API_ENDPOINTS.UPLOAD,
      formData
    );

    return response;
  }

  async pauseTranslation(
    sessionId: string
  ): Promise<ControlTranslationResponse> {
    const response = await apiClient.post<ControlTranslationResponse>(
      API_ENDPOINTS.TRANSLATE_CONTROL(sessionId),
      { action: "pause" } as ControlTranslationRequest
    );

    return response;
  }

  async resumeTranslation(
    sessionId: string
  ): Promise<ControlTranslationResponse> {
    const response = await apiClient.post<ControlTranslationResponse>(
      API_ENDPOINTS.TRANSLATE_CONTROL(sessionId),
      { action: "resume" } as ControlTranslationRequest
    );

    return response;
  }

  async cancelTranslation(
    sessionId: string
  ): Promise<ControlTranslationResponse> {
    const response = await apiClient.post<ControlTranslationResponse>(
      API_ENDPOINTS.TRANSLATE_CONTROL(sessionId),
      { action: "cancel" } as ControlTranslationRequest
    );

    return response;
  }

  async getAvailableSegments(sessionId: string): Promise<string[]> {
    const response = await apiClient.get<{ segments: string[] }>(
      API_ENDPOINTS.TRANSLATE_SEGMENTS(sessionId)
    );

    return response.segments;
  }

  async downloadSegment(sessionId: string, segmentId: string): Promise<Blob> {
    const url = `${API_ENDPOINTS.TRANSLATE_SEGMENTS(sessionId)}?segmentId=${segmentId}&download=true`;
    const response = await apiClient.get<Blob>(url);
    return response;
  }

  getSegmentUrl(sessionId: string, segmentId: string): string {
    return `${API_ENDPOINTS.TRANSLATE_SEGMENTS(sessionId)}?segmentId=${segmentId}&download=true`;
  }

  async getTranslationStatus(sessionId: string): Promise<ProcessingSession> {
    const response = await apiClient.get<ProcessingSession>(
      API_ENDPOINTS.TRANSLATE(sessionId)
    );

    return response;
  }
}

// Create singleton instance
export const translationService = new TranslationService();
