// Upload service for API calls
import { apiClient } from "@/lib/api-client";
import { API_ENDPOINTS } from "@/constants/api-endpoints";
import type { UploadResponse, VideoMetadata } from "../types";

export class UploadService {
  static async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    return apiClient.post<UploadResponse>(API_ENDPOINTS.UPLOAD, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  static async getVideoMetadata(file: File): Promise<VideoMetadata> {
    const formData = new FormData();
    formData.append("file", file);

    return apiClient.post<VideoMetadata>(API_ENDPOINTS.UPLOAD, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }
}
