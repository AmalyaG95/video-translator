
import { apiClient } from "@/lib/api-client";
import { API_ENDPOINTS } from "@/constants";
import { validateVideoFile } from "@/utils/validation/video.validation";
import type { ApiResponse, UploadResponse } from "@/shared/types";

export type UploadProgress = {
  loaded: number;
  total: number;
  percentage: number;
};

export type UploadOptions = {
  onProgress?: (progress: UploadProgress) => void;
  onSuccess?: (response: UploadResponse) => void;
  onError?: (error: Error) => void;
};

export class UploadService {
  async uploadVideo(
    file: File,
    options: UploadOptions = {}
  ): Promise<UploadResponse> {
    // Validate file before upload
    const validation = validateVideoFile(file);
    if (!validation.valid) {
      const error = new Error(validation.error);
      options.onError?.(error);
      throw error;
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Create custom API client for upload with progress tracking
      const uploadClient = apiClient;

      // Override the request method to track progress
      const response = await this.uploadWithProgress(
        API_ENDPOINTS.UPLOAD,
        formData,
        options.onProgress
      );

      const uploadResponse: UploadResponse = {
        sessionId: response.sessionId,
        fileName: response.fileName,
        fileSize: response.fileSize,
        message: response.message,
      };

      options.onSuccess?.(uploadResponse);
      return uploadResponse;
    } catch (error) {
      const uploadError =
        error instanceof Error ? error : new Error("Upload failed");
      options.onError?.(uploadError);
      throw uploadError;
    }
  }

  private async uploadWithProgress(
    endpoint: string,
    formData: FormData,
    onProgress?: (progress: UploadProgress) => void
  ): Promise<UploadResponse> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      // Track upload progress
      xhr.upload.addEventListener("progress", event => {
        if (event.lengthComputable && onProgress) {
          const progress: UploadProgress = {
            loaded: event.loaded,
            total: event.total,
            percentage: Math.round((event.loaded / event.total) * 100),
          };
          onProgress(progress);
        }
      });

      // Handle successful upload
      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            resolve(response);
          } catch (error) {
            reject(new Error("Invalid response format"));
          }
        } else {
          reject(new Error(`Upload failed with status ${xhr.status}`));
        }
      });

      // Handle upload error
      xhr.addEventListener("error", () => {
        reject(new Error("Upload failed due to network error"));
      });

      // Handle upload timeout
      xhr.addEventListener("timeout", () => {
        reject(new Error("Upload timed out"));
      });

      // Configure request
      xhr.open("POST", endpoint);
      xhr.timeout = 300_000; // 5 minutes timeout for uploads

      // Send request
      xhr.send(formData);
    });
  }

  validateFile(file: File): { valid: boolean; error?: string } {
    return validateVideoFile(file);
  }

  getFileSize(file: File): string {
    const units = ["B", "KB", "MB", "GB"];
    let size = file.size;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  getFileType(file: File): string {
    const typeMap: Record<string, string> = {
      "video/mp4": "MP4",
      "video/webm": "WebM",
      "video/quicktime": "MOV",
      "video/avi": "AVI",
      "video/mov": "MOV",
    };

    return typeMap[file.type] ?? "Unknown";
  }

  createPreviewUrl(file: File): string {
    return URL.createObjectURL(file);
  }

  revokePreviewUrl(url: string): void {
    URL.revokeObjectURL(url);
  }
};

// Create singleton instance
export const uploadService = new UploadService();

