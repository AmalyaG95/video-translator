// Upload feature constants
export const UPLOAD_CONSTANTS = {
  MAX_FILE_SIZE: 50 * 1_024 * 1_024, // 50MB
  CHUNK_SIZE: 30, // 30 seconds per chunk
  SUPPORTED_VIDEO_TYPES: [
    "video/mp4",
    "video/avi",
    "video/mov",
    "video/wmv",
    "video/flv",
    "video/webm",
    "video/mkv",
  ],
  SUPPORTED_AUDIO_TYPES: [
    "audio/mp3",
    "audio/wav",
    "audio/m4a",
    "audio/aac",
    "audio/ogg",
  ],
} as const;

export const UPLOAD_MESSAGES = {
  FILE_TOO_LARGE: "File size must be less than 50MB",
  INVALID_FILE_TYPE: "Please upload a valid video or audio file",
  UPLOAD_ERROR: "Failed to upload file. Please try again.",
  DETECTING_LANGUAGE: "AI is detecting the language...",
  LANGUAGE_DETECTED: "Language detected successfully",
  UPLOAD_SUCCESS: "File uploaded successfully",
} as const;
