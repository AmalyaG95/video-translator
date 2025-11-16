
const ALLOWED_VIDEO_TYPES = [
  "video/mp4",
  "video/webm",
  "video/quicktime",
  "video/avi",
  "video/mov",
] as const;

const MAX_FILE_SIZE = 100 * 1_024 * 1_024 * 1_024; // 100GB - supports 15+ hour videos
const MIN_FILE_SIZE = 1024; // 1KB

export type VideoValidationResult = {
  valid: boolean;
  error?: string;
};

export const validateVideoFile = (file: File): VideoValidationResult => {
  if (!file) {
    return { valid: false, error: "No file provided" };
  }

  if (!ALLOWED_VIDEO_TYPES.includes(file.type as any)) {
    return {
      valid: false,
      error: `Invalid file type. Supported types: ${ALLOWED_VIDEO_TYPES.join(", ")}`,
    };
  }

  if (file.size < MIN_FILE_SIZE) {
    return { valid: false, error: "File too small" };
  }

  if (file.size > MAX_FILE_SIZE) {
    return {
      valid: false,
      error: `File too large. Maximum size: ${Math.round(MAX_FILE_SIZE / (1024 * 1024))}MB`,
    };
  }

  return { valid: true };
};

export const validateVideoFileWithOptions = (
  file: File,
  options: {
    maxSize?: number;
    allowedTypes?: readonly string[];
    minSize?: number;
  } = {}
): VideoValidationResult => {
  if (!file) {
    return { valid: false, error: "No file provided" };
  }

  const {
    maxSize = MAX_FILE_SIZE,
    allowedTypes = ALLOWED_VIDEO_TYPES,
    minSize = MIN_FILE_SIZE,
  } = options;

  if (!allowedTypes.includes(file.type)) {
    return {
      valid: false,
      error: `Invalid file type. Supported types: ${allowedTypes.join(", ")}`,
    };
  }

  if (file.size < minSize) {
    return { valid: false, error: "File too small" };
  }

  if (file.size > maxSize) {
    return {
      valid: false,
      error: `File too large. Maximum size: ${Math.round(maxSize / (1024 * 1024))}MB`,
    };
  }

  return { valid: true };
};

export const getFileTypeFromMime = (mimeType: string): string => {
  const typeMap: Record<string, string> = {
    "video/mp4": "MP4",
    "video/webm": "WebM",
    "video/quicktime": "MOV",
    "video/avi": "AVI",
    "video/mov": "MOV",
  };

  return typeMap[mimeType] ?? "Unknown";
};

export const isVideoFile = (file: File): boolean => {
  return file.type.startsWith("video/");
};

