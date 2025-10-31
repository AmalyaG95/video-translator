const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

export const API_ENDPOINTS = {
  // Upload endpoints
  UPLOAD: `${API_BASE}/upload`,

  // Translation endpoints
  TRANSLATE: (sessionId: string) => `${API_BASE}/translate/${sessionId}`,
  TRANSLATE_CONTROL: (sessionId: string) =>
    `${API_BASE}/translate/${sessionId}/control`,
  TRANSLATE_SEGMENTS: (sessionId: string) =>
    `${API_BASE}/translate/${sessionId}/segments`,

  // Session endpoints
  SESSION: (sessionId: string) => `${API_BASE}/sessions/${sessionId}`,
  SESSIONS: `${API_BASE}/sessions`,
  SESSION_PROGRESS: (sessionId: string) =>
    `${API_BASE}/sessions/${sessionId}/progress`,
  SESSION_LOGS: (sessionId: string) => `${API_BASE}/sessions/${sessionId}/logs`,

  // Download endpoints
  DOWNLOAD: (sessionId: string) => `${API_BASE}/download/${sessionId}`,

  // Preview and streaming endpoints
  EARLY_PREVIEW: (sessionId: string) =>
    `${API_BASE}/early-preview/${sessionId}`,
  STREAM: (sessionId: string) => `${API_BASE}/stream/${sessionId}`,

  // Language detection endpoints
  LANGUAGE_DETECTION_SESSION: (sessionId: string) =>
    `${API_BASE}/${sessionId}/detect-language`,

  // Health check
  HEALTH: `${API_BASE}/health`,
} as const;

export const buildEndpointWithParams = (
  baseEndpoint: string,
  params: Record<string, string | number | boolean>
): string => {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value != null) {
      searchParams.append(key, String(value));
    }
  });

  const queryString = searchParams.toString();
  return queryString ? `${baseEndpoint}?${queryString}` : baseEndpoint;
};

export const buildSegmentUrl = (
  sessionId: string,
  segmentId: string,
  download: boolean = false
): string => {
  return buildEndpointWithParams(API_ENDPOINTS.TRANSLATE_SEGMENTS(sessionId), {
    segmentId,
    download: download.toString(),
  });
};

export const buildPreviewUrl = (sessionId: string): string => {
  return `/api/preview/${sessionId}`;
};

export const buildVideoProxyUrl = (
  sessionId: string,
  segmentId: string
): string => {
  return `${API_BASE}/translate/${sessionId}/segments?segmentId=${segmentId}&download=true`;
};
