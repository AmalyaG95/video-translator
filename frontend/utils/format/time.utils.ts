
export const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
};

export const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3_600);
  const mins = Math.floor((seconds % 3_600) / 60);
  const secs = Math.floor(seconds % 60);

  return hours > 0
    ? `${hours}h ${mins}m ${secs}s`
    : mins > 0
      ? `${mins}m ${secs}s`
      : `${secs}s`;
};

export const formatETA = (etaSeconds: number): string => {
  if (etaSeconds < 60) {
    return `${Math.round(etaSeconds)}s`;
  }

  if (etaSeconds < 3_600) {
    const minutes = Math.round(etaSeconds / 60);
    return `${minutes}m`;
  }

  const hours = Math.floor(etaSeconds / 3_600);
  const minutes = Math.round((etaSeconds % 3_600) / 60);
  return `${hours}h ${minutes}m`;
};

export const formatProcessingSpeed = (speed: number): string => {
  return `${speed.toFixed(1)} chunks/min`;
};

