export function calculateChunkInfo(duration: number) {
  const chunkSize = 30; // seconds
  const chunkCount = Math.ceil(duration / chunkSize);
  return {
    chunkCount,
    chunkSize,
    message: `AI will split into ~${chunkCount} chunks to ensure perfect ${formatDuration(duration)} sync`,
  };
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3_600) {
    const minutes = Math.round(seconds / 60);
    return `${minutes}m`;
  } else {
    const hours = Math.floor(seconds / 3_600);
    const minutes = Math.round((seconds % 3_600) / 60);
    return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  }
}
