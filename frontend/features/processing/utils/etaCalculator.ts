/**
 * Calculate ETA (Estimated Time to Arrival) from real progress data
 */

interface ETAInput {
  progress?: number;
  progress_percent?: number;
  elapsed_time?: number;
  segments_processed?: number;
  total_segments?: number;
  current_time?: number;
  total_duration?: number;
  processingSpeed?: number;
  currentChunk?: number;
  totalChunks?: number;
  status?: string;
}

/**
 * Calculate ETA in seconds based on current progress
 */
export function calculateETA(data: ETAInput): number | null {
  // If completed, no ETA
  if (data.status === "completed" || data.status === "failed") {
    return null;
  }

  // Method 1: Use progress_percent and elapsed_time (most accurate)
  if (
    data.progress_percent !== undefined &&
    data.progress_percent > 0 &&
    data.progress_percent < 100 &&
    data.elapsed_time !== undefined &&
    data.elapsed_time > 0
  ) {
    const remainingPercent = 100 - data.progress_percent;
    const timePerPercent = data.elapsed_time / data.progress_percent;
    const eta = remainingPercent * timePerPercent;
    
    // Only return if ETA is reasonable (not negative, not too large)
    if (eta > 0 && eta < 86400) { // Less than 24 hours
      return Math.ceil(eta);
    }
  }

  // Method 2: Use segments_processed and total_segments
  if (
    data.segments_processed !== undefined &&
    data.segments_processed > 0 &&
    data.total_segments !== undefined &&
    data.total_segments > data.segments_processed &&
    data.elapsed_time !== undefined &&
    data.elapsed_time > 0
  ) {
    const remainingSegments = data.total_segments - data.segments_processed;
    const timePerSegment = data.elapsed_time / data.segments_processed;
    const eta = remainingSegments * timePerSegment;
    
    if (eta > 0 && eta < 86400) {
      return Math.ceil(eta);
    }
  }

  // Method 3: Use currentChunk and totalChunks with processingSpeed
  if (
    data.currentChunk !== undefined &&
    data.currentChunk >= 0 &&
    data.totalChunks !== undefined &&
    data.totalChunks > data.currentChunk &&
    data.processingSpeed !== undefined &&
    data.processingSpeed > 0
  ) {
    const remainingChunks = data.totalChunks - (data.currentChunk + 1);
    const eta = (remainingChunks / data.processingSpeed) * 60; // processingSpeed is chunks/min
    
    if (eta > 0 && eta < 86400) {
      return Math.ceil(eta);
    }
  }

  // Method 4: Use progress (0-100) and elapsed_time
  if (
    data.progress !== undefined &&
    data.progress > 0 &&
    data.progress < 100 &&
    data.elapsed_time !== undefined &&
    data.elapsed_time > 0
  ) {
    const remainingProgress = 100 - data.progress;
    const timePerProgress = data.elapsed_time / data.progress;
    const eta = remainingProgress * timePerProgress;
    
    if (eta > 0 && eta < 86400) {
      return Math.ceil(eta);
    }
  }

  // Method 5: Use current_time and total_duration (for transcription/translation stages)
  if (
    data.current_time !== undefined &&
    data.current_time > 0 &&
    data.total_duration !== undefined &&
    data.total_duration > data.current_time &&
    data.elapsed_time !== undefined &&
    data.elapsed_time > 0
  ) {
    const remainingTime = data.total_duration - data.current_time;
    const timeRatio = data.elapsed_time / data.current_time;
    const eta = remainingTime * timeRatio;
    
    if (eta > 0 && eta < 86400) {
      return Math.ceil(eta);
    }
  }

  return null;
}

/**
 * Calculate processing speed (chunks/min or segments/min)
 */
export function calculateProcessingSpeed(data: ETAInput): number | null {
  if (data.processingSpeed !== undefined && data.processingSpeed > 0) {
    return data.processingSpeed;
  }

  // Calculate from segments_processed and elapsed_time
  if (
    data.segments_processed !== undefined &&
    data.segments_processed > 0 &&
    data.elapsed_time !== undefined &&
    data.elapsed_time > 0
  ) {
    const speed = (data.segments_processed / data.elapsed_time) * 60; // segments per minute
    return speed > 0 ? speed : null;
  }

  // Calculate from currentChunk and elapsed_time
  if (
    data.currentChunk !== undefined &&
    data.currentChunk > 0 &&
    data.elapsed_time !== undefined &&
    data.elapsed_time > 0
  ) {
    const speed = ((data.currentChunk + 1) / data.elapsed_time) * 60; // chunks per minute
    return speed > 0 ? speed : null;
  }

  return null;
}

