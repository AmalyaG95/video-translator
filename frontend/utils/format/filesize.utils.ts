
const FILE_SIZE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;
const FILE_SIZE_THRESHOLD = 1_024; // 1024 bytes = 1 KB

export const formatFileSize = (bytes: number): string => {
  let size = bytes;
  let unitIndex = 0;

  while (
    size >= FILE_SIZE_THRESHOLD &&
    unitIndex < FILE_SIZE_UNITS.length - 1
  ) {
    size /= FILE_SIZE_THRESHOLD;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${FILE_SIZE_UNITS[unitIndex]}`;
};

export const formatFileSizePrecise = (
  bytes: number,
  precision: number = 1
): string => {
  let size = bytes;
  let unitIndex = 0;

  while (
    size >= FILE_SIZE_THRESHOLD &&
    unitIndex < FILE_SIZE_UNITS.length - 1
  ) {
    size /= FILE_SIZE_THRESHOLD;
    unitIndex++;
  }

  return `${size.toFixed(precision)} ${FILE_SIZE_UNITS[unitIndex]}`;
};

export const calculateCompressionRatio = (
  originalSize: number,
  compressedSize: number
): number => {
  if (originalSize === 0) return 0;
  return Math.round(((originalSize - compressedSize) / originalSize) * 100);
};

export const formatCompressionRatio = (
  originalSize: number,
  compressedSize: number
): string => {
  const ratio = calculateCompressionRatio(originalSize, compressedSize);
  return ratio > 0 ? `${ratio}% smaller` : "No compression";
};

