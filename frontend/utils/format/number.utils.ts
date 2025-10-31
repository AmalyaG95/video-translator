
export const formatNumber = (num: number): string => {
  return new Intl.NumberFormat("en-US").format(num);
};

export const formatPercentage = (
  value: number,
  decimals: number = 1
): string => {
  return `${(value * 100).toFixed(decimals)}%`;
};

export const formatSmartNumber = (num: number): string => {
  if (num === 0) return "0";

  const absNum = Math.abs(num);

  if (absNum >= 1_000) {
    return formatNumber(Math.round(num * 100) / 100);
  }

  if (absNum >= 0.01) {
    return num.toFixed(2);
  }

  return num.toFixed(3);
};

export const formatCompactNumber = (num: number): string => {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(num);
};

export const clamp = (num: number, min: number, max: number): number => {
  return Math.min(Math.max(num, min), max);
};

export const roundTo = (num: number, decimals: number = 0): number => {
  const factor = 10 ** decimals;
  return Math.round(num * factor) / factor;
};

