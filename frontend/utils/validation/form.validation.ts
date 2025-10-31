
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const isValidUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

export const isRequired = (value: unknown): boolean => {
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  return value != null && value !== "";
};

export const isValidLength = (
  value: string,
  min: number,
  max: number
): boolean => {
  return value.length >= min && value.length <= max;
};

export const isValidLanguageCode = (code: string): boolean => {
  const languageCodeRegex = /^[a-z]{2}$/;
  return languageCodeRegex.test(code);
};

export const isValidSessionId = (sessionId: string): boolean => {
  const uuidRegex =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  return uuidRegex.test(sessionId);
};

export const isValidQualityPreset = (preset: string): boolean => {
  const validPresets = ["low", "medium", "high", "ultra"];
  return validPresets.includes(preset);
};

export const createValidationError = (
  field: string,
  message: string
): string => {
  const fieldName = field.charAt(0).toUpperCase() + field.slice(1);
  return `${fieldName}: ${message}`;
};

