export const capitalize = (str: string): string => {
  if (str.length === 0) return str;
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

export const toCamelCase = (str: string): string => {
  return str
    .replace(/[-_\s]+(.)?/g, (_, char) => char?.toUpperCase() ?? "")
    .replace(/^[A-Z]/, char => char.toLowerCase());
};

export const toKebabCase = (str: string): string => {
  return str
    .replace(/([a-z])([A-Z])/g, "$1-$2")
    .replace(/[\s_]+/g, "-")
    .toLowerCase();
};

export const toPascalCase = (str: string): string => {
  return str
    .replace(/[-_\s]+(.)?/g, (_, char) => char?.toUpperCase() ?? "")
    .replace(/^[a-z]/, char => char.toUpperCase());
};

export const truncate = (
  str: string,
  length: number,
  ellipsis: string = "..."
): string => {
  if (str.length <= length) return str;
  return str.slice(0, length - ellipsis.length) + ellipsis;
};

export const stripHtml = (str: string): string => {
  return str.replace(/<[^>]*>/g, "");
};

export const escapeHtml = (str: string): string => {
  const htmlEscapes: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  return str.replace(/[&<>"']/g, char => htmlEscapes[char] ?? char);
};

export const randomString = (
  length: number,
  charset: string = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
): string => {
  let result = "";
  for (let i = 0; i < length; i++) {
    result += charset.charAt(Math.floor(Math.random() * charset.length));
  }
  return result;
};

export const isEmpty = (str: string): boolean => {
  return str.trim().length === 0;
};

export const pad = (
  str: string,
  length: number,
  padString: string = " ",
  position: "start" | "end" = "end"
): string => {
  if (str.length >= length) return str;

  const padLength = length - str.length;
  const padding = padString
    .repeat(Math.ceil(padLength / padString.length))
    .slice(0, padLength);

  return position === "start" ? padding + str : str + padding;
};

export const removeDuplicates = (str: string): string => {
  return [...new Set(str)].join("");
};

export const countOccurrences = (str: string, substring: string): number => {
  return (str.match(new RegExp(substring, "g")) ?? []).length;
};
