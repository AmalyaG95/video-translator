
export const unique = <T>(array: T[]): T[] => {
  return [...new Set(array)];
};

export const groupBy = <T, K extends string | number | symbol>(
  array: T[],
  keyFn: (item: T) => K
): Record<K, T[]> => {
  return array.reduce(
    (groups, item) => {
      const key = keyFn(item);
      groups[key] = groups[key] ?? [];
      groups[key].push(item);
      return groups;
    },
    {} as Record<K, T[]>
  );
};

export const chunk = <T>(array: T[], size: number): T[][] => {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
};

export const flatten = <T>(array: T[][]): T[] => {
  return array.flat();
};

export const sample = <T>(array: T[]): T | undefined => {
  if (array.length === 0) return undefined;
  const randomIndex = Math.floor(Math.random() * array.length);
  return array[randomIndex];
};

export const sampleMultiple = <T>(array: T[], count: number): T[] => {
  const shuffled = [...array].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
};

export const partition = <T>(
  array: T[],
  predicate: (item: T) => boolean
): [T[], T[]] => {
  return array.reduce(
    (acc, item) => {
      acc[predicate(item) ? 0 : 1].push(item);
      return acc;
    },
    [[], []] as [T[], T[]]
  );
};

export const sortBy = <T, K>(
  array: T[],
  keyFn: (item: T) => K,
  direction: "asc" | "desc" = "asc"
): T[] => {
  return [...array].sort((a, b) => {
    const aKey = keyFn(a);
    const bKey = keyFn(b);

    if (aKey < bKey) return direction === "asc" ? -1 : 1;
    if (aKey > bKey) return direction === "asc" ? 1 : -1;
    return 0;
  });
};

export const findLast = <T>(
  array: T[],
  predicate: (item: T) => boolean
): T | undefined => {
  for (let i = array.length - 1; i >= 0; i--) {
    if (array[i] !== undefined && predicate(array[i]!)) {
      return array[i];
    }
  }
  return undefined;
};
