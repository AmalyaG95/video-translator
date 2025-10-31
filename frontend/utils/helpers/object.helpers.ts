
export const deepClone = <T>(obj: T): T => {
  if (typeof structuredClone !== "undefined") {
    return structuredClone(obj);
  }

  // Fallback for older environments
  return JSON.parse(JSON.stringify(obj));
};

export const pick = <T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> => {
  const result = {} as Pick<T, K>;
  for (const key of keys) {
    if (key in obj) {
      result[key] = obj[key];
    }
  }
  return result;
};

export const omit = <T, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> => {
  const result = { ...obj };
  for (const key of keys) {
    delete result[key];
  }
  return result;
};

export const hasAllKeys = <T extends object>(
  obj: T,
  keys: (keyof T)[]
): boolean => {
  return keys.every(key => key in obj);
};

export const hasAnyKey = <T extends object>(
  obj: T,
  keys: (keyof T)[]
): boolean => {
  return keys.some(key => key in obj);
};

export const getNested = <T>(obj: T, path: string): unknown => {
  return path.split(".").reduce((current, key) => {
    return current?.[key as keyof typeof current];
  }, obj as any);
};

export const setNested = <T>(obj: T, path: string, value: unknown): T => {
  const keys = path.split(".");
  const result = deepClone(obj);
  let current = result as any;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (key === undefined) continue;
    if (!(key in current) || typeof current[key] !== "object") {
      current[key] = {};
    }
    current = current[key];
  }

  const lastKey = keys[keys.length - 1];
  if (lastKey !== undefined) {
    current[lastKey] = value;
  }
  return result;
};

export const deepMerge = <T>(target: T, ...sources: Partial<T>[]): T => {
  if (!sources.length) return target;
  const source = sources.shift();

  if (isObject(target) && isObject(source)) {
    for (const key in source) {
      if (isObject(source[key])) {
        if (!(key in target)) {
          Object.assign(target, { [key]: {} });
        }
        deepMerge(target[key] as any, source[key] as any);
      } else {
        Object.assign(target, { [key]: source[key] });
      }
    }
  }

  return deepMerge(target, ...sources);
};

export const isObject = (value: unknown): value is Record<string, unknown> => {
  return value !== null && typeof value === "object" && !Array.isArray(value);
};

export const fromPairs = <K extends string | number | symbol, V>(
  pairs: [K, V][]
): Record<K, V> => {
  return pairs.reduce(
    (obj, [key, value]) => {
      obj[key] = value;
      return obj;
    },
    {} as Record<K, V>
  );
};

export const toPairs = <T extends Record<string, unknown>>(
  obj: T
): [keyof T, T[keyof T]][] => {
  return Object.entries(obj) as [keyof T, T[keyof T]][];
};
