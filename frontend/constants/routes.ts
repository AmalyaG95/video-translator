
export const ROUTES = {
  HOME: "/",
  PROCESSING: "/processing",
  RESULTS: "/results",
  HISTORY: "/history",
} as const;

export type Route = (typeof ROUTES)[keyof typeof ROUTES];

export const buildResultsRoute = (sessionId: string): string => {
  return `${ROUTES.RESULTS}/${sessionId}`;
};

export const buildProcessingRoute = (sessionId: string): string => {
  return `${ROUTES.PROCESSING}?session=${sessionId}`;
};

export const isResultsRoute = (pathname: string): boolean => {
  return pathname.startsWith(ROUTES.RESULTS);
};

export const isProcessingRoute = (pathname: string): boolean => {
  return pathname.startsWith(ROUTES.PROCESSING);
};

export const extractSessionIdFromResults = (
  pathname: string
): string | undefined => {
  const match = pathname.match(new RegExp(`^${ROUTES.RESULTS}/(.+)$`));
  return match?.[1];
};

export const extractSessionIdFromProcessing = (
  searchParams: string
): string | undefined => {
  const params = new URLSearchParams(searchParams);
  return params.get("session") ?? undefined;
};
