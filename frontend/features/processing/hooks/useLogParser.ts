import { useRef, useEffect, useCallback } from "react";

export function useLogParser() {
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      workerRef.current = new Worker(
        new URL("../../public/workers/logParser.worker.ts", import.meta.url)
      );

      return () => {
        workerRef.current?.terminate();
        workerRef.current = null;
      };
    }
  }, []);

  const parseLogsAsync = useCallback((logs: any[]): Promise<any[]> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) {
        resolve(logs);
        return;
      }

      const handleMessage = (e: MessageEvent) => {
        if (e.data.type === "LOGS_PARSED") {
          workerRef.current!.removeEventListener("message", handleMessage);
          resolve(e.data.data);
        } else if (e.data.type === "ERROR") {
          workerRef.current!.removeEventListener("message", handleMessage);
          reject(new Error(e.data.error));
        }
      };

      workerRef.current.addEventListener("message", handleMessage);
      workerRef.current.postMessage({ type: "PARSE_LOGS", data: logs });
    });
  }, []);

  const parseAIReasoningAsync = useCallback((logs: any[]): Promise<any[]> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) {
        resolve(logs);
        return;
      }

      const handleMessage = (e: MessageEvent) => {
        if (e.data.type === "AI_PARSED") {
          workerRef.current!.removeEventListener("message", handleMessage);
          resolve(e.data.data);
        } else if (e.data.type === "ERROR") {
          workerRef.current!.removeEventListener("message", handleMessage);
          reject(new Error(e.data.error));
        }
      };

      workerRef.current.addEventListener("message", handleMessage);
      workerRef.current.postMessage({ type: "PARSE_AI_REASONING", data: logs });
    });
  }, []);

  return { parseLogsAsync, parseAIReasoningAsync };
}










