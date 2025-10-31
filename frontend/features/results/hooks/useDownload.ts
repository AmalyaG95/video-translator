"use client";

import { useCallback } from "react";
import toast from "react-hot-toast";
import { resultsService } from "../services/resultsService";

export function useDownload() {
  const handleDownload = useCallback(async (sessionId: string) => {
    try {
      // Download the blob
      const blob = await resultsService.downloadVideo(sessionId);
      const url = window.URL.createObjectURL(blob);

      // Create a popup window that will appear above the app
      const downloadWindow = window.open(
        "",
        "download",
        "width=1,height=1,left=0,top=0,menubar=no,toolbar=no,location=no,status=no"
      );

      if (downloadWindow) {
        downloadWindow.document.write(`
          <html>
            <head><title>Downloading...</title></head>
            <body>
              <script>
                const a = document.createElement('a');
                a.href = '${url}';
                a.download = 'translated_${sessionId}.mp4';
                document.body.appendChild(a);
                a.click();
                setTimeout(() => {
                  window.close();
                }, 100);
              </script>
            </body>
          </html>
        `);
        downloadWindow.document.close();
      } else {
        // Fallback to direct download if popup blocked
        const a = document.createElement("a");
        a.href = url;
        a.download = `translated_${sessionId}.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }

      // Clean up after download starts
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
      }, 1000);

      toast.success("Download started!");
    } catch (error) {
      console.error("Download error:", error);
      toast.error("Failed to download result");
    }
  }, []);

  return { handleDownload };
}
