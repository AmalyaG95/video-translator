import { apiClient } from "@/lib/api-client";
import { API_ENDPOINTS } from "@/constants/api-endpoints";

export class ProcessingService {
  static async validateSession(sessionId: string): Promise<boolean> {
    try {
      const response = await fetch(
        `http://localhost:3001/sessions/${sessionId}`
      );
      // 404 or other errors mean session is invalid
      if (!response.ok) {
        // Don't log 404s - they're expected for missing sessions
        if (response.status !== 404) {
          console.warn(`Session validation failed: ${response.status}`);
        }
        return false;
      }
      return true;
    } catch (error) {
      // Silently handle connection errors (backend not running, etc.)
      // Don't log these as they're common when backend isn't running
      return false;
    }
  }

  static async getSession(sessionId: string): Promise<any> {
    try {
      const response = await fetch(
        `http://localhost:3001/sessions/${sessionId}`
      );
      if (response.ok) {
        return await response.json();
      }
      throw new Error(`Failed to get session: ${response.statusText}`);
    } catch (error) {
      console.error("Get session failed:", error);
      throw error;
    }
  }

  static async startTranslation(
    sessionId: string
  ): Promise<{ success: boolean; message?: string }> {
    const url = `http://localhost:3001/translate/${sessionId}`;
    console.log("🟡🟡🟡 [PROCESSING SERVICE] startTranslation CALLED");
    console.log("   Session ID:", sessionId);
    console.log("   URL:", url);
    
    try {
      const startTime = Date.now();
      console.log("🟡 [PROCESSING SERVICE] Making fetch request...");
      
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const duration = Date.now() - startTime;
      console.log("🟡 [PROCESSING SERVICE] Fetch response received");
      console.log("   Status:", response.status, response.statusText);
      console.log("   OK:", response.ok);
      console.log("   Duration:", duration, "ms");
      console.log("   Headers:", Object.fromEntries(response.headers.entries()));

      if (response.ok) {
        const responseText = await response.text();
        console.log("🟡 [PROCESSING SERVICE] Response body:", responseText);
        console.log("🟡 [PROCESSING SERVICE] Returning success");
        return { success: true };
      } else {
        const errorText = await response.text();
        console.error("❌ [PROCESSING SERVICE] Request failed");
        console.error("   Status:", response.status);
        console.error("   Error body:", errorText);
        return { success: false, message: "Failed to start translation" };
      }
    } catch (error) {
      console.error("❌❌❌ [PROCESSING SERVICE] Fetch exception:", error);
      console.error("   Error type:", error instanceof Error ? error.constructor.name : typeof error);
      console.error("   Error message:", error instanceof Error ? error.message : String(error));
      if (error instanceof Error && error.stack) {
        console.error("   Stack trace:", error.stack);
      }
      return { success: false, message: "Failed to start translation" };
    }
  }

  static async pauseTranslation(
    sessionId: string
  ): Promise<{ success: boolean; message?: string }> {
    try {
      const endpoint = API_ENDPOINTS.TRANSLATE_CONTROL(sessionId);
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "pause" }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return { success: true };
    } catch (error) {
      console.error("Pause translation failed:", error);
      return { success: false, message: "Failed to pause translation" };
    }
  }

  static async resumeTranslation(
    sessionId: string
  ): Promise<{ success: boolean; message?: string }> {
    try {
      const endpoint = API_ENDPOINTS.TRANSLATE_CONTROL(sessionId);
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "resume" }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return { success: true };
    } catch (error) {
      console.error("Resume translation failed:", error);
      return { success: false, message: "Failed to resume translation" };
    }
  }

  static async cancelTranslation(
    sessionId: string
  ): Promise<{ success: boolean; message?: string }> {
    try {
      const endpoint = API_ENDPOINTS.TRANSLATE_CONTROL(sessionId);
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "cancel" }),
      });

      if (!response.ok) {
        // If session not found (404), treat as successful cancellation
        if (response.status === 404) {
          console.log("Session already removed, treating as cancelled");
          return { success: true, message: "Session already cancelled" };
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return { success: true };
    } catch (error) {
      console.error("Cancel translation failed:", error);
      // Return success if session doesn't exist (already cancelled/removed)
      return { success: true, message: "Session no longer exists" };
    }
  }

  static async getEarlyPreview(sessionId: string): Promise<Blob> {
    try {
      const response = await fetch(API_ENDPOINTS.EARLY_PREVIEW(sessionId));
      if (!response.ok) {
        throw new Error(`Preview not available: ${response.statusText}`);
      }
      return await response.blob();
    } catch (error) {
      console.error("Get early preview failed:", error);
      throw error;
    }
  }
}
