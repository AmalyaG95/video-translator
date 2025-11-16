const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3001";

interface ApiResponse<T> {
  data: T;
  status: number;
  statusText: string;
}

class ApiClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // Handle both full URLs and relative paths
    const url = endpoint.startsWith("http")
      ? endpoint
      : `${this.baseURL}${endpoint}`;

    console.log("🔵🔵🔵 [API CLIENT] Making request");
    console.log("   Method:", options.method || "GET");
    console.log("   URL:", url);
    console.log("   Headers:", options.headers);
    console.log("   Body:", options.body ? (typeof options.body === 'string' ? options.body.substring(0, 200) : '[FormData or Blob]') : '[empty]');
    
    const startTime = Date.now();
    
    try {
      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        ...options,
      });

      const duration = Date.now() - startTime;
      console.log("🔵 [API CLIENT] Response received");
      console.log("   Status:", response.status, response.statusText);
      console.log("   OK:", response.ok);
      console.log("   Duration:", duration, "ms");
      console.log("   Content-Type:", response.headers.get("content-type"));

      if (!response.ok) {
        const errorText = await response.text().catch(() => "[Could not read error body]");
        console.error("❌ [API CLIENT] HTTP error!");
        console.error("   Status:", response.status);
        console.error("   Error body:", errorText.substring(0, 500));
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle different response types
      const contentType = response.headers.get("content-type");
      let result: T;
      
      if (contentType && contentType.includes("application/json")) {
        result = await response.json();
        console.log("🔵 [API CLIENT] Parsed JSON response:", result);
      } else if (
        contentType &&
        contentType.includes("application/octet-stream")
      ) {
        result = (await response.blob()) as T;
        console.log("🔵 [API CLIENT] Received blob response");
      } else {
        result = (await response.text()) as T;
        console.log("🔵 [API CLIENT] Text response:", String(result).substring(0, 200));
      }
      
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      console.error("❌❌❌ [API CLIENT] Request failed");
      console.error("   URL:", url);
      console.error("   Duration before error:", duration, "ms");
      console.error("   Error:", error);
      console.error("   Error type:", error instanceof Error ? error.constructor.name : typeof error);
      if (error instanceof Error && error.stack) {
        console.error("   Stack trace:", error.stack);
      }
      throw error;
    }
  }

  async get<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: "GET" });
  }

  async post<T>(
    endpoint: string,
    data?: any,
    options: RequestInit = {}
  ): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: "POST",
      body:
        data instanceof FormData
          ? data
          : data
            ? JSON.stringify(data)
            : undefined,
    });
  }

  async patch<T>(
    endpoint: string,
    data?: any,
    options: RequestInit = {}
  ): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: "PATCH",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: "DELETE" });
  }

  async uploadFile<T>(
    endpoint: string,
    file: File,
    additionalData?: Record<string, any>
  ): Promise<T> {
    const formData = new FormData();
    formData.append("file", file);

    if (additionalData) {
      console.log("[API Client] Adding to FormData:", additionalData);
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, value);
      });
    }

    // Handle both full URLs and relative paths
    const url = endpoint.startsWith("http")
      ? endpoint
      : `${this.baseURL}${endpoint}`;
    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
}

export const apiClient = new ApiClient();
