import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Toaster } from "react-hot-toast";
import { Providers } from "./providers";
import { LayoutContent } from "./LayoutContent";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Video Translator",
  description:
    "AI-powered video translation with perfect lip-sync and natural voice quality",
};

function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <LayoutContent>{children}</LayoutContent>
          <Toaster
            position="bottom-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: "#fff",
                color: "#363636",
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: "#10b981",
                  secondary: "#fff",
                },
              },
              error: {
                duration: 4000,
                iconTheme: {
                  primary: "#ef4444",
                  secondary: "#fff",
                },
              },
            }}
          />
        </Providers>
      </body>
    </html>
  );
}

export default RootLayout;

/*
{
  "sessionId": "eba708e4-320d-47c4-9fd7-de14b6d96345",
  "status": "processing",
  "progress": 51.904761904761905,
  "currentStep": "Processing segments... (2/21)",
  "sourceLang": "en",
  "targetLang": "hy",
  "filePath": "/home/amalya/Desktop/translate-v/backend-nestjs/uploads/1761513241034-508784865.mp4",
  "fileName": "English - Daily routine (A1-A2).mp4",
  "fileSize": 4549683,
  "createdAt": "2025-10-26T21:14:01.394Z",
  "updatedAt": "2025-10-26T21:14:01.394Z",
  "totalChunks": 21,
  "currentChunk": 2,
  "etaSeconds": 201,
  "processingSpeed": 14,
  "hardwareInfo": {
    "cpu": "Intel Core i7-12700K",
    "gpu": "NVIDIA RTX 4080",
    "vram_gb": 16,
    "ram_gb": 32
  },
  "availableSegments": [
    "seg_001",
    "seg_002"
  ],
  "early_preview_available": false,
  "early_preview_path": "",
  "isPaused": false
}
 */
