"use client";

import React, { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import {
  Upload,
  FileVideo,
  AlertCircle,
  CheckCircle,
  X,
  Brain,
} from "lucide-react";
import toast from "react-hot-toast";
import VideoMetadataCard from "./VideoMetadataCard/VideoMetadataCard";
// import { AIInsightBanner } from "@/components/AIInsightBanner";
import { VideoPreviewModal } from "@/shared/components/video/VideoPreviewModal";
import { useLanguageDetection } from "../../../hooks/useLanguageDetection";
import type { VideoMetadata, FileUploadProps } from "../../../types";

function FileUpload({
  onFileSelect,
  maxSize,
  onMetadataDetected,
  onLanguageDetected,
  onFileUnload,
}: FileUploadProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [videoMetadata, setVideoMetadata] = useState<VideoMetadata | null>(
    null
  );
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiInsights, setAiInsights] = useState<any[]>([]);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  const { isDetecting, detectedLanguage, detectLanguage } =
    useLanguageDetection();

  // No extra preview component needed; we will render existing card in place

  // Unload file and reset state
  const handleUnload = useCallback(() => {
    setUploadedFile(null);
    setVideoMetadata(null);
    setAiInsights([]);
    setIsAnalyzing(false);
    onFileUnload?.();
  }, [onFileUnload]);

  // Analyze video file to extract metadata
  const analyzeVideo = useCallback(async (file: File) => {
    setIsAnalyzing(true);
    try {
      // Create video element to extract metadata
      const video = document.createElement("video");
      const url = URL.createObjectURL(file);

      return new Promise<VideoMetadata>((resolve, reject) => {
        const generateThumbnail = () => {
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");

          // Set canvas size to video dimensions
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;

          // Generate thumbnail
          if (ctx) {
            try {
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
              const thumbnail = canvas.toDataURL("image/jpeg", 0.7);

              const metadata: VideoMetadata = {
                name: file.name,
                size: file.size,
                duration: video.duration,
                width: video.videoWidth,
                height: video.videoHeight,
                fps: 30, // Default assumption
                bitrate: 0, // Will be calculated later
                codec: "MP4", // Default assumption, could be enhanced
                audioCodec: "AAC", // Default assumption
                audioChannels: 2, // Default assumption
                audioSampleRate: 44100, // Default assumption
                resolution: `${video.videoWidth}x${video.videoHeight}`,
                thumbnail: thumbnail,
              };

              URL.revokeObjectURL(url);
              resolve(metadata);
            } catch (error) {
              console.error("Error generating thumbnail:", error);
              // Fallback without thumbnail
              const metadata: VideoMetadata = {
                name: file.name,
                size: file.size,
                duration: video.duration,
                width: video.videoWidth,
                height: video.videoHeight,
                fps: 30, // Default assumption
                bitrate: 0, // Will be calculated later
                codec: "MP4",
                audioCodec: "AAC", // Default assumption
                audioChannels: 2, // Default assumption
                audioSampleRate: 44100, // Default assumption
                resolution: `${video.videoWidth}x${video.videoHeight}`,
              };

              URL.revokeObjectURL(url);
              resolve(metadata);
            }
          } else {
            // Fallback if canvas context is not available
            const metadata: VideoMetadata = {
              name: file.name,
              size: file.size,
              duration: video.duration,
              width: video.videoWidth,
              height: video.videoHeight,
              fps: 30, // Default assumption
              bitrate: 0, // Will be calculated later
              codec: "MP4",
              audioCodec: "AAC", // Default assumption
              audioChannels: 2, // Default assumption
              audioSampleRate: 44100, // Default assumption
              resolution: `${video.videoWidth}x${video.videoHeight}`,
            };

            URL.revokeObjectURL(url);
            resolve(metadata);
          }
        };

        video.onloadedmetadata = () => {
          // Seek to 1 second or 10% of duration to get a better thumbnail
          const seekTime = Math.min(1, video.duration * 0.1);
          video.currentTime = seekTime;

          // Fallback: if seeking fails, try to generate thumbnail from first frame
          setTimeout(() => {
            if (video.readyState >= 2) {
              // HAVE_CURRENT_DATA
              generateThumbnail();
            }
          }, 1_000);
        };

        video.onseeked = () => {
          generateThumbnail();
        };

        video.onerror = e => {
          console.error("Video loading error:", e);
          URL.revokeObjectURL(url);
          reject(new Error("Failed to load video"));
        };

        // Add timeout to prevent hanging
        const timeout = setTimeout(() => {
          URL.revokeObjectURL(url);
          reject(new Error("Video analysis timeout"));
        }, 10_000);

        video.onloadeddata = () => {
          clearTimeout(timeout);
        };

        video.src = url;
      });
    } catch (error) {
      console.error("Error analyzing video:", error);
      throw error;
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        setUploadedFile(file);
        onFileSelect(file);

        try {
          const metadata = await analyzeVideo(file);
          setVideoMetadata(metadata);
          onMetadataDetected?.(metadata);

          // Language detection is now handled by the backend after upload
          // const languageResult = await detectLanguage(file);
          // if (languageResult?.language) {
          //   onLanguageDetected?.(
          //     languageResult.language,
          //     languageResult.confidence
          //   );
          // }

          // Generate AI insights based on video metadata
          const insights = [];
          const estimatedChunks = Math.ceil(metadata.duration / 30); // Assume 30s chunks

          if (metadata.duration > 300) {
            // 5 minutes
            insights.push({
              type: "info" as const,
              title: "Long Video Detected",
              message: `This ${Math.round(metadata.duration / 60)} minute video will be processed in ${estimatedChunks} segments for optimal quality.`,
            });
          }

          if (metadata.duration < 30) {
            insights.push({
              type: "warning" as const,
              title: "Short Video",
              message:
                "Very short videos may not benefit from chunking. Processing will be optimized accordingly.",
            });
          }

          // Language detection insight will be added by backend after session creation
          // if (languageResult?.language) {
          //   insights.push({
          //     type: "success" as const,
          //     title: "Language Auto-Detected",
          //     message: `Source language detected as ${languageResult.language} with ${Math.round(languageResult.confidence * 100)}% confidence.`,
          //   });
          // }

          setAiInsights(insights);
        } catch (error) {
          console.error("Failed to analyze video:", error);
          toast.error("Failed to analyze video metadata");
        }
      }
    },
    [onFileSelect, onMetadataDetected, analyzeVideo]
  );

  const { getRootProps, getInputProps, isDragReject } = useDropzone({
    onDrop,
    accept: {
      "video/mp4": [".mp4"],
      "video/avi": [".avi"],
      "video/mov": [".mov"],
      "video/wmv": [".wmv"],
      "video/flv": [".flv"],
      "video/webm": [".webm"],
      "video/mkv": [".mkv"],
    },
    maxSize,
    multiple: false,
    onDragEnter: () => setIsDragActive(true),
    onDragLeave: () => setIsDragActive(false),
    onDropAccepted: () => {
      setIsDragActive(false);
      toast.success("File uploaded successfully!");
    },
    onDropRejected: fileRejections => {
      setIsDragActive(false);
      const error = fileRejections[0]?.errors[0];
      if (error?.code === "file-too-large") {
        const maxSizeGB = maxSize ?? 100 * 1_024 * 1_024 * 1_024;
        const sizeInGB = maxSizeGB / (1_024 * 1_024 * 1_024);
        toast.error(
          `File is too large. Maximum size is ${sizeInGB >= 1 ? Math.round(sizeInGB) + 'GB' : Math.round(sizeInGB * 1024) + 'MB'}`
        );
      } else if (error?.code === "file-invalid-type") {
        toast.error("Invalid file type. Please upload a video file.");
      } else {
        toast.error("Failed to upload file");
      }
    },
  });

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1_024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="w-full space-y-6">
      {uploadedFile ? (
        // Move the existing metadata card to this position and hide the dropzone
        <VideoMetadataCard
          metadata={videoMetadata}
          fileName={uploadedFile?.name ?? ""}
          fileSize={uploadedFile?.size ?? 0}
          onPreview={() => setIsPreviewOpen(true)}
          onUnload={handleUnload}
        />
      ) : (
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? "dropzone-active" : ""} ${isDragReject ? "border-red-500 bg-red-50 dark:bg-red-900/20" : ""}`}
          role="button"
          tabIndex={0}
          aria-label="Upload video file"
          aria-describedby="upload-instructions"
          onKeyDown={e => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              const input = e.currentTarget.querySelector(
                'input[type="file"]'
              ) as HTMLInputElement;
              input?.click();
            }
          }}
        >
          <input
            {...getInputProps()}
            aria-label="Select video file"
            aria-describedby="upload-instructions"
          />

          <div className="text-center">
            {(isAnalyzing ?? false) || (isDetecting ?? false) ? (
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="space-y-4"
              >
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/20">
                  <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-blue-600"></div>
                </div>
                <div className="flex flex-col gap-1">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {isDetecting ? "Detecting Language..." : "Analyzing Video..."}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {isDetecting
                      ? "Using AI to detect the source language"
                      : "Extracting metadata and generating AI insights"}
                  </p>
                </div>
              </motion.div>
            ) : isDragReject ? (
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="space-y-4"
              >
                <AlertCircle className="mx-auto h-16 w-16 text-red-500" />
                <div className="flex flex-col gap-2">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    Invalid File Type
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Please upload a video file
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="space-y-4"
              >
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900/20">
                  <Upload className="h-8 w-8 text-blue-600" />
                </div>

                <div className="flex flex-col gap-2">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {isDragActive
                      ? "Drop your video here"
                      : "Upload a video file"}
                  </h3>
                  <p
                    id="upload-instructions"
                    className="text-sm text-gray-600 dark:text-gray-400"
                  >
                    Drag and drop your video file here, or click to browse. You
                    can also press Enter or Space to open the file dialog.
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-500">
                    Supported formats: MP4, AVI, MOV, MKV, WebM
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-500">
                    Maximum size:{" "}
                    {(() => {
                      const maxSizeGB = maxSize ?? 100 * 1_024 * 1_024 * 1_024;
                      const sizeInGB = maxSizeGB / (1_024 * 1_024 * 1_024);
                      return sizeInGB >= 1 ? Math.round(sizeInGB) + 'GB' : Math.round(sizeInGB * 1024) + 'MB';
                    })()}
                  </p>
                </div>

                <div className="flex items-center justify-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
                  <FileVideo className="h-4 w-4" />
                  <span>Video files only</span>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      )}

      {/* Metadata card moved above when file is uploaded */}

      {/* AI Insights Banner */}
      {/* {aiInsights.length > 0 && videoMetadata && (
        <AIInsightBanner
          insights={aiInsights}
          estimatedChunks={Math.ceil(videoMetadata.duration / 30)}
          processingTime={`${Math.round((videoMetadata.duration / 60) * 0.5)} minutes`}
        />
      )} */}

      {/* Video Preview Modal */}
      <VideoPreviewModal
        isOpen={isPreviewOpen}
        onClose={() => setIsPreviewOpen(false)}
        videoFile={uploadedFile}
        metadata={videoMetadata}
      />
    </div>
  );
}

export default FileUpload;
