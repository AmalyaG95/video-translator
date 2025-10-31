"use client";

import React from "react";
import { motion } from "framer-motion";
import { CheckCircle, Clock, Play, Pause, AlertCircle } from "lucide-react";
import {
  Brain,
  Zap,
  Music,
  Scissors,
  Mic,
  Globe,
  Volume2,
  RefreshCw,
} from "lucide-react";

interface ProcessingStep {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  status: "pending" | "in_progress" | "completed" | "failed";
  progress?: number;
  duration?: number;
  aiReasoning?: string;
}

interface ProcessingTimelineProps {
  session: {
    progress: number;
    currentStep: string;
    status: string;
    isPaused?: boolean;
    etaSeconds?: number;
    currentChunk?: number;
    totalChunks?: number;
    processingSpeed?: number;
  };
  onStart?: () => void;
  onPause?: () => void;
  onResume?: () => void;
  onCancel?: () => void;
  onViewResults?: () => void;
}

export function ProcessingTimeline({
  session,
  onStart,
  onPause,
  onResume,
  onCancel,
  onViewResults,
}: ProcessingTimelineProps) {
  // Use totalChunks from session (calculated during upload)
  // Preserve it even if it becomes 0 (prefer saved value over 0)
  // Check multiple sources: session.totalChunks, currentChunk
  const totalChunks =
    session.totalChunks && session.totalChunks > 0
      ? session.totalChunks
      : session.currentChunk && session.currentChunk > 0
        ? session.currentChunk // If totalChunks is lost, use currentChunk as fallback
        : session.totalChunks || 0; // Keep whatever value exists
  const steps: ProcessingStep[] = [
    {
      id: "initialization",
      name: "Initialization",
      description: "Loading AI models and preparing pipeline",
      icon: <Zap className="h-4 w-4" />,
      status: session.progress > 0 ? "completed" : "pending",
      aiReasoning: "Using optimized model loading strategy for faster startup",
    },
    {
      id: "audio_extraction",
      name: "Audio Extraction",
      description: "Extracting audio from video file",
      icon: <Music className="h-4 w-4" />,
      status:
        session.progress > 10
          ? "completed"
          : session.progress > 5
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, (session.progress - 5) * 20)),
      aiReasoning: "High-quality audio extraction with noise reduction",
    },
    {
      id: "segmentation",
      name: "Segmentation",
      description: "Splitting audio into speech segments",
      icon: <Scissors className="h-4 w-4" />,
      status:
        session.progress > 30
          ? "completed"
          : session.progress > 15
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, (session.progress - 15) * 6.67)),
      aiReasoning: "Adaptive chunking based on speech density analysis",
    },
    {
      id: "stt_processing",
      name: "Speech Recognition",
      description: "Converting speech to text (STT)",
      icon: <Mic className="h-4 w-4" />,
      status:
        session.progress > 60
          ? "completed"
          : session.progress > 40
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, (session.progress - 40) * 5)),
      aiReasoning: "Whisper model with VAD for accurate transcription",
    },
    {
      id: "translation",
      name: "Translation",
      description: "Translating text to target language",
      icon: <Globe className="h-4 w-4" />,
      status:
        session.progress > 80
          ? "completed"
          : session.progress > 60
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, (session.progress - 60) * 5)),
      aiReasoning: "Helsinki-NLP with text condensation for timing",
    },
    {
      id: "tts_processing",
      name: "Voice Synthesis",
      description: "Converting text to speech (TTS)",
      icon: <Volume2 className="h-4 w-4" />,
      status:
        session.progress > 95
          ? "completed"
          : session.progress > 80
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, (session.progress - 80) * 6.67)),
      aiReasoning: "Edge-TTS with voice matching for natural pronunciation",
    },
    {
      id: "synchronization",
      name: "Synchronization",
      description: "Syncing audio with video timing",
      icon: <RefreshCw className="h-4 w-4" />,
      status:
        session.progress >= 100
          ? "completed"
          : session.progress > 95
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, (session.progress - 95) * 20)),
      aiReasoning: "Frame-accurate sync with atempo adjustments",
    },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "in_progress":
        return <Clock className="h-5 w-5 text-blue-500" />;
      case "failed":
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      default:
        return (
          <div className="h-5 w-5 rounded-full border-2 border-gray-300" />
        );
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700";
      case "in_progress":
        return "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-700";
      case "failed":
        return "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700";
      default:
        return "bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600";
    }
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800">
      {/* Header with Progress and Controls */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex-1">
          <div className="mb-2 flex items-center gap-3">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Processing Timeline
            </h3>
            {session.isPaused && (
              <span className="rounded-full bg-orange-100 px-2 py-1 text-xs font-medium text-orange-800">
                Paused
              </span>
            )}
          </div>

          {/* Progress Bar */}
          <div className="mb-2 h-3 w-full rounded-full bg-gray-200">
            <motion.div
              className="h-3 rounded-full bg-gradient-to-r from-blue-500 to-green-500"
              initial={{ width: 0 }}
              animate={{ width: `${session.progress}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          </div>

          <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-300">
            <span>
              {session.status === "completed" && session.progress === 100
                ? "100.0% Complete"
                : session.progress > 0
                  ? `${session.progress.toFixed(1)}% Complete`
                  : "Starting translation..."}
            </span>
            <span>
              {session.status === "completed" && session.progress === 100
                ? `All ${totalChunks > 0 ? totalChunks : session.currentChunk && session.currentChunk > 0 ? session.currentChunk : "?"} chunks completed`
                : session.currentStep &&
                    session.currentStep.includes("Processing segment")
                  ? session.currentStep.match(
                      /Processing segment (\d+)\/(\d+)/
                    )?.[0] || "Processing..."
                  : session.currentChunk !== undefined &&
                      session.currentChunk >= 0
                    ? totalChunks
                      ? `Chunk ${session.currentChunk + 1}/${totalChunks}`
                      : `Chunk ${session.currentChunk + 1}`
                    : session.status === "processing"
                      ? "Processing..."
                      : "Initializing..."}
            </span>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="ml-4 flex gap-2">
          {session.status === "completed" && session.progress === 100 ? (
            <button
              onClick={onViewResults}
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700"
            >
              <CheckCircle className="h-4 w-4" />
              View Results
            </button>
          ) : (
            <>
              {session.isPaused ? (
                <button
                  onClick={onResume}
                  className="flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-white transition-colors hover:bg-green-700"
                >
                  <Play className="h-4 w-4" />
                  Resume
                </button>
              ) : (
                <button
                  onClick={onPause}
                  className="flex items-center gap-2 rounded-lg bg-orange-600 px-4 py-2 text-white transition-colors hover:bg-orange-700"
                >
                  <Pause className="h-4 w-4" />
                  Pause
                </button>
              )}
              <button
                onClick={onCancel}
                className="flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2 text-white transition-colors hover:bg-red-700"
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>

      {/* Steps Timeline */}
      <div className="space-y-4">
        {steps.map((step, index) => (
          <motion.div
            key={step.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`relative rounded-lg border-2 p-4 transition-all duration-300 ${getStatusColor(
              step.status
            )}`}
          >
            {/* Connector Line */}
            {index < steps.length - 1 && (
              <div className="absolute left-6 top-16 h-8 w-0.5 bg-gray-300" />
            )}

            <div className="flex items-start gap-4">
              {/* Step Icon */}
              <div className="mt-1 flex-shrink-0">
                {getStatusIcon(step.status)}
              </div>

              {/* Step Content */}
              <div className="min-w-0 flex-1">
                <div className="mb-2 flex items-center gap-3">
                  <div className="flex items-center gap-2 text-gray-700 dark:text-gray-200">
                    {step.icon}
                    <h4 className="font-medium">{step.name}</h4>
                  </div>
                  {step.aiReasoning && (
                    <div className="group relative">
                      <Brain className="h-4 w-4 cursor-help text-blue-500" />
                      <div className="absolute bottom-full left-1/2 z-10 mb-2 -translate-x-1/2 transform whitespace-nowrap rounded-lg bg-gray-900 px-3 py-2 text-xs text-white opacity-0 transition-opacity duration-200 group-hover:opacity-100">
                        {step.aiReasoning}
                      </div>
                    </div>
                  )}
                </div>

                <p className="mb-3 text-sm text-gray-600 dark:text-gray-300">
                  {step.description}
                </p>

                {/* Progress Bar for In-Progress Steps */}
                {step.status === "in_progress" &&
                  step.progress !== undefined && (
                    <div className="mb-2 h-2 w-full rounded-full bg-gray-200">
                      <motion.div
                        className="h-2 rounded-full bg-blue-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${step.progress}%` }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  )}

                {/* Status Text */}
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                  <span className="capitalize">
                    {step.status.replace("_", " ")}
                  </span>
                  {step.duration && <span>• {step.duration}s</span>}
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Processing Speed Info */}
      {!!session.processingSpeed && (
        <div className="mt-4 rounded-lg bg-gray-50 p-3 dark:bg-gray-700/50">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-300">
              Processing Speed
            </span>
            <span className="font-medium text-gray-900 dark:text-white">
              {session.processingSpeed} chunks/min
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default ProcessingTimeline;
