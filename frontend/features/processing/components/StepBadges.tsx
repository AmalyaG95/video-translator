"use client";

import { motion } from "framer-motion";
import {
  CheckCircle,
  Clock,
  AlertCircle,
  FileVideo,
  Mic,
  Volume2,
  Globe,
  Zap,
  Download,
  Play,
  Brain,
  Info,
} from "lucide-react";

export type ProcessingStep = {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  status: "pending" | "in_progress" | "completed" | "failed" | "skipped";
  progress?: number;
  duration?: number;
  error?: string;
  aiReasoning?: string;
  substeps?: Array<{
    name: string;
    status: "pending" | "in_progress" | "completed" | "failed";
    progress?: number;
  }>;
};

interface StepBadgesProps {
  steps: ProcessingStep[];
  currentStep?: string;
  className?: string;
}

export function StepBadges({
  steps,
  currentStep,
  className = "",
}: StepBadgesProps) {
  const getStepColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-400 dark:border-green-800";
      case "in_progress":
        return "bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/20 dark:text-blue-400 dark:border-blue-800";
      case "failed":
        return "bg-red-100 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-800";
      case "skipped":
        return "bg-gray-100 text-gray-600 border-gray-200 dark:bg-gray-800/20 dark:text-gray-500 dark:border-gray-700";
      default:
        return "bg-gray-100 text-gray-600 border-gray-200 dark:bg-gray-800/20 dark:text-gray-500 dark:border-gray-700";
    }
  };

  const getStepIcon = (step: ProcessingStep) => {
    if (step.status === "in_progress") {
      return <Clock className="h-4 w-4 animate-spin" />;
    } else if (step.status === "failed") {
      return <AlertCircle className="h-4 w-4" />;
    } else {
      return step.icon;
    }
  };

  return (
    <div className={`space-y-4 ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
        Processing Steps
      </h3>
      <div className="relative">
        {/* Timeline connector line */}
        <div className="absolute bottom-0 left-8 top-0 hidden w-0.5 bg-gray-200 dark:bg-gray-700 md:block" />

        <div className="space-y-6">
          {steps.map((step, index) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="relative flex items-start space-x-4"
            >
              {/* Timeline dot */}
              <div
                className={`relative z-10 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border-2 ${
                  step.status === "completed"
                    ? "border-green-500 bg-green-500"
                    : step.status === "in_progress"
                      ? "border-blue-500 bg-blue-500"
                      : step.status === "failed"
                        ? "border-red-500 bg-red-500"
                        : "border-gray-300 bg-gray-300 dark:border-gray-600 dark:bg-gray-600"
                }`}
              >
                {getStepIcon(step)}
              </div>

              {/* Step content */}
              <motion.div
                className={`flex-1 rounded-lg border-2 p-4 transition-all duration-300 ${getStepColor(
                  step.status
                )} ${
                  currentStep === step.id
                    ? "ring-2 ring-blue-500 ring-opacity-50"
                    : ""
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="min-w-0 flex-1">
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                      {step.name}
                    </h4>
                    <p className="mt-1 text-xs text-gray-600 dark:text-gray-400">
                      {step.description}
                    </p>

                    {/* Duration display */}
                    {step.duration && (
                      <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Duration: {step.duration.toFixed(1)}s
                      </div>
                    )}
                  </div>

                  {/* AI Reasoning tooltip */}
                  {step.aiReasoning && (
                    <div className="group relative ml-2">
                      <Brain className="h-4 w-4 cursor-help text-blue-500" />
                      <div className="absolute right-0 top-6 z-10 w-64 rounded bg-gray-900 p-2 text-xs text-white opacity-0 shadow-lg transition-opacity group-hover:opacity-100 dark:bg-gray-100 dark:text-gray-900">
                        <div className="mb-1 font-medium">AI Reasoning:</div>
                        <div>{step.aiReasoning}</div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Progress bar for in-progress steps */}
                {step.status === "in_progress" &&
                  step.progress !== undefined && (
                    <div className="mt-3">
                      <div className="mb-1 flex justify-between text-xs text-gray-600 dark:text-gray-400">
                        <span>Progress</span>
                        <span>{Math.round(step.progress)}%</span>
                      </div>
                      <div className="h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                        <motion.div
                          className="h-2 rounded-full bg-blue-500"
                          initial={{ width: 0 }}
                          animate={{ width: `${step.progress}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                    </div>
                  )}

                {/* Substeps */}
                {step.substeps && step.substeps.length > 0 && (
                  <div className="mt-3 space-y-2">
                    <div className="text-xs font-medium text-gray-700 dark:text-gray-300">
                      Substeps:
                    </div>
                    {step.substeps.map((substep, subIndex) => (
                      <div
                        key={subIndex}
                        className="flex items-center space-x-2 text-xs"
                      >
                        <div
                          className={`h-2 w-2 rounded-full ${
                            substep.status === "completed"
                              ? "bg-green-500"
                              : substep.status === "in_progress"
                                ? "bg-blue-500"
                                : substep.status === "failed"
                                  ? "bg-red-500"
                                  : "bg-gray-300 dark:bg-gray-600"
                          }`}
                        />
                        <span className="text-gray-600 dark:text-gray-400">
                          {substep.name}
                        </span>
                        {substep.progress !== undefined && (
                          <span className="text-gray-500 dark:text-gray-500">
                            ({Math.round(substep.progress)}%)
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Error display */}
                {step.error && (
                  <div className="mt-3 rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-400">
                    <div className="font-medium">Error:</div>
                    <div>{step.error}</div>
                  </div>
                )}
              </motion.div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Default processing steps
export const DEFAULT_PROCESSING_STEPS: ProcessingStep[] = [
  {
    id: "initialization",
    name: "Initialization",
    description: "Loading AI models and preparing pipeline",
    icon: <Zap className="h-4 w-4" />,
    status: "pending",
  },
  {
    id: "audio_extraction",
    name: "Audio Extraction",
    description: "Extracting audio from video file",
    icon: <Volume2 className="h-4 w-4" />,
    status: "pending",
  },
  {
    id: "segmentation",
    name: "Segmentation",
    description: "Splitting audio into speech segments",
    icon: <FileVideo className="h-4 w-4" />,
    status: "pending",
  },
  {
    id: "stt_processing",
    name: "Speech Recognition",
    description: "Converting speech to text (STT)",
    icon: <Mic className="h-4 w-4" />,
    status: "pending",
  },
  {
    id: "translation",
    name: "Translation",
    description: "Translating text to target language",
    icon: <Globe className="h-4 w-4" />,
    status: "pending",
  },
  {
    id: "tts_generation",
    name: "Speech Synthesis",
    description: "Converting translated text to speech (TTS)",
    icon: <Volume2 className="h-4 w-4" />,
    status: "pending",
  },
  {
    id: "audio_sync",
    name: "Audio Synchronization",
    description: "Syncing translated audio with video timing",
    icon: <Play className="h-4 w-4" />,
    status: "pending",
  },
  {
    id: "video_assembly",
    name: "Video Assembly",
    description: "Combining video with translated audio",
    icon: <FileVideo className="h-4 w-4" />,
    status: "pending",
  },
  {
    id: "finalization",
    name: "Finalization",
    description: "Generating preview and exporting files",
    icon: <Download className="h-4 w-4" />,
    status: "pending",
  },
];
