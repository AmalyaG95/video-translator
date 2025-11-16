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
  FileVideo,
  Layers,
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
    metadata?: { duration?: number };
    result?: { segments?: any[]; originalDuration?: number };
    duration?: number;
    // Detailed progress fields
    segments_processed?: number;
    current_time?: number;
    current_time_formatted?: string;
    total_duration?: number;
    total_duration_formatted?: string;
    progress_percent?: number;
    elapsed_time?: number;
    stage?: string;
    stage_number?: number;
    total_stages?: number;
    stage_progress_percent?: number;
  };
  onStart?: () => void;
  onPause?: () => void;
  onResume?: () => void;
  onCancel?: () => void;
  onViewResults?: () => void;
}

// Helper function to format time
function formatTime(seconds: number): string {
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
}

// Helper function to format stage name
function formatStageName(stage: string | undefined): string {
  if (!stage) return 'Processing';
  return stage.replace('_', ' ').split(' ').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1)
  ).join(' ');
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
  // totalChunks actually represents the number of speech segments from Whisper
  // For completed sessions, try to get a meaningful count even if data was lost
  const getTotalChunks = (): number | null => {
    // Priority 1: If we have a valid totalChunks, use it
    if (session.totalChunks !== undefined && session.totalChunks !== null && session.totalChunks > 0) {
      return session.totalChunks;
    }
    
    // Priority 2: For completed sessions, if currentChunk > 0, use it as total
    if (session.status === "completed" && session.currentChunk !== undefined && session.currentChunk !== null && session.currentChunk > 0) {
      return session.currentChunk;
    }
    
    // Priority 3: If processing is complete but no chunk data, check metadata
    if (session.status === "completed" && session.progress === 100) {
      // Try to estimate from duration if available (approximate: 1 segment per 3-5 seconds of speech)
      if (session.metadata?.duration && session.metadata.duration > 0) {
        // Estimate based on typical speech density (assuming ~40% speech, 1 segment per 4 seconds of speech)
        const estimatedSegments = Math.max(1, Math.ceil((session.metadata.duration * 0.4) / 4));
        return estimatedSegments;
      }
      // If we have result data with segments, use that
      if (session.result?.segments && Array.isArray(session.result.segments) && session.result.segments.length > 0) {
        return session.result.segments.length;
      }
      // Check if we have duration from result
      if (session.result?.originalDuration && session.result.originalDuration > 0) {
        const estimatedSegments = Math.max(1, Math.ceil((session.result.originalDuration * 0.4) / 4));
        return estimatedSegments;
      }
    }
    
    // Priority 4: During processing, if we have currentChunk, use it as estimate (multiply by 1.2 for estimate)
    if (session.currentChunk !== undefined && session.currentChunk !== null && session.currentChunk > 0) {
      return Math.ceil(session.currentChunk * 1.2); // Estimate total as 20% more than current
    }
    
    // Priority 5: Check if duration exists to estimate
    if (session.duration && session.duration > 0) {
      const estimatedSegments = Math.max(1, Math.ceil((session.duration * 0.4) / 4));
      return estimatedSegments;
    }
    
    return null; // Return null if we truly have no data
  };
  
  const totalChunks = getTotalChunks();
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
        session.progress > 30
          ? "completed"
          : session.progress > 20
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, (session.progress - 20) * 10)),
      aiReasoning: "Whisper model with VAD for accurate transcription",
    },
    {
      id: "translation",
      name: "Translation",
      description: "Translating text to target language",
      icon: <Globe className="h-4 w-4" />,
      status:
        session.progress > 70
          ? "completed"
          : session.progress > 30
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, ((session.progress - 30) / 40) * 100)),
      aiReasoning: "Helsinki-NLP with text condensation for timing",
    },
    {
      id: "tts_processing",
      name: "Voice Synthesis",
      description: "Converting text to speech (TTS)",
      icon: <Volume2 className="h-4 w-4" />,
      status:
        session.progress > 70
          ? "completed"
          : session.progress > 30
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, ((session.progress - 30) / 40) * 100)),
      aiReasoning: "Edge-TTS with voice matching for natural pronunciation",
    },
    {
      id: "audio_sync",
      name: "Audio Synchronization",
      description: "Syncing translated audio with video timing",
      icon: <RefreshCw className="h-4 w-4" />,
      status:
        session.progress >= 85
          ? "completed"
          : session.progress > 70
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, ((session.progress - 70) / 15) * 100)),
      aiReasoning: "Frame-accurate sync with atempo adjustments for natural speech timing",
    },
    {
      id: "video_combination",
      name: "Video Combination",
      description: "Combining video with translated audio and subtitles",
      icon: <Layers className="h-4 w-4" />,
      status:
        session.progress >= 100
          ? "completed"
          : session.progress > 85
            ? "in_progress"
            : "pending",
      progress: Math.max(0, Math.min(100, ((session.progress - 85) / 15) * 100)),
      aiReasoning: "Final video assembly with translated audio and embedded subtitles",
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

          <div className="flex flex-col gap-2 text-sm text-gray-600 dark:text-gray-300">
            {/* Main progress info */}
            <div className="flex items-center justify-between">
              <span>
                {session.status === "completed" && session.progress === 100
                  ? "100.0% Complete"
                  : session.progress > 0
                    ? `${session.progress.toFixed(1)}% Complete`
                    : "Starting translation..."}
              </span>
              <span className="font-medium">
                {session.currentStep || "Processing..."}
              </span>
            </div>
            
            {/* Detailed progress info - show when processing or when detailed data is available */}
            {(session.status === "processing" || 
              session.segments_processed !== undefined || 
              session.current_time_formatted !== undefined ||
              session.stage !== undefined) && (
              <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                {/* Stage info */}
                {session.stage_number && session.total_stages && (
                  <span>
                    Stage {session.stage_number}/{session.total_stages}: {formatStageName(session.stage)}
                  </span>
                )}
                
                {/* Segments info */}
                {session.segments_processed !== undefined && (
                  <span>
                    {session.segments_processed} {session.totalChunks ? `/${session.totalChunks}` : ''} segments
                    {session.progress_percent !== undefined && ` (${session.progress_percent.toFixed(1)}%)`}
                  </span>
                )}
                
                {/* Time info */}
                {session.current_time_formatted && session.total_duration_formatted && (
                  <span>
                    Time: {session.current_time_formatted} / {session.total_duration_formatted}
                  </span>
                )}
                
                {/* Elapsed time */}
                {session.elapsed_time !== undefined && (
                  <span>
                    Elapsed: {formatTime(session.elapsed_time)}
                  </span>
                )}
              </div>
            )}
            
            {/* Completed status */}
            {session.status === "completed" && session.progress === 100 && (
              <div className="text-xs text-green-600 dark:text-green-400">
                {totalChunks && totalChunks > 0
                  ? `All ${totalChunks} ${totalChunks === 1 ? 'segment' : 'segments'} processed successfully`
                  : session.currentChunk && session.currentChunk > 0
                    ? `All ${session.currentChunk} ${session.currentChunk === 1 ? 'segment' : 'segments'} processed successfully`
                    : "Translation completed successfully"}
              </div>
            )}
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

                {/* Status Text with detailed info */}
                <div className="flex flex-col gap-1 text-xs text-gray-500 dark:text-gray-400">
                  <div className="flex items-center gap-2">
                    <span className="capitalize">
                      {step.status.replace("_", " ")}
                    </span>
                    {step.duration && <span>• {step.duration}s</span>}
                    {step.id === "initialization" && step.status === "in_progress" && session.etaSeconds && (
                      <span>• Est. {Math.ceil(session.etaSeconds / 60)} min</span>
                    )}
                  </div>
                  
                  {/* Show detailed info for active stages */}
                  {step.status === "in_progress" && 
                    (session.stage === step.id || 
                     (step.id === "stt_processing" && session.stage === "transcription") ||
                     (step.id === "tts_processing" && session.stage === "tts") ||
                     (step.id === "audio_sync" && session.stage === "audio_sync") ||
                     (step.id === "video_combination" && session.stage === "video_combination") ||
                     (step.id === "segmentation" && session.stage === "transcription")) && (
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      {session.segments_processed !== undefined && (
                        <span className="text-blue-600 dark:text-blue-400">
                          {session.segments_processed} {session.totalChunks ? `/${session.totalChunks}` : ''} segments
                        </span>
                      )}
                      {session.current_time_formatted && session.total_duration_formatted && (
                        <span className="text-gray-600 dark:text-gray-300">
                          {session.current_time_formatted} / {session.total_duration_formatted}
                        </span>
                      )}
                      {session.progress_percent !== undefined && (
                        <span className="text-green-600 dark:text-green-400">
                          {session.progress_percent.toFixed(1)}%
                        </span>
                      )}
                    </div>
                  )}
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
