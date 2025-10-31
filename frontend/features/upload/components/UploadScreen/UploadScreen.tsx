"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  startTransition,
  useDeferredValue,
  useState,
} from "react";
import dynamic from "next/dynamic";
import { useTranslationStore } from "@/stores/translationStore";
import { UPLOAD_CONSTANTS } from "../../constants";
import { formatDuration } from "../../utils/chunking";
import { HeroSection, FeaturesSection } from "..";
import LanguageSelection from "./LanguageSelection/LanguageSelection";
import FileUpload from "./FileUpload/FileUpload";
import { useLanguageDetection, useFileUpload } from "../../hooks";

// Next.js dynamic imports for better code splitting
const DynamicAIChunkingStrategy = dynamic(
  () => import("./AIChunkingStrategy/AIChunkingStrategy"),
  {
    loading: () => <ComponentLoader />,
    ssr: false,
  }
);

const DynamicTranslationReady = dynamic(
  () => import("./TranslationReady/TranslationReady"),
  {
    loading: () => <ComponentLoader />,
    ssr: false,
  }
);

const DynamicStatsSection = dynamic(
  () => import("./StatsSection/StatsSection"),
  {
    loading: () => <ComponentLoader />,
    ssr: false,
  }
);

const DynamicLanguageDetectionLoading = dynamic(
  () => import("./LanguageDetectionLoading/LanguageDetectionLoading"),
  {
    loading: () => <ComponentLoader />,
    ssr: false,
  }
);

// Loading fallback component
const ComponentLoader = () => (
  <div className="flex items-center justify-center p-4">
    <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-blue-600"></div>
  </div>
);

// Preload components that are likely to be needed
const preloadComponents = () => {
  if (typeof window !== "undefined") {
    import(".").then(() => {
      // Components are now in cache
    });
  }
};

// Render boundary components
const StaticContent = () => (
  <>
    <HeroSection />
    <FeaturesSection />
  </>
);

const InteractiveContent = ({
  sourceLanguage,
  targetLanguage,
  isDetecting,
  detectedLanguage,
  setSourceLanguage,
  setTargetLanguage,
}: {
  sourceLanguage: string;
  targetLanguage: string;
  isDetecting: boolean;
  detectedLanguage: string | null;
  setSourceLanguage: (lang: string) => void;
  setTargetLanguage: (lang: string) => void;
}) => {
  const languageProps = {
    sourceLanguage,
    targetLanguage,
    isDetecting,
    detectedLanguage,
    onSourceLanguageChange: setSourceLanguage,
    onTargetLanguageChange: setTargetLanguage,
  };

  return <LanguageSelection {...languageProps} />;
};

export default function UploadPage() {
  // React Compiler automatically optimizes selectors
  const currentSession = useTranslationStore(state => state.currentSession);
  const sourceLanguage = useTranslationStore(state => state.sourceLanguage);
  const targetLanguage = useTranslationStore(state => state.targetLanguage);
  const setSourceLanguage = useTranslationStore(
    state => state.setSourceLanguage
  );
  const setTargetLanguage = useTranslationStore(
    state => state.setTargetLanguage
  );

  // Local state for chunking info
  const [chunkingInfo, setChunkingInfo] = useState<any>(null);

  // Language detection hook
  const {
    isDetecting,
    detectedLanguage,
    setDetectedLanguage,
    handleLanguageDetected,
    detectLanguage,
  } = useLanguageDetection();

  // Local state for detection control
  const [isDetectingLocal, setIsDetectingLocal] = useState(false);

  // File upload hook
  const {
    videoMetadata,
    handleFileUpload,
    handleMetadataDetected,
    handleFileUnload,
  } = useFileUpload();

  // Enhanced file upload handler
  const handleFileUploadWithDetection = useCallback(
    async (file: File) => {
      try {
        setIsDetectingLocal(true);
        setDetectedLanguage(null);

        // Upload file first
        const uploadResult = await handleFileUpload(
          file,
          sourceLanguage,
          targetLanguage
        );

        // Then detect language using the session ID (not the file)
        if (uploadResult?.sessionId) {
          await detectLanguage(uploadResult.sessionId);
        }

        return uploadResult;
      } catch (error) {
        console.error("Upload with detection failed:", error);
        throw error;
      } finally {
        setIsDetectingLocal(false);
      }
    },
    [
      handleFileUpload,
      sourceLanguage,
      targetLanguage,
      handleLanguageDetected,
      detectLanguage,
      setIsDetectingLocal,
      setDetectedLanguage,
    ]
  );

  // Clear only current session on page load, preserve history
  useEffect(() => {
    const { currentSession, setCurrentSession } =
      useTranslationStore.getState();

    if (currentSession) {
      setCurrentSession(null);
    }
  }, []);

  // Update chunking info when session data changes
  useEffect(() => {
    const { totalChunks, sessionId } = currentSession || {};
    const { duration } = videoMetadata || {};

    let newChunkingInfo = null;

    if (totalChunks && totalChunks > 0) {
      newChunkingInfo = {
        chunkCount: totalChunks,
        chunkSize: UPLOAD_CONSTANTS.CHUNK_SIZE,
        message: `AI will split into ~${totalChunks} chunks to ensure perfect ${formatDuration(duration || 0)} sync`,
      };
    } else if (sessionId && duration && totalChunks === 0) {
      const estimatedChunks = Math.ceil(duration / UPLOAD_CONSTANTS.CHUNK_SIZE);
      newChunkingInfo = {
        chunkCount: estimatedChunks,
        chunkSize: UPLOAD_CONSTANTS.CHUNK_SIZE,
        message: `AI will analyze and split into ~${estimatedChunks} chunks to ensure perfect ${formatDuration(duration)} sync`,
      };
    }

    startTransition(() => {
      setChunkingInfo(newChunkingInfo);
    });
  }, [
    currentSession?.totalChunks,
    currentSession?.sessionId,
    videoMetadata?.duration,
    setChunkingInfo,
  ]);

  // Memoized translation readiness check
  const canStartTranslation = useMemo(() => {
    const { status } = currentSession || {};
    return (
      currentSession &&
      status === "uploaded" &&
      sourceLanguage &&
      targetLanguage &&
      detectedLanguage
    );
  }, [currentSession, sourceLanguage, targetLanguage, detectedLanguage]);

  // Deferred values for better partial pre-rendering
  const deferredChunkingInfo = useDeferredValue(chunkingInfo);
  const deferredCanStartTranslation = useDeferredValue(canStartTranslation);

  // Memoized conditional rendering flags
  const showChunkingInfo = useMemo(
    () => deferredChunkingInfo && sourceLanguage && targetLanguage,
    [deferredChunkingInfo, sourceLanguage, targetLanguage]
  );

  return (
    <div className="flex flex-col gap-8">
      <StaticContent />

      <InteractiveContent
        sourceLanguage={sourceLanguage}
        targetLanguage={targetLanguage}
        isDetecting={isDetecting}
        detectedLanguage={detectedLanguage}
        setSourceLanguage={setSourceLanguage}
        setTargetLanguage={setTargetLanguage}
      />

      <div
        className="mx-auto w-full max-w-4xl"
        onMouseEnter={preloadComponents}
      >
        <FileUpload
          onFileSelect={handleFileUploadWithDetection}
          maxSize={UPLOAD_CONSTANTS.MAX_FILE_SIZE}
          onMetadataDetected={metadata =>
            handleMetadataDetected(metadata, currentSession)
          }
          onLanguageDetected={handleLanguageDetected}
          onFileUnload={handleFileUnload}
        />
      </div>

      {showChunkingInfo && (
        <DynamicAIChunkingStrategy chunkingInfo={deferredChunkingInfo} />
      )}

      {isDetectingLocal && <DynamicLanguageDetectionLoading />}

      {deferredCanStartTranslation && <DynamicTranslationReady />}

      <DynamicStatsSection />
    </div>
  );
}
