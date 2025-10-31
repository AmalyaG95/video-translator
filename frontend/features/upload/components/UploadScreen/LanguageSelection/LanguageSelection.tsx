"use client";

import LanguageSelector from "./LanguageSelector/LanguageSelector";
import { SUPPORTED_LANGUAGES } from "@/constants";
import SourceLanguageSelector from "./SourceLanguageSelector/SourceLanguageSelector";

interface LanguageSelectionProps {
  sourceLanguage: string;
  targetLanguage: string;
  isDetecting: boolean;
  detectedLanguage: string | null;
  onSourceLanguageChange: (lang: string) => void;
  onTargetLanguageChange: (lang: string) => void;
}

function LanguageSelection({
  sourceLanguage,
  targetLanguage,
  isDetecting,
  detectedLanguage,
  onSourceLanguageChange,
  onTargetLanguageChange,
}: LanguageSelectionProps) {
  return (
    <div className="card mx-auto flex w-full max-w-4xl flex-col gap-6">
      <h2 className="text-center text-2xl font-bold text-gray-900 dark:text-white">
        Choose Your Languages
      </h2>

      <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
        <SourceLanguageSelector
          sourceLanguage={sourceLanguage}
          isDetecting={isDetecting}
          detectedLanguage={detectedLanguage}
        />

        <div className="flex flex-col gap-3">
          <LanguageSelector
            value={targetLanguage}
            onChange={onTargetLanguageChange}
            selectedLanguage={targetLanguage}
            onLanguageChange={onTargetLanguageChange}
            languages={SUPPORTED_LANGUAGES}
            label="Target Language"
            disabledLanguages={sourceLanguage ? [sourceLanguage] : []}
            disabled={!sourceLanguage}
          />
          {!sourceLanguage && (
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Upload a video to detect source language first
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default LanguageSelection;
