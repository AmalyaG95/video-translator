"use client";

import React from "react";
import { motion } from "framer-motion";
import { CheckCircle, Play, Loader2 } from "lucide-react";
import { useTranslation } from "../../../hooks/useTranslation";

const TranslationReady = () => {
  const { handleStartTranslation, isStartingTranslation } = useTranslation();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="mx-auto w-full max-w-4xl text-center"
    >
      <div className="card border-blue-200 bg-gradient-to-r from-blue-50 to-purple-50 dark:border-blue-700 dark:from-blue-900/20 dark:to-purple-900/20">
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-center space-x-2 text-blue-600 dark:text-blue-400">
            <CheckCircle className="h-6 w-6" />
            <span className="text-lg font-semibold">
              Video Ready for Translation
            </span>
          </div>
          <p className="text-gray-600 dark:text-gray-300">
            Your video has been uploaded successfully. Click below to start the
            AI translation process.
          </p>

          <button
            onClick={handleStartTranslation}
            disabled={isStartingTranslation}
            className="btn-primary mx-auto flex items-center space-x-2 px-8 py-4 text-lg disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isStartingTranslation ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Play className="h-5 w-5" />
            )}
            <span>
              {isStartingTranslation ? "Starting..." : "Start Translation"}
            </span>
          </button>
        </div>
      </div>
    </motion.div>
  );
};

export default TranslationReady;
