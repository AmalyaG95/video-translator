"use client";

import React from "react";
import { motion } from "framer-motion";
import { Brain, Loader2 } from "lucide-react";

interface ChunkingInfo {
  message: string;
  chunkCount: number;
  chunkSize: number;
}

interface AIChunkingStrategyProps {
  chunkingInfo: ChunkingInfo | null;
}

function AIChunkingStrategy({ chunkingInfo }: AIChunkingStrategyProps) {
  if (!chunkingInfo) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="mx-auto w-full max-w-4xl"
    >
      <div className="rounded-lg border border-blue-200 bg-gradient-to-r from-blue-50 to-purple-50 p-6 dark:border-blue-800 dark:from-blue-900/20 dark:to-purple-900/20">
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0">
            <Brain className="h-6 w-6 text-blue-600 dark:text-blue-400" />
          </div>
          <div className="flex-1">
            <h3 className="mb-2 text-lg font-semibold text-gray-900 dark:text-white">
              AI Chunking Strategy
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              {chunkingInfo.message}
            </p>
            <div className="mt-3 flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
              {chunkingInfo.chunkCount === 0 ? (
                <div className="flex items-center space-x-2">
                  <Loader2 className="inline h-4 w-4 animate-spin" />
                  <span>Calculating segments...</span>
                </div>
              ) : (
                <>
                  <span>• {chunkingInfo.chunkCount} segments</span>
                  <span>• {chunkingInfo.chunkSize}s each</span>
                  <span>• Perfect sync guaranteed</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default AIChunkingStrategy;
