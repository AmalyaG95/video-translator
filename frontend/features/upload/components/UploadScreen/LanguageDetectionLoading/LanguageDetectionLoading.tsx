"use client";

import React from "react";
import { motion } from "framer-motion";

function LanguageDetectionLoading() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="mx-auto w-full max-w-4xl"
    >
      <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4 dark:border-yellow-800 dark:bg-yellow-900/20">
        <div className="flex items-center space-x-3">
          <div className="h-5 w-5 animate-spin rounded-full border-b-2 border-yellow-600"></div>
          <span className="font-medium text-yellow-800 dark:text-yellow-200">
            AI detecting language...
          </span>
        </div>
      </div>
    </motion.div>
  );
}

export default LanguageDetectionLoading;
