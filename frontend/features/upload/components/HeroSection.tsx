"use client";

import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";

function HeroSection() {
  return (
    <div className="flex flex-col gap-4 text-center">
      <motion.div
        initial={{ scale: 0.9 }}
        animate={{ scale: 1 }}
        transition={{ duration: 0.5 }}
        className="mx-auto inline-flex items-center space-x-2 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-3 text-sm font-medium text-white"
      >
        <Sparkles className="h-4 w-4" />
        <span>AI-Powered Video Translation</span>
      </motion.div>

      <h1 className="text-4xl font-bold text-gray-900 dark:text-white md:text-6xl">
        Translate Videos with
        <span className="text-gradient block">Perfect Lip-Sync</span>
      </h1>

      <p className="mx-auto max-w-3xl text-xl text-gray-600 dark:text-gray-300">
        Transform your videos into any language while preserving natural
        lip-sync, voice quality, and timing using advanced AI technology.
      </p>
    </div>
  );
}

export default HeroSection;
