"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Check } from "lucide-react";
import type { LanguageSelectorProps, Language } from "../../../../types";

function LanguageSelector({
  value,
  onChange,
  languages,
  disabled = false,
  disabledLanguages = [],
  label,
}: LanguageSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);

  const selectedLanguage = languages.find(lang => lang.code === value);

  return (
    <div className="relative">
      {label && (
        <label className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
          {label}
        </label>
      )}
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        aria-label={label || "Select language"}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        className={`flex w-full items-center justify-between rounded-lg border border-gray-300 bg-white px-4 py-3 transition-colors dark:border-gray-600 dark:bg-gray-700 ${
          disabled
            ? "cursor-not-allowed opacity-50"
            : "hover:border-blue-500 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 dark:hover:border-blue-400"
        }`}
      >
        <div className="flex items-center space-x-3">
          <span className="text-2xl" style={{ fontFamily: '"Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", "EmojiOne", "Twemoji Mozilla", system-ui, sans-serif' }}>
            {selectedLanguage?.flag || "🌐"}
          </span>
          <span className="font-medium text-gray-900 dark:text-white">
            {selectedLanguage?.name}
          </span>
        </div>
        <ChevronDown
          className={`h-5 w-5 text-gray-500 transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full z-50 max-h-60 w-full overflow-y-auto rounded-lg border border-gray-200 bg-white shadow-lg dark:border-gray-600 dark:bg-gray-700"
          >
            {languages.map(language => {
              const isDisabled = disabledLanguages.includes(language.code);
              return (
                <button
                  key={language.code}
                  onClick={() => {
                    if (!isDisabled) {
                      onChange?.(language.code);
                      setIsOpen(false);
                    }
                  }}
                  disabled={isDisabled}
                  aria-selected={language.code === value}
                  className={`flex w-full items-center justify-between px-4 py-3 transition-colors ${
                    isDisabled
                      ? "cursor-not-allowed text-gray-400 opacity-50"
                      : "hover:bg-gray-100 dark:hover:bg-gray-600"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl" style={{ fontFamily: '"Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", "EmojiOne", "Twemoji Mozilla", system-ui, sans-serif' }}>
                      {language.flag || "🌐"}
                    </span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {language.name}
                    </span>
                  </div>
                  {value === language.code && (
                    <Check className="h-5 w-5 text-blue-600" />
                  )}
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default LanguageSelector;
