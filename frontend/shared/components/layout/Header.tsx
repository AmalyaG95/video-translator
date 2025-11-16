"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Image from "next/image";
import {
  Menu,
  X,
  Sun,
  Moon,
  Monitor,
  Settings,
  Wifi,
  WifiOff,
  Sparkles,
} from "lucide-react";
import { useTranslationStore } from "@/stores/translationStore";

export function Header() {
  const { sidebarOpen, setSidebarOpen, theme, setTheme } =
    useTranslationStore();
  const [showThemeMenu, setShowThemeMenu] = useState(false);
  const isConnected = true; // Always connected for now

  const themes = [
    { value: "light", label: "Light", icon: Sun },
    { value: "dark", label: "Dark", icon: Moon },
    { value: "system", label: "System", icon: Monitor },
  ];

  const currentTheme = themes.find(t => t.value === theme) || themes[2];
  const ThemeIcon = currentTheme?.icon;

  return (
    <header className="fixed left-0 right-0 top-0 z-50 rounded-b-lg border-b border-gray-200 bg-white/80 backdrop-blur-md dark:border-gray-700 dark:bg-gray-900/80">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Left side */}
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="rounded-lg p-2 text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
            >
              {sidebarOpen ? (
                <X className="h-5 w-5" />
              ) : (
                <Menu className="h-5 w-5" />
              )}
            </button>

            {/* Logo */}
            <div className="flex items-center space-x-2">
              <div className="relative h-12 w-12 flex-shrink-0">
                <Image
                  src="/images/logo.png"
                  alt="Video Translator"
                  fill
                  className="object-contain"
                  priority
                />
              </div>
              <span className="hidden font-semibold text-gray-900 dark:text-white sm:inline-block">
                Video Translator
              </span>
            </div>
          </div>

          {/* Right side */}
          <div className="flex items-center space-x-4">
            {/* Theme selector */}
            <div className="relative">
              <button
                onClick={() => setShowThemeMenu(!showThemeMenu)}
                className="rounded-lg p-2 text-gray-700 transition-colors hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
              >
                {ThemeIcon && <ThemeIcon className="h-5 w-5" />}
              </button>

              {showThemeMenu && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute right-0 top-full z-50 w-48 rounded-lg border border-gray-200 bg-white py-2 shadow-lg dark:border-gray-700 dark:bg-gray-800"
                  style={{ marginTop: "8px" }}
                >
                  {themes.map(themeOption => {
                    const Icon = themeOption.icon;
                    return (
                      <button
                        key={themeOption.value}
                        onClick={() => {
                          setTheme(themeOption.value as any);
                          setShowThemeMenu(false);
                        }}
                        className={`flex w-full items-center space-x-3 px-4 py-2 text-left transition-colors hover:bg-gray-100 dark:hover:bg-gray-700 ${
                          theme === themeOption.value
                            ? "bg-blue-50 dark:bg-blue-900/20"
                            : ""
                        }`}
                      >
                        <Icon className="h-4 w-4 text-gray-700 dark:text-gray-300" />
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {themeOption.label}
                        </span>
                        {theme === themeOption.value && (
                          <div className="ml-auto h-2 w-2 rounded-full bg-blue-600 dark:bg-blue-400" />
                        )}
                      </button>
                    );
                  })}
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
