"use client";

import { motion } from "framer-motion";
import { Header, Sidebar, Footer } from "@/shared/components/layout";
import { useTranslationStore } from "@/stores/translationStore";

export function LayoutContent({ children }: { children: React.ReactNode }) {
  const { sidebarOpen } = useTranslationStore();

  return (
    <div className="flex min-h-screen flex-col bg-gradient-to-br from-slate-50 to-blue-50 dark:from-gray-900 dark:to-slate-900">
      {/* Skip to main content link for accessibility */}
      <a
        href="#main-content"
        className="sr-only z-50 rounded-md bg-blue-600 px-4 py-2 text-white focus:not-sr-only focus:absolute focus:left-4 focus:top-4"
      >
        Skip to main content
      </a>

      <Header />

      <Sidebar />

      <motion.main
        id="main-content"
        className={`flex-1 p-6 pb-20 pt-24 transition-all duration-300 ease-in-out ${
          sidebarOpen ? "ml-64" : "ml-0"
        }`}
        tabIndex={-1}
      >
        {children}
      </motion.main>

      <Footer />
    </div>
  );
}
