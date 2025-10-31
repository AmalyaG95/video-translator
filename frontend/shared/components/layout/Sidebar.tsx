"use client";

import { motion } from "framer-motion";
import { usePathname, useRouter } from "next/navigation";
import Image from "next/image";
import {
  Upload,
  Play,
  History,
  Home,
  FileVideo,
  Clock,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { useTranslationStore } from "@/stores/translationStore";

export function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const { sidebarOpen, recentSessions, currentSession } = useTranslationStore();

  const menuItems = [
    { id: "/", label: "Upload", icon: Upload, path: "/" },
    { id: "/processing", label: "Processing", icon: Play, path: "/processing" },
    { id: "/history", label: "History", icon: History, path: "/history" },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "processing":
        return <Clock className="h-4 w-4 animate-spin text-blue-500" />;
      case "failed":
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <FileVideo className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <motion.aside
      initial={{ x: -300 }}
      animate={{ x: sidebarOpen ? 0 : -300 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="fixed left-0 top-0 z-40 h-full w-64 overflow-hidden rounded-r-lg border-r border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800"
    >
      <div className="flex h-full flex-col gap-8 px-6 pb-20 pt-24">
        {/* Logo */}

        {/* Navigation */}
        <nav className="flex flex-shrink-0 flex-col gap-2">
          {menuItems.map(item => {
            const Icon = item.icon;
            const isActive =
              pathname === item.path ||
              (item.path === "/processing" && pathname.startsWith("/results/"));

            return (
              <button
                key={item.id}
                onClick={() => router.push(item.path)}
                className={`flex w-full items-center space-x-3 rounded-lg px-4 py-3 text-left transition-colors ${
                  isActive
                    ? "bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400"
                    : "text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700"
                }`}
              >
                <Icon className="h-5 w-5" />
                <span className="font-medium">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Recent Sessions */}
        {recentSessions && recentSessions.length > 0 && (
          <div className="flex min-h-0 flex-1 flex-col gap-4">
            <h3 className="flex-shrink-0 text-sm font-semibold text-gray-900 dark:text-white">
              Recent Sessions
            </h3>

            <div className="flex flex-col gap-2 overflow-y-auto scrollbar-hide">
              {recentSessions
                .filter(
                  (session: any) =>
                    session.status !== "cancelled" &&
                    session.status !== "failed"
                )
                .map((session: any) => (
                  <div
                    key={session.sessionId}
                    className={`flex-shrink-0 cursor-pointer rounded-lg border p-3 transition-colors ${
                      currentSession?.sessionId === session.sessionId
                        ? "border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-900/20"
                        : "border-gray-200 hover:border-gray-300 dark:border-gray-700 dark:hover:border-gray-600"
                    }`}
                    onClick={() => {
                      useTranslationStore.getState().setCurrentSession(session);
                      if (session.status === "completed") {
                        router.push(`/results/${session.sessionId}`);
                      } else {
                        router.push("/processing");
                      }
                    }}
                  >
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(session.status)}
                          <span className="text-sm font-medium text-gray-900 dark:text-white">
                            {(session.sourceLang || "en").toUpperCase()} →{" "}
                            {(session.targetLang || "hy").toUpperCase()}
                          </span>
                        </div>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {Math.round(session.progress)}%
                        </span>
                      </div>

                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        {session.status === "completed"
                          ? "Completed"
                          : session.currentChunk !== undefined &&
                              session.totalChunks
                            ? `Processing ${session.currentChunk}/${session.totalChunks}`
                            : session.currentStep || "Processing..."}
                      </div>

                      <div className="text-xs text-gray-500 dark:text-gray-500">
                        {new Date(session.createdAt).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </motion.aside>
  );
}
