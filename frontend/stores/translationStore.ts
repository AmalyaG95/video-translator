import { create } from "zustand";
import {
  devtools,
  persist,
  subscribeWithSelector,
  createJSONStorage,
} from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type {
  ProcessingSession,
  VideoFile,
  VideoMetadata,
  HardwareInfo,
} from "@/shared/types";

// ============================================================================
// STATE INTERFACES
// ============================================================================
interface SessionState {
  currentSession: ProcessingSession | null;
  sessions: ProcessingSession[];
  recentSessions: ProcessingSession[];
}

interface UploadState {
  uploadedFile: VideoFile | null;
  metadata: VideoMetadata | null;
  isUploading: boolean;
  uploadProgress: number;
}

interface SettingsState {
  sourceLanguage: string;
  targetLanguage: string;
  qualityPreset: string;
  preserveTiming: boolean;
  lipSyncEnabled: boolean;
  sidebarOpen: boolean;
  theme: string;
}

interface UIState {
  isLoading: boolean;
  error: string | null;
  notifications: Notification[];
  activeTab: string;
  aiInsights: AIInsight[];
  processLogs: LogEntry[];
}

interface AIInsight {
  id: string;
  type: "decision" | "optimization" | "warning" | "success";
  title: string;
  description: string;
  impact: "high" | "medium" | "low";
  timestamp: string;
  sessionId?: string;
  data?: any;
}

interface LogEntry {
  id: string;
  timestamp: string;
  level: "info" | "warning" | "error" | "debug" | "success";
  stage: string;
  message: string;
  chunkId?: string;
  duration?: number;
  sessionId?: string;
  data?: any;
  details?: string;
}

interface Notification {
  id: string;
  type: "success" | "error" | "warning" | "info";
  message: string;
  timestamp: Date;
  duration?: number;
}

// ============================================================================
// ACTIONS INTERFACES
// ============================================================================
interface SessionActions {
  setCurrentSession: (session: ProcessingSession | null) => void;
  updateSessionProgress: (
    sessionId: string,
    updates: Partial<ProcessingSession>
  ) => void;
  addSession: (session: ProcessingSession) => void;
  removeSession: (sessionId: string) => void;
  clearSessions: () => void;
}

interface UploadActions {
  setUploadedFile: (file: VideoFile | null) => void;
  setMetadata: (metadata: VideoMetadata | null) => void;
  setUploading: (isUploading: boolean) => void;
  setUploadProgress: (progress: number) => void;
  clearUpload: () => void;
}

interface SettingsActions {
  setSourceLanguage: (language: string) => void;
  setTargetLanguage: (language: string) => void;
  setQualityPreset: (preset: string) => void;
  setPreserveTiming: (preserve: boolean) => void;
  setLipSyncEnabled: (enabled: boolean) => void;
  setSidebarOpen: (open: boolean) => void;
  setTheme: (theme: string) => void;
  resetSettings: () => void;
}

interface UIActions {
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  addNotification: (
    notification: Omit<Notification, "id" | "timestamp">
  ) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  setActiveTab: (tab: string) => void;
  setAIInsights: (insights: AIInsight[]) => void;
  addAIInsight: (insight: AIInsight) => void;
  clearAIInsights: () => void;
  clearAIInsightsForSession: (sessionId: string) => void;
  setProcessLogs: (logs: LogEntry[]) => void;
  addProcessLog: (log: LogEntry) => void;
  clearProcessLogs: () => void;
}

// ============================================================================
// MAIN STORE INTERFACE
// ============================================================================
interface TranslationStore
  extends SessionState,
    UploadState,
    SettingsState,
    UIState,
    SessionActions,
    UploadActions,
    SettingsActions,
    UIActions {
  reset: () => void;
}

// ============================================================================
// DEFAULT VALUES
// ============================================================================
const DEFAULT_SESSION_STATE: SessionState = {
  currentSession: null,
  sessions: [],
  recentSessions: [],
};

const DEFAULT_UPLOAD_STATE: UploadState = {
  uploadedFile: null,
  metadata: null,
  isUploading: false,
  uploadProgress: 0,
};

const DEFAULT_SETTINGS_STATE: SettingsState = {
  sourceLanguage: "en",
  targetLanguage: "hy", // Default to Armenian
  qualityPreset: "high",
  preserveTiming: true,
  lipSyncEnabled: true,
  sidebarOpen: false,
  theme: "light",
};

const DEFAULT_UI_STATE: UIState = {
  isLoading: false,
  error: null,
  notifications: [],
  activeTab: "overview",
  aiInsights: [],
  processLogs: [],
};

// ============================================================================
// PERSIST STORAGE (string-based, with merge & null fallback)
//  - Session-scoped data → sessionStorage under `${name}-session`
//  - Persistent data (settings, history, logs, insights) → localStorage under `${name}-persistent`
//  - getItem merges both; returns null if both missing/empty
//  - Uses createJSONStorage so persist handles (de)serialization cleanly
// ============================================================================
const STORE_NAME = "translation-store";

const dualJSONStorage = createJSONStorage(() => ({
  getItem: (name: string): string | null => {
    try {
      // Check if we're in browser environment
      if (
        typeof window === "undefined" ||
        typeof sessionStorage === "undefined" ||
        typeof localStorage === "undefined"
      ) {
        return null;
      }

      // Read from sessionStorage for session data, localStorage for persistent data
      const sessionRaw = sessionStorage.getItem(`${name}-session`);
      const localRaw = localStorage.getItem(`${name}-persistent`);

      let sessionObj: any = null;
      let localObj: any = null;

      if (sessionRaw) {
        try {
          const parsed = JSON.parse(sessionRaw);
          if (
            parsed &&
            typeof parsed === "object" &&
            Object.keys(parsed).length
          ) {
            sessionObj = parsed;
          }
        } catch (error) {
          console.error(
            `[Storage] Failed to parse session data for ${name}:`,
            error
          );
          sessionStorage.removeItem(`${name}-session`);
        }
      }

      if (localRaw) {
        try {
          const parsed = JSON.parse(localRaw);
          if (
            parsed &&
            typeof parsed === "object" &&
            Object.keys(parsed).length
          ) {
            localObj = parsed;
          }
        } catch (error) {
          console.error(
            `[Storage] Failed to parse persistent settings for ${name}:`,
            error
          );
          localStorage.removeItem(`${name}-persistent`);
        }
      }

      if (!sessionObj && !localObj) {
        return null;
      }

      const merged = { ...(localObj || {}), ...(sessionObj || {}) };
      if (!Object.keys(merged).length) {
        return null;
      }

      // Reconstruct Zustand's persist structure
      const persistStructure = {
        state: merged,
        version: 0,
      };

      return JSON.stringify(persistStructure);
    } catch (error) {
      console.error(`[Storage] Error in getItem for ${name}:`, error);
      return null;
    }
  },

  setItem: (name: string, value: string): void => {
    try {
      // Check if we're in browser environment
      if (
        typeof window === "undefined" ||
        typeof sessionStorage === "undefined" ||
        typeof localStorage === "undefined"
      ) {
        return;
      }

      const persistData = JSON.parse(value);
      // Storage logging removed for production - keep errors only

      // Extract the actual state from Zustand's persist structure
      const full = persistData.state || persistData;

      // persistent data (localStorage) - settings, history, logs, insights persist across browser sessions
      const persistentData = {
        // Settings
        theme: full.theme,
        sourceLanguage: full.sourceLanguage,
        targetLanguage: full.targetLanguage,
        qualityPreset: full.qualityPreset,
        preserveTiming: full.preserveTiming,
        lipSyncEnabled: full.lipSyncEnabled,
        sidebarOpen: full.sidebarOpen,
        // History data - NOW PERSISTENT
        sessions: full.sessions,
        recentSessions: full.recentSessions,
        // Insights - NOW PERSISTENT
        aiInsights: full.aiInsights,
        // Logs - NOW PERSISTENT (with size limit)
        processLogs: (full.processLogs || []).slice(0, 100),
      };

      // session-scoped data (sessionStorage) - current upload/processing state only
      const sessionData = {
        currentSession: full.currentSession,
        uploadedFile: full.uploadedFile,
        metadata: full.metadata,
        isUploading: full.isUploading,
        uploadProgress: full.uploadProgress,
        isLoading: full.isLoading,
        error: full.error,
        notifications: full.notifications,
        activeTab: full.activeTab,
      };

      // Store persistent data in localStorage, session data in sessionStorage
      localStorage.setItem(
        `${name}-persistent`,
        JSON.stringify(persistentData)
      );
      sessionStorage.setItem(`${name}-session`, JSON.stringify(sessionData));
    } catch (error) {
      console.error(`[Storage] Failed to store item ${name}:`, error);
      console.error(`[Storage] Value was:`, value);
    }
  },

  removeItem: (name: string): void => {
    // Check if we're in browser environment
    if (
      typeof window === "undefined" ||
      typeof sessionStorage === "undefined" ||
      typeof localStorage === "undefined"
    ) {
      return;
    }

    // Remove from localStorage (persistent data) and sessionStorage (session data)
    localStorage.removeItem(`${name}-persistent`);
    sessionStorage.removeItem(`${name}-session`);
  },
}));

// ============================================================================
// STORE IMPLEMENTATION
// ============================================================================
export const useTranslationStore = create<TranslationStore>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer<TranslationStore>((set, get) => ({
          // ----- STATE -----
          ...DEFAULT_SESSION_STATE,
          ...DEFAULT_UPLOAD_STATE,
          ...DEFAULT_SETTINGS_STATE,
          ...DEFAULT_UI_STATE,

          // ----- SESSION ACTIONS -----
          setCurrentSession: session =>
            set(state => {
              state.currentSession = session;
              if (session) {
                const exists = state.recentSessions.some(
                  s => s.sessionId === session.sessionId
                );
                if (!exists) {
                  state.recentSessions.unshift(session);
                  state.recentSessions = state.recentSessions.slice(0, 10);
                }
              }
            }),

          updateSessionProgress: (sessionId, updates) =>
            set(state => {
              // Preserve totalChunks if it exists and updates doesn't provide a valid value
              const preserveChunks = (existing: any, incoming: any) => {
                if (
                  incoming?.totalChunks !== undefined &&
                  incoming.totalChunks > 0
                ) {
                  return incoming.totalChunks;
                }
                if (
                  existing?.totalChunks !== undefined &&
                  existing.totalChunks > 0
                ) {
                  return existing.totalChunks;
                }
                return incoming?.totalChunks ?? existing?.totalChunks;
              };

              // Handle logs: merge new logs with existing, avoiding duplicates
              const mergeLogs = (existing: any[] | undefined, incoming: any[] | undefined) => {
                if (!incoming || incoming.length === 0) {
                  return existing || [];
                }
                
                const logMap = new Map<string, any>();
                
                // Add existing logs
                (existing || []).forEach(log => {
                  const key = log.id || `${log.timestamp}-${log.message}`;
                  logMap.set(key, log);
                });
                
                // Add new logs (override if same id)
                incoming.forEach(log => {
                  const key = log.id || `${log.timestamp}-${log.message}`;
                  logMap.set(key, log);
                });
                
                // Convert to array, sort by timestamp, limit to 200
                return Array.from(logMap.values())
                  .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
                  .slice(-200);
              };

              if (state.currentSession?.sessionId === sessionId) {
                const preservedTotalChunks = preserveChunks(
                  state.currentSession,
                  updates
                );
                // Merge logs if provided
                if (updates.logs) {
                  updates.logs = mergeLogs(state.currentSession.logs, updates.logs);
                }
                Object.assign(state.currentSession, updates);
                if (preservedTotalChunks !== undefined) {
                  state.currentSession.totalChunks = preservedTotalChunks;
                }
              }
              const sIdx = state.sessions.findIndex(
                s => s.sessionId === sessionId
              );
              if (sIdx !== -1) {
                const preservedTotalChunks = preserveChunks(
                  state.sessions[sIdx]!,
                  updates
                );
                // Merge logs if provided
                if (updates.logs) {
                  updates.logs = mergeLogs(state.sessions[sIdx]!.logs, updates.logs);
                }
                Object.assign(state.sessions[sIdx]!, updates);
                if (preservedTotalChunks !== undefined) {
                  state.sessions[sIdx]!.totalChunks = preservedTotalChunks;
                }
              }

              const rIdx = state.recentSessions.findIndex(
                s => s.sessionId === sessionId
              );
              if (rIdx !== -1) {
                const preservedTotalChunks = preserveChunks(
                  state.recentSessions[rIdx]!,
                  updates
                );
                // Merge logs if provided
                if (updates.logs) {
                  updates.logs = mergeLogs(state.recentSessions[rIdx]!.logs, updates.logs);
                }
                Object.assign(state.recentSessions[rIdx]!, updates);
                if (preservedTotalChunks !== undefined) {
                  state.recentSessions[rIdx]!.totalChunks =
                    preservedTotalChunks;
                }
              }
            }),

          addSession: session =>
            set(state => {
              const exists = state.sessions.findIndex(
                s => s.sessionId === session.sessionId
              );
              if (exists === -1) state.sessions.unshift(session);
              else state.sessions[exists] = session;

              const rIdx = state.recentSessions.findIndex(
                s => s.sessionId === session.sessionId
              );
              if (rIdx === -1) {
                state.recentSessions.unshift(session);
                state.recentSessions = state.recentSessions.slice(0, 10);
              } else {
                state.recentSessions[rIdx] = session;
              }
            }),

          removeSession: sessionId =>
            set(state => {
              state.sessions = state.sessions.filter(
                s => s.sessionId !== sessionId
              );
              state.recentSessions = state.recentSessions.filter(
                s => s.sessionId !== sessionId
              );
              if (state.currentSession?.sessionId === sessionId) {
                state.currentSession = null;
              }
            }),

          clearSessions: () =>
            set(state => {
              state.sessions = [];
              state.recentSessions = [];
              state.currentSession = null;
            }),

          // ----- UPLOAD ACTIONS -----
          setUploadedFile: file =>
            set(state => {
              state.uploadedFile = file;
            }),
          setMetadata: metadata =>
            set(state => {
              state.metadata = metadata;
            }),
          setUploading: isUploading =>
            set(state => {
              state.isUploading = isUploading;
            }),
          setUploadProgress: progress =>
            set(state => {
              state.uploadProgress = progress;
            }),
          clearUpload: () =>
            set(state => {
              state.uploadedFile = null;
              state.metadata = null;
              state.isUploading = false;
              state.uploadProgress = 0;
            }),

          // ----- SETTINGS ACTIONS -----
          setSourceLanguage: language =>
            set(state => {
              state.sourceLanguage = language;
            }),
          setTargetLanguage: language =>
            set(state => {
              state.targetLanguage = language;
            }),
          setQualityPreset: preset =>
            set(state => {
              state.qualityPreset = preset;
            }),
          setPreserveTiming: preserve =>
            set(state => {
              state.preserveTiming = preserve;
            }),
          setLipSyncEnabled: enabled =>
            set(state => {
              state.lipSyncEnabled = enabled;
            }),
          setSidebarOpen: open =>
            set(state => {
              state.sidebarOpen = open;
            }),
          setTheme: theme =>
            set(state => {
              state.theme = theme;
            }),
          resetSettings: () =>
            set(state => {
              Object.assign(state, DEFAULT_SETTINGS_STATE);
            }),

          // ----- UI ACTIONS -----
          setLoading: loading =>
            set(state => {
              state.isLoading = loading;
            }),
          setError: error =>
            set(state => {
              state.error = error;
            }),
          addNotification: notification =>
            set(state => {
              const newNotification: Notification = {
                ...notification,
                id: Math.random().toString(36).slice(2, 11),
                timestamp: new Date(),
              };
              state.notifications.push(newNotification);

              if (notification.duration) {
                setTimeout(() => {
                  get().removeNotification(newNotification.id);
                }, notification.duration);
              }
            }),
          removeNotification: id =>
            set(state => {
              state.notifications = state.notifications.filter(
                n => n.id !== id
              );
            }),
          clearNotifications: () =>
            set(state => {
              state.notifications = [];
            }),
          setActiveTab: tab =>
            set(state => {
              state.activeTab = tab;
            }),

          setAIInsights: insights =>
            set(state => {
              state.aiInsights = insights;
            }),
          addAIInsight: insight =>
            set(state => {
              state.aiInsights.push(insight);
            }),
          clearAIInsights: () =>
            set(state => {
              state.aiInsights = [];
            }),
          clearAIInsightsForSession: (sessionId: string) =>
            set(state => {
              state.aiInsights = state.aiInsights.filter(
                i => i.sessionId !== sessionId
              );
            }),

          setProcessLogs: logs =>
            set(state => ({
              ...state,
              processLogs: logs,
            })),
          addProcessLog: log =>
            set(state => {
              const currentLogs = state.processLogs;
              // Generate ID if not present
              const logWithId = log.id
                ? log
                : { ...log, id: `${log.timestamp}-${Math.random()}` };

              // Check for duplicates based on id first, then timestamp + message
              const isDuplicate = currentLogs.some(
                existingLog =>
                  existingLog.id === logWithId.id ||
                  (existingLog.timestamp === logWithId.timestamp &&
                    existingLog.message === logWithId.message &&
                    Math.abs(
                      new Date(existingLog.timestamp).getTime() -
                        new Date(logWithId.timestamp).getTime()
                    ) < 100)
              );

              if (!isDuplicate) {
                // Create a new array to avoid mutating state
                const updatedLogs = [logWithId, ...currentLogs].slice(0, 50);
                return { ...state, processLogs: updatedLogs };
              }

              return state; // Return state unchanged if duplicate
            }),
          clearProcessLogs: () =>
            set(state => ({
              ...state,
              processLogs: [],
            })),

          // ----- RESET ALL -----
          reset: () =>
            set(state => {
              Object.assign(state, {
                ...DEFAULT_SESSION_STATE,
                ...DEFAULT_UPLOAD_STATE,
                ...DEFAULT_SETTINGS_STATE,
                ...DEFAULT_UI_STATE,
              });
            }),
        }))
      ),
      {
        name: STORE_NAME,
        storage: dualJSONStorage, // ✅ uses our split storage with merge & null fallback
        // keep everything; splitting handled inside storage.setItem
        partialize: state => ({ ...state }),
      }
    )
  )
);

// ============================================================================
// SELECTORS
// ============================================================================
export const selectCurrentSession = (s: TranslationStore) => s.currentSession;

export const selectAllSessions = (s: TranslationStore) => s.sessions;
export const selectRecentSessions = (s: TranslationStore) => s.recentSessions;
export const selectSessionsByStatus =
  (status: ProcessingSession["status"]) => (s: TranslationStore) =>
    s.sessions.filter(x => x.status === status);

export const selectCompletedSessions = (s: TranslationStore) =>
  s.sessions.filter(x => x.status === "completed");
export const selectProcessingSessions = (s: TranslationStore) =>
  s.sessions.filter(x => x.status === "processing");
export const selectFailedSessions = (s: TranslationStore) =>
  s.sessions.filter(x => x.status === "failed");

export const selectCurrentSessionStatus = (s: TranslationStore) =>
  s.currentSession?.status ?? null;
export const selectCurrentSessionProgress = (s: TranslationStore) =>
  s.currentSession?.progress ?? 0;
export const selectIsProcessing = (s: TranslationStore) =>
  s.currentSession?.status === "processing";
export const selectIsPaused = (s: TranslationStore) =>
  s.currentSession?.isPaused ?? false;

export const selectUploadedFile = (s: TranslationStore) => s.uploadedFile;
export const selectMetadata = (s: TranslationStore) => s.metadata;
export const selectUploadProgress = (s: TranslationStore) => s.uploadProgress;
export const selectIsUploading = (s: TranslationStore) => s.isUploading;

export const selectLanguageSettings = (s: TranslationStore) => ({
  sourceLanguage: s.sourceLanguage,
  targetLanguage: s.targetLanguage,
});
export const selectQualitySettings = (s: TranslationStore) => ({
  qualityPreset: s.qualityPreset,
  preserveTiming: s.preserveTiming,
  lipSyncEnabled: s.lipSyncEnabled,
});

export const selectUIState = (s: TranslationStore) => ({
  isLoading: s.isLoading,
  error: s.error,
  sidebarOpen: s.sidebarOpen,
});

export const selectNotifications = (s: TranslationStore) => s.notifications;
export const selectError = (s: TranslationStore) => s.error;
export const selectLoading = (s: TranslationStore) => s.isLoading;

// ============================================================================
// COMPUTED SELECTORS
// ============================================================================
export const selectSessionStats = (s: TranslationStore) => {
  const sessions = s.sessions;
  return {
    total: sessions.length,
    completed: sessions.filter(x => x.status === "completed").length,
    processing: sessions.filter(x => x.status === "processing").length,
    failed: sessions.filter(x => x.status === "failed").length,
    paused: sessions.filter(x => x.status === "paused").length,
  };
};

export const selectCurrentSessionHardware = (s: TranslationStore) =>
  (s.currentSession?.hardwareInfo as HardwareInfo | undefined) ?? null;
export const selectCurrentSessionETA = (s: TranslationStore) =>
  s.currentSession?.etaSeconds ?? null;
export const selectCurrentSessionSpeed = (s: TranslationStore) =>
  s.currentSession?.processingSpeed ?? null;

export const selectCurrentSessionChunks = (s: TranslationStore) => ({
  current: s.currentSession?.currentChunk ?? 0,
  total: s.currentSession?.totalChunks ?? 0,
  percentage: s.currentSession?.totalChunks
    ? Math.round(
        ((s.currentSession?.currentChunk ?? 0) /
          (s.currentSession?.totalChunks ?? 1)) *
          100
      )
    : 0,
});
