"use client";

import { ReactNode, useEffect } from "react";
import { useTranslationStore } from "@/stores/translationStore";

interface ProvidersProps {
  children: ReactNode;
}

function ThemeProvider({ children }: { children: ReactNode }) {
  const theme = useTranslationStore(state => state.theme);

  useEffect(() => {
    const root = document.documentElement;

    // Remove both classes first
    root.classList.remove("light", "dark");

    if (theme === "system") {
      // Check system preference
      const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      if (isDark) {
        root.classList.add("dark");
      } else {
        root.classList.add("light");
      }
    } else {
      // Apply selected theme
      root.classList.add(theme);
    }
  }, [theme]);

  // Listen for system theme changes when theme is set to "system"
  useEffect(() => {
    if (theme !== "system") return;

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (e: MediaQueryListEvent) => {
      const root = document.documentElement;
      root.classList.remove("light", "dark");
      if (e.matches) {
        root.classList.add("dark");
      } else {
        root.classList.add("light");
      }
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [theme]);

  return <>{children}</>;
}

export function Providers({ children }: ProvidersProps) {
  return <ThemeProvider>{children}</ThemeProvider>;
}
