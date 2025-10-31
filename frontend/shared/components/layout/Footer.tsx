import { Heart } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="fixed bottom-0 left-0 right-0 z-40 rounded-t-lg border-t border-gray-200 bg-white/80 backdrop-blur-md dark:border-gray-700 dark:bg-gray-900/80">
      <div className="flex flex-col items-center justify-between gap-4 px-6 py-4 md:flex-row">
        <p className="flex items-center gap-1 text-sm text-gray-600 dark:text-gray-400">
          Made with <Heart className="h-4 w-4 fill-red-500 text-red-500" /> by
          Amalya Ghazaryan
        </p>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          © {currentYear} Video Translator. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
