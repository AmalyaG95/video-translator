"use client";

import Link from "next/link";
import { Home, ArrowLeft } from "lucide-react";

function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="w-full max-w-md text-center">
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-gray-300 dark:text-gray-600">
            404
          </h1>
          <h2 className="mt-4 text-2xl font-semibold text-gray-900 dark:text-white">
            Page Not Found
          </h2>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            The page you're looking for doesn't exist.
          </p>
        </div>

        <div className="space-y-4">
          <Link
            href="/"
            className="inline-flex items-center space-x-2 rounded-lg bg-blue-600 px-6 py-3 text-white transition-colors hover:bg-blue-700"
          >
            <Home className="h-5 w-5" />
            <span>Go Home</span>
          </Link>

          <button
            onClick={() => window.history.back()}
            className="ml-4 inline-flex items-center space-x-2 text-gray-600 transition-colors hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
          >
            <ArrowLeft className="h-5 w-5" />
            <span>Go Back</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default NotFound;
