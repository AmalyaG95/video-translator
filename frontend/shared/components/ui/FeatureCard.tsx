"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  color:
    | "blue"
    | "green"
    | "purple"
    | "red"
    | "yellow"
    | "indigo"
    | "pink"
    | "emerald";
}

const colorClasses = {
  blue: "text-blue-600 bg-blue-100 dark:bg-blue-900/20",
  green: "text-green-600 bg-green-100 dark:bg-green-900/20",
  purple: "text-purple-600 bg-purple-100 dark:bg-purple-900/20",
  red: "text-red-600 bg-red-100 dark:bg-red-900/20",
  yellow: "text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20",
  indigo: "text-indigo-600 bg-indigo-100 dark:bg-indigo-900/20",
  pink: "text-pink-600 bg-pink-100 dark:bg-pink-900/20",
  emerald: "text-emerald-600 bg-emerald-100 dark:bg-emerald-900/20",
};

export function FeatureCard({
  icon,
  title,
  description,
  color,
}: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02, y: -5 }}
      className="card-hover text-center"
    >
      <div className="flex flex-col items-center gap-2">
        <div
          className={`h-16 w-16 rounded-full ${colorClasses[color]} flex items-center justify-center`}
        >
          {icon}
        </div>

        <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
          {title}
        </h3>

        <p className="text-gray-600 dark:text-gray-400">{description}</p>
      </div>
    </motion.div>
  );
}
