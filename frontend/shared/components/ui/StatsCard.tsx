"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface StatsCardProps {
  title: string;
  value: string;
  icon: ReactNode;
  color:
    | "blue"
    | "green"
    | "purple"
    | "red"
    | "yellow"
    | "indigo"
    | "pink"
    | "emerald";
  trend?: {
    value: number;
    isPositive: boolean;
  };
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

export function StatsCard({
  title,
  value,
  icon: Icon,
  color,
  trend,
}: StatsCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      className="card-hover"
    >
      <div className="flex items-center justify-between">
        <div className="flex flex-1 flex-col gap-2">
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
            {title}
          </p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {value}
          </p>
          {trend && (
            <div className="flex items-center space-x-1">
              <span
                className={`text-sm font-medium ${
                  trend.isPositive ? "text-green-600" : "text-red-600"
                }`}
              >
                {trend.isPositive ? "+" : ""}
                {trend.value}%
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                vs last month
              </span>
            </div>
          )}
        </div>

        <div className={`rounded-lg p-3 ${colorClasses[color]}`}>{Icon}</div>
      </div>
    </motion.div>
  );
}
