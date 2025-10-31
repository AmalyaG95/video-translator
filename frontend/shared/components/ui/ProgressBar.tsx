/**
 * ProgressBar component
 * Reusable progress bar with different variants and animations
 */

import { forwardRef } from "react";
import { clsx } from "clsx";

export type ProgressBarProps = React.HTMLAttributes<HTMLDivElement> & {
  value: number;
  max?: number;
  size?: "sm" | "md" | "lg";
  variant?: "default" | "success" | "warning" | "danger";
  showLabel?: boolean;
  label?: string;
  animated?: boolean;
  striped?: boolean;
};

const progressSizes = {
  sm: "h-1",
  md: "h-2",
  lg: "h-3",
};

const progressVariants = {
  default: "bg-blue-600",
  success: "bg-green-600",
  warning: "bg-yellow-600",
  danger: "bg-red-600",
};

export const ProgressBar = forwardRef<HTMLDivElement, ProgressBarProps>(
  (
    {
      value,
      max = 100,
      size = "md",
      variant = "default",
      showLabel = false,
      label,
      animated = false,
      striped = false,
      className,
      ...props
    },
    ref
  ) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
    const displayLabel = label ?? `${Math.round(percentage)}%`;

    return (
      <div className={clsx("w-full", className)} {...props}>
        {showLabel && (
          <div className="mb-2 flex justify-between text-sm">
            <span className="text-gray-700 dark:text-gray-300">
              {displayLabel}
            </span>
            <span className="text-gray-500 dark:text-gray-400">
              {value}/{max}
            </span>
          </div>
        )}

        <div
          ref={ref}
          className={clsx(
            "w-full rounded-full bg-gray-200 dark:bg-gray-700",
            progressSizes[size]
          )}
        >
          <div
            className={clsx(
              "h-full rounded-full transition-all duration-300 ease-out",
              progressVariants[variant],
              animated && "animate-pulse",
              striped && "bg-stripes"
            )}
            style={{
              width: `${percentage}%`,
            }}
            role="progressbar"
            aria-valuenow={value}
            aria-valuemin={0}
            aria-valuemax={max}
            aria-label={showLabel ? displayLabel : undefined}
          />
        </div>
      </div>
    );
  }
);

ProgressBar.displayName = "ProgressBar";

/**
 * Circular progress component
 */
export type CircularProgressProps = React.HTMLAttributes<HTMLDivElement> & {
  value: number;
  max?: number;
  size?: number;
  strokeWidth?: number;
  variant?: "default" | "success" | "warning" | "danger";
  showLabel?: boolean;
  label?: string;
};

export const CircularProgress = forwardRef<
  HTMLDivElement,
  CircularProgressProps
>(
  (
    {
      value,
      max = 100,
      size = 40,
      strokeWidth = 4,
      variant = "default",
      showLabel = false,
      label,
      className,
      ...props
    },
    ref
  ) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;
    const displayLabel = label ?? `${Math.round(percentage)}%`;

    const strokeColors = {
      default: "stroke-blue-600",
      success: "stroke-green-600",
      warning: "stroke-yellow-600",
      danger: "stroke-red-600",
    };

    return (
      <div
        ref={ref}
        className={clsx(
          "relative inline-flex items-center justify-center",
          className
        )}
        {...props}
      >
        <svg width={size} height={size} className="-rotate-90 transform">
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth={strokeWidth}
            fill="transparent"
            className="text-gray-200 dark:text-gray-700"
          />

          {/* Progress circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth={strokeWidth}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className={clsx(
              "transition-all duration-300 ease-out",
              strokeColors[variant]
            )}
          />
        </svg>

        {showLabel && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
              {displayLabel}
            </span>
          </div>
        )}
      </div>
    );
  }
);

CircularProgress.displayName = "CircularProgress";
