/**
 * Card component
 * Reusable card container with header, body, and footer
 */

import { forwardRef } from "react";
import { clsx } from "clsx";

export type CardProps = React.HTMLAttributes<HTMLDivElement> & {
  variant?: "default" | "elevated" | "outlined";
  padding?: "none" | "sm" | "md" | "lg";
  hover?: boolean;
};

export type CardHeaderProps = React.HTMLAttributes<HTMLDivElement> & {
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
};

export type CardBodyProps = React.HTMLAttributes<HTMLDivElement> & {
  padding?: "none" | "sm" | "md" | "lg";
};

export type CardFooterProps = React.HTMLAttributes<HTMLDivElement> & {
  justify?: "start" | "center" | "end" | "between";
};

const cardVariants = {
  default: "bg-white dark:bg-gray-800",
  elevated: "bg-white dark:bg-gray-800 shadow-lg",
  outlined:
    "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700",
};

const cardPadding = {
  none: "",
  sm: "p-3",
  md: "p-4",
  lg: "p-6",
};

export const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      variant = "default",
      padding = "md",
      hover = false,
      className,
      children,
      ...props
    },
    ref
  ) => {
    return (
      <div
        ref={ref}
        className={clsx(
          // Base styles
          "rounded-lg",

          // Variant styles
          cardVariants[variant],

          // Padding
          cardPadding[padding],

          // Hover effect
          hover && "transition-shadow hover:shadow-md",

          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);

Card.displayName = "Card";

export const CardHeader = forwardRef<HTMLDivElement, CardHeaderProps>(
  ({ title, subtitle, action, className, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={clsx("mb-4 flex items-start justify-between", className)}
        {...props}
      >
        <div className="flex-1">
          {title && (
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {title}
            </h3>
          )}
          {subtitle && (
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              {subtitle}
            </p>
          )}
          {children}
        </div>

        {action && <div className="ml-4 flex-shrink-0">{action}</div>}
      </div>
    );
  }
);

CardHeader.displayName = "CardHeader";

export const CardBody = forwardRef<HTMLDivElement, CardBodyProps>(
  ({ padding = "md", className, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={clsx(cardPadding[padding], className)}
        {...props}
      >
        {children}
      </div>
    );
  }
);

CardBody.displayName = "CardBody";

export const CardFooter = forwardRef<HTMLDivElement, CardFooterProps>(
  ({ justify = "end", className, children, ...props }, ref) => {
    const justifyClasses = {
      start: "justify-start",
      center: "justify-center",
      end: "justify-end",
      between: "justify-between",
    };

    return (
      <div
        ref={ref}
        className={clsx(
          "flex items-center border-t border-gray-200 pt-4 dark:border-gray-700",
          justifyClasses[justify],
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);

CardFooter.displayName = "CardFooter";
