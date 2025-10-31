/**
 * PageContainer component
 * Main layout container with consistent spacing and structure
 */

import { forwardRef } from "react";
import { clsx } from "clsx";

export type PageContainerProps = React.HTMLAttributes<HTMLDivElement> & {
  size?: "sm" | "md" | "lg" | "xl" | "full";
  padding?: "none" | "sm" | "md" | "lg";
  maxWidth?: boolean;
  center?: boolean;
};

const containerSizes = {
  sm: "max-w-2xl",
  md: "max-w-4xl",
  lg: "max-w-6xl",
  xl: "max-w-7xl",
  full: "max-w-full",
};

const containerPadding = {
  none: "",
  sm: "px-4 py-6",
  md: "px-6 py-8",
  lg: "px-8 py-12",
};

export const PageContainer = forwardRef<HTMLDivElement, PageContainerProps>(
  (
    {
      size = "lg",
      padding = "md",
      maxWidth = true,
      center = true,
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
          "w-full",

          // Size constraints
          maxWidth && containerSizes[size],

          // Centering
          center && "mx-auto",

          // Padding
          containerPadding[padding],

          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);

PageContainer.displayName = "PageContainer";
