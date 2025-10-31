"use client";

/**
 * Tooltip component
 * Reusable tooltip with positioning and animations
 */

import { forwardRef, useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import { clsx } from "clsx";

export type TooltipProps = {
  content: React.ReactNode;
  children: React.ReactNode;
  placement?: "top" | "bottom" | "left" | "right";
  delay?: number;
  disabled?: boolean;
  className?: string;
  contentClassName?: string;
};

const tooltipPlacements = {
  top: "bottom-full left-1/2 transform -translate-x-1/2 mb-2",
  bottom: "top-full left-1/2 transform -translate-x-1/2 mt-2",
  left: "right-full top-1/2 transform -translate-y-1/2 mr-2",
  right: "left-full top-1/2 transform -translate-y-1/2 ml-2",
};

const tooltipArrows = {
  top: "top-full left-1/2 transform -translate-x-1/2 border-t-gray-900 dark:border-t-gray-700",
  bottom:
    "bottom-full left-1/2 transform -translate-x-1/2 border-b-gray-900 dark:border-b-gray-700",
  left: "left-full top-1/2 transform -translate-y-1/2 border-l-gray-900 dark:border-l-gray-700",
  right:
    "right-full top-1/2 transform -translate-y-1/2 border-r-gray-900 dark:border-r-gray-700",
};

export const Tooltip = forwardRef<HTMLDivElement, TooltipProps>(
  (
    {
      content,
      children,
      placement = "top",
      delay = 200,
      disabled = false,
      className,
      contentClassName,
    },
    ref
  ) => {
    const [isVisible, setIsVisible] = useState(false);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const triggerRef = useRef<HTMLDivElement>(null);
    const tooltipRef = useRef<HTMLDivElement>(null);
    const timeoutRef = useRef<NodeJS.Timeout>();

    const showTooltip = () => {
      if (disabled) return;

      timeoutRef.current = setTimeout(() => {
        setIsVisible(true);
        updatePosition();
      }, delay);
    };

    const hideTooltip = () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      setIsVisible(false);
    };

    const updatePosition = () => {
      if (!triggerRef.current || !tooltipRef.current) return;

      const triggerRect = triggerRef.current.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      const scrollX = window.pageXOffset || document.documentElement.scrollLeft;
      const scrollY = window.pageYOffset || document.documentElement.scrollTop;

      let x = 0;
      let y = 0;

      switch (placement) {
        case "top":
          x =
            triggerRect.left +
            triggerRect.width / 2 -
            tooltipRect.width / 2 +
            scrollX;
          y = triggerRect.top - tooltipRect.height - 8 + scrollY;
          break;
        case "bottom":
          x =
            triggerRect.left +
            triggerRect.width / 2 -
            tooltipRect.width / 2 +
            scrollX;
          y = triggerRect.bottom + 8 + scrollY;
          break;
        case "left":
          x = triggerRect.left - tooltipRect.width - 8 + scrollX;
          y =
            triggerRect.top +
            triggerRect.height / 2 -
            tooltipRect.height / 2 +
            scrollY;
          break;
        case "right":
          x = triggerRect.right + 8 + scrollX;
          y =
            triggerRect.top +
            triggerRect.height / 2 -
            tooltipRect.height / 2 +
            scrollY;
          break;
      }

      setPosition({ x, y });
    };

    useEffect(() => {
      if (isVisible) {
        updatePosition();

        const handleScroll = () => updatePosition();
        const handleResize = () => updatePosition();

        window.addEventListener("scroll", handleScroll);
        window.addEventListener("resize", handleResize);

        return () => {
          window.removeEventListener("scroll", handleScroll);
          window.removeEventListener("resize", handleResize);
        };
      }
    }, [isVisible, placement]);

    useEffect(() => {
      return () => {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
      };
    }, []);

    return (
      <>
        <div
          ref={triggerRef}
          className={clsx("inline-block", className)}
          onMouseEnter={showTooltip}
          onMouseLeave={hideTooltip}
          onFocus={showTooltip}
          onBlur={hideTooltip}
        >
          {children}
        </div>

        {isVisible &&
          createPortal(
            <div
              ref={tooltipRef}
              className={clsx(
                "absolute z-50 rounded bg-gray-900 px-2 py-1 text-xs text-white shadow-lg dark:bg-gray-700",
                "animate-in fade-in-0 zoom-in-95 duration-200",
                contentClassName
              )}
              style={{
                left: position.x,
                top: position.y,
              }}
            >
              {content}

              {/* Arrow */}
              <div
                className={clsx(
                  "absolute h-0 w-0 border-4 border-transparent",
                  tooltipArrows[placement]
                )}
              />
            </div>,
            document.body
          )}
      </>
    );
  }
);

Tooltip.displayName = "Tooltip";
