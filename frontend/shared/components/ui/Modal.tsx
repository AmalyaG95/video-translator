"use client";

/**
 * Modal component
 * Reusable modal with backdrop, header, body, and footer
 */

import { forwardRef, useEffect } from "react";
import { createPortal } from "react-dom";
import { clsx } from "clsx";
import { X } from "lucide-react";

export type ModalProps = {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  description?: string;
  size?: "sm" | "md" | "lg" | "xl" | "full";
  closeOnBackdropClick?: boolean;
  closeOnEscape?: boolean;
  showCloseButton?: boolean;
  children: React.ReactNode;
};

export type ModalHeaderProps = {
  title?: string;
  description?: string;
  onClose?: () => void;
  showCloseButton?: boolean;
  children?: React.ReactNode;
};

export type ModalBodyProps = {
  children: React.ReactNode;
  className?: string;
};

export type ModalFooterProps = {
  children: React.ReactNode;
  className?: string;
};

const modalSizes = {
  sm: "max-w-md",
  md: "max-w-lg",
  lg: "max-w-2xl",
  xl: "max-w-4xl",
  full: "max-w-full mx-4",
};

export const Modal = forwardRef<HTMLDivElement, ModalProps>(
  (
    {
      isOpen,
      onClose,
      title,
      description,
      size = "md",
      closeOnBackdropClick = true,
      closeOnEscape = true,
      showCloseButton = true,
      children,
    },
    ref
  ) => {
    // Handle escape key
    useEffect(() => {
      if (!isOpen || !closeOnEscape) return;

      const handleEscape = (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          onClose();
        }
      };

      document.addEventListener("keydown", handleEscape);
      return () => document.removeEventListener("keydown", handleEscape);
    }, [isOpen, closeOnEscape, onClose]);

    // Prevent body scroll when modal is open
    useEffect(() => {
      if (isOpen) {
        document.body.style.overflow = "hidden";
      } else {
        document.body.style.overflow = "unset";
      }

      return () => {
        document.body.style.overflow = "unset";
      };
    }, [isOpen]);

    if (!isOpen) return null;

    const handleBackdropClick = (event: React.MouseEvent) => {
      if (closeOnBackdropClick && event.target === event.currentTarget) {
        onClose();
      }
    };

    return createPortal(
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        onClick={handleBackdropClick}
      >
        {/* Backdrop */}
        <div className="absolute inset-0 bg-black bg-opacity-50" />

        {/* Modal */}
        <div
          ref={ref}
          className={clsx(
            "relative w-full rounded-lg bg-white shadow-xl dark:bg-gray-800",
            modalSizes[size],
            "animate-in fade-in-0 zoom-in-95 duration-200"
          )}
          role="dialog"
          aria-modal="true"
          aria-labelledby={title ? "modal-title" : undefined}
          aria-describedby={description ? "modal-description" : undefined}
        >
          {title && (
            <ModalHeader
              title={title}
              description={description}
              onClose={onClose}
              showCloseButton={showCloseButton}
            />
          )}

          {children}
        </div>
      </div>,
      document.body
    );
  }
);

Modal.displayName = "Modal";

export const ModalHeader = forwardRef<HTMLDivElement, ModalHeaderProps>(
  ({ title, description, onClose, showCloseButton = true, children }, ref) => {
    return (
      <div
        ref={ref}
        className="flex items-start justify-between border-b border-gray-200 p-6 dark:border-gray-700"
      >
        <div className="flex-1">
          {title && (
            <h2
              id="modal-title"
              className="text-lg font-semibold text-gray-900 dark:text-white"
            >
              {title}
            </h2>
          )}
          {description && (
            <p
              id="modal-description"
              className="mt-1 text-sm text-gray-600 dark:text-gray-400"
            >
              {description}
            </p>
          )}
          {children}
        </div>

        {showCloseButton && onClose && (
          <button
            type="button"
            onClick={onClose}
            className="ml-4 flex-shrink-0 rounded-lg p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-700 dark:hover:text-gray-300"
            aria-label="Close modal"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>
    );
  }
);

ModalHeader.displayName = "ModalHeader";

export const ModalBody = forwardRef<HTMLDivElement, ModalBodyProps>(
  ({ children, className }, ref) => {
    return (
      <div ref={ref} className={clsx("p-6", className)}>
        {children}
      </div>
    );
  }
);

ModalBody.displayName = "ModalBody";

export const ModalFooter = forwardRef<HTMLDivElement, ModalFooterProps>(
  ({ children, className }, ref) => {
    return (
      <div
        ref={ref}
        className={clsx(
          "flex items-center justify-end gap-3 border-t border-gray-200 p-6 dark:border-gray-700",
          className
        )}
      >
        {children}
      </div>
    );
  }
);

ModalFooter.displayName = "ModalFooter";
