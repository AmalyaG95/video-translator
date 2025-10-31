// Upload feature components - Nested hierarchy based on component relationships

// Main page components (siblings)
export { default as HeroSection } from "./HeroSection";
export { default as FeaturesSection } from "./FeaturesSection";

// UploadScreen and its nested children
export { default as UploadScreen } from "./UploadScreen/UploadScreen";
export * from "./UploadScreen";

// Individual component exports for direct access
export { default as AIChunkingStrategy } from "./UploadScreen/AIChunkingStrategy/AIChunkingStrategy";
export { default as TranslationReady } from "./UploadScreen/TranslationReady/TranslationReady";
export { default as StatsSection } from "./UploadScreen/StatsSection/StatsSection";
export { default as LanguageDetectionLoading } from "./UploadScreen/LanguageDetectionLoading/LanguageDetectionLoading";

// Default export for main page
export { default } from "./UploadScreen/UploadScreen";
