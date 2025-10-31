# Upload Feature - Modular Architecture

This directory contains the upload feature with a clean, modular file structure.

## 📁 Directory Structure

```
upload/
├── components/           # UI Components
│   ├── index.ts         # Component exports
│   ├── HeroSection.tsx
│   ├── FeaturesSection.tsx
│   ├── LanguageSelection.tsx
│   ├── FileUpload.tsx
│   ├── AIChunkingStrategy.tsx
│   ├── TranslationReady.tsx
│   ├── StatsSection.tsx
│   └── ...
├── hooks/               # Custom Hooks
│   ├── index.ts         # Hook exports
│   ├── useLanguageDetection.ts
│   ├── useFileUpload.ts
│   └── useTranslation.ts
├── services/            # API Services
│   ├── index.ts         # Service exports
│   └── uploadService.ts
├── pages/               # Page Components
│   ├── index.ts         # Page exports
│   └── UploadPage.tsx
├── utils/               # Utility Functions
│   ├── index.ts         # Utility exports
│   ├── chunking.ts
│   └── statistics.ts
├── constants.ts         # Feature Constants
├── types.ts            # TypeScript Types
└── index.ts            # Main exports
```

## 🚀 Key Features

### **1. Modular Components**

- Each component is self-contained
- Clear separation of concerns
- Easy to test and maintain

### **2. Custom Hooks**

- Reusable logic extraction
- Clean component code
- Better testability

### **3. Shared State Management**

- Uses existing Zustand store (`useTranslationStore`)
- Centralized state management
- No duplicate state logic

### **4. Service Layer**

- API calls abstraction
- Easy to mock for testing
- Centralized error handling

### **5. Type Safety**

- Comprehensive TypeScript types
- Better developer experience
- Runtime safety

## 📖 Usage

### **Import Components**

```typescript
import { HeroSection, FileUpload, LanguageSelection } from "@/features/upload";
```

### **Use Hooks**

```typescript
import { useLanguageDetection, useFileUpload } from "@/features/upload";
```

### **Use Store**

```typescript
import { useTranslationStore } from "@/stores/translationStore";
```

### **Use Services**

```typescript
import { UploadService } from "@/features/upload";
```

## 🔧 Benefits

- **Better Organization**: Clear file structure
- **Easier Maintenance**: Modular components
- **Better Testing**: Isolated units
- **Type Safety**: Comprehensive TypeScript
- **Performance**: Optimized state management
- **Reusability**: Shared utilities and hooks

## 🎯 Best Practices

1. **Keep components small and focused**
2. **Use custom hooks for complex logic**
3. **Leverage the existing Zustand store for state**
4. **Use services for API calls**
5. **Export everything through index files**
6. **Maintain type safety throughout**
