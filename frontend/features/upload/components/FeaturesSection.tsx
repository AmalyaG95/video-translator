"use client";

import { Globe, Mic, Zap } from "lucide-react";
import { FeatureCard } from "@/shared/components/ui";

const FEATURES = [
  {
    icon: Globe,
    title: "Multi-Language Support",
    description: "Translate to 10+ languages with native-quality voices",
    color: "blue" as const,
  },
  {
    icon: Mic,
    title: "Perfect Lip-Sync",
    description: "Advanced AI ensures natural lip-sync alignment",
    color: "green" as const,
  },
  {
    icon: Zap,
    title: "Lightning Fast",
    description: "Process videos up to 2x faster than real-time",
    color: "purple" as const,
  },
];

function FeaturesSection() {
  return (
    <div className="mx-auto grid max-w-6xl grid-cols-1 gap-8 md:grid-cols-3">
      {FEATURES.map((feature, index) => {
        const Icon = feature.icon;
        return (
          <FeatureCard
            key={index}
            icon={<Icon className="h-8 w-8" />}
            title={feature.title}
            description={feature.description}
            color={feature.color}
          />
        );
      })}
    </div>
  );
}

export default FeaturesSection;
