"use client";

import React from "react";
import { FileVideo, Languages, Clock, Shield } from "lucide-react";
import { StatsCard } from "@/shared/components/ui";
import { SUPPORTED_LANGUAGES } from "@/constants";

function StatsSection() {
  return (
    <div className="mx-auto grid max-w-4xl grid-cols-1 gap-6 md:grid-cols-4">
      <StatsCard
        title="Videos Processed"
        value="0"
        icon={<FileVideo className="h-6 w-6" />}
        color="blue"
      />
      <StatsCard
        title="Languages Supported"
        value={SUPPORTED_LANGUAGES.length.toString()}
        icon={<Languages className="h-6 w-6" />}
        color="green"
      />
      <StatsCard
        title="Average Processing Time"
        value="2-5 min"
        icon={<Clock className="h-6 w-6" />}
        color="purple"
      />
      <StatsCard
        title="Success Rate"
        value="95%"
        icon={<Shield className="h-6 w-6" />}
        color="emerald"
      />
    </div>
  );
}

export default StatsSection;
