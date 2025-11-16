"use client";

import React, { useMemo } from "react";
import { FileVideo, Languages, Clock, Shield } from "lucide-react";
import { StatsCard } from "@/shared/components/ui";
import { SUPPORTED_LANGUAGES } from "@/constants";
import { useTranslationStore, selectAllSessions } from "@/stores/translationStore";

function StatsSection() {
  // Get all sessions from localStorage (persistent history)
  // Sessions are stored in localStorage as persistent data
  const sessions = useTranslationStore(selectAllSessions);
  
  // Calculate real statistics
  const stats = useMemo(() => {
    const completedSessions = sessions.filter(
      (s) => s.status === "completed"
    );
    const totalSessions = sessions.length;
    const videosProcessed = completedSessions.length;
    
    // Debug: Log session data to help troubleshoot
    if (completedSessions.length > 0) {
      console.log("[StatsSection] Completed sessions found:", completedSessions.length);
      console.log("[StatsSection] Sample session data:", {
        hasResult: !!completedSessions[0]?.result,
        processingTimeSeconds: completedSessions[0]?.result?.processingTimeSeconds,
        processingTime: completedSessions[0]?.result?.processingTime,
        startedAt: completedSessions[0]?.startedAt,
        completedAt: completedSessions[0]?.completedAt,
        createdAt: completedSessions[0]?.createdAt,
      });
    } else {
      console.log("[StatsSection] No completed sessions found. Total sessions:", totalSessions);
    }
    
    // Calculate success rate
    const successRate =
      totalSessions > 0
        ? Math.round((completedSessions.length / totalSessions) * 100)
        : 0;
    
    // Calculate average processing time
    const averageProcessingTime = (() => {
      if (completedSessions.length === 0) return "0 min";
      
      const totalTimeSeconds = completedSessions.reduce((acc: number, session: any) => {
        let sessionSeconds = 0;
        
        // Try processingTimeSeconds first (most accurate)
        if (session.result?.processingTimeSeconds !== undefined) {
          const seconds = typeof session.result.processingTimeSeconds === 'number' 
            ? session.result.processingTimeSeconds 
            : parseFloat(session.result.processingTimeSeconds);
          if (!isNaN(seconds) && seconds > 0) {
            sessionSeconds = seconds;
          }
        }
        
        // Try processingTime (could be number or string like "5m 30s")
        if (sessionSeconds === 0 && session.result?.processingTime !== undefined) {
          if (typeof session.result.processingTime === 'string') {
            // Parse string format like "5m 30s", "2h 15m", etc.
            const timeStr = session.result.processingTime;
            const patterns = [
              { regex: /(\d+)h\s*(\d+)m/, mult: [3600, 60] }, // hours and minutes
              { regex: /(\d+)m\s*(\d+)s/, mult: [60, 1] }, // minutes and seconds
              { regex: /(\d+)h\s*(\d+)m\s*(\d+)s/, mult: [3600, 60, 1] }, // hours, minutes, seconds
              { regex: /(\d+)h/, mult: [3600] }, // hours only
              { regex: /(\d+)m/, mult: [60] }, // minutes only
              { regex: /(\d+)s/, mult: [1] }, // seconds only
            ];

            for (const { regex, mult } of patterns) {
              const match = timeStr.match(regex);
              if (match) {
                let seconds = 0;
                for (let i = 0; i < mult.length; i++) {
                  const matchValue = match[i + 1];
                  const multiplier = mult[i];
                  if (matchValue !== undefined && multiplier !== undefined) {
                    seconds += parseInt(matchValue, 10) * multiplier;
                  }
                }
                sessionSeconds = seconds;
                break;
              }
            }
          } else {
            // It's a number
            const timeValue = typeof session.result.processingTime === 'number'
              ? session.result.processingTime
              : parseFloat(session.result.processingTime);
            
            if (!isNaN(timeValue)) {
              // If it's a large number (> 1000), assume it's milliseconds
              if (timeValue > 1000) {
                sessionSeconds = timeValue / 1000;
              } else {
                // Otherwise assume it's already in seconds
                sessionSeconds = timeValue;
              }
            }
          }
        }
        
        // Calculate from timestamps if still no time found
        if (sessionSeconds === 0) {
          const startTime = session.startedAt || session.createdAt;
          if (session.completedAt && startTime) {
            try {
              // Handle both Date objects and ISO strings
              const completedTime = session.completedAt instanceof Date 
                ? session.completedAt.getTime()
                : new Date(session.completedAt).getTime();
              
              const startTimeMs = startTime instanceof Date
                ? startTime.getTime()
                : new Date(startTime).getTime();
              
              const duration = (completedTime - startTimeMs) / 1000; // Convert to seconds
              if (duration > 0 && !isNaN(duration) && isFinite(duration)) {
                sessionSeconds = duration;
              }
            } catch (error) {
              console.warn("[StatsSection] Failed to parse timestamps:", error);
            }
          }
        }
        
        // Debug: Log if we still couldn't find processing time
        if (sessionSeconds === 0 && completedSessions.indexOf(session) < 3) {
          console.log("[StatsSection] Could not calculate processing time for session:", {
            sessionId: session.sessionId,
            hasResult: !!session.result,
            hasProcessingTimeSeconds: session.result?.processingTimeSeconds !== undefined,
            hasProcessingTime: session.result?.processingTime !== undefined,
            hasStartedAt: !!session.startedAt,
            hasCreatedAt: !!session.createdAt,
            hasCompletedAt: !!session.completedAt,
            hasDuration: !!session.duration,
            resultKeys: session.result ? Object.keys(session.result) : [],
          });
        }
        
        return acc + sessionSeconds;
      }, 0);
      
      if (totalTimeSeconds === 0) return "0 min";
      
      const avgSeconds = totalTimeSeconds / completedSessions.length;
      const avgMinutes = avgSeconds / 60;
      
      if (avgMinutes < 1) {
        return `${Math.round(avgSeconds)}s`;
      } else if (avgMinutes < 60) {
        return `${Math.round(avgMinutes)}m`;
      } else {
        const hours = Math.floor(avgMinutes / 60);
        const minutes = Math.round(avgMinutes % 60);
        return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
      }
    })();
    
    return {
      videosProcessed,
      successRate,
      averageProcessingTime,
    };
  }, [sessions]);
  
  return (
    <div className="mx-auto grid max-w-4xl grid-cols-1 gap-6 md:grid-cols-4">
      <StatsCard
        title="Videos Processed"
        value={stats.videosProcessed.toString()}
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
        value={stats.averageProcessingTime}
        icon={<Clock className="h-6 w-6" />}
        color="purple"
      />
      <StatsCard
        title="Success Rate"
        value={`${stats.successRate}%`}
        icon={<Shield className="h-6 w-6" />}
        color="emerald"
      />
    </div>
  );
}

export default StatsSection;
