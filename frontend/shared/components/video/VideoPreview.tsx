"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Play, Pause, Volume2, VolumeX, Maximize } from "lucide-react";

// Extended TextTrack interface with browser-specific properties
interface TextTrackWithState extends TextTrack {
  readyState?: number;
  error?: Error | null;
}

// Fullscreen API vendor-prefixed types
interface FullscreenElement extends HTMLElement {
  webkitRequestFullscreen?: () => Promise<void>;
  msRequestFullscreen?: () => Promise<void>;
  mozRequestFullScreen?: () => Promise<void>;
}

interface FullscreenDocument extends Document {
  webkitExitFullscreen?: () => Promise<void>;
  msExitFullscreen?: () => Promise<void>;
  mozCancelFullScreen?: () => Promise<void>;
}

interface SubtitleCue {
  startTime: number;
  endTime: number;
  text: string;
}

interface VideoPreviewProps {
  src: string;
  title?: string;
  className?: string;
  startTime?: number;
  duration?: number;
  onTimeUpdate?: (time: number) => void;
  onDurationChange?: (duration: number) => void;
  onPlay?: () => void;
  onPause?: () => void;
  subtitleUrl?: string | null;
  subtitleLabel?: string;
  embeddedSubtitleUrl?: string | null; // SRT file URL for tracking embedded subtitles
}

export function VideoPreview({
  src,
  title = "Video Preview",
  className = "w-full h-full object-contain",
  startTime = 0,
  duration,
  onTimeUpdate,
  onDurationChange,
  onPlay,
  onPause,
  subtitleUrl,
  subtitleLabel = "Subtitles",
  embeddedSubtitleUrl,
}: VideoPreviewProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [videoAspectRatio, setVideoAspectRatio] = useState<number | null>(null);
  const [subtitleTrackReady, setSubtitleTrackReady] = useState(false);
  const [subtitleError, setSubtitleError] = useState<string | null>(null);
  const [subtitleUrlValid, setSubtitleUrlValid] = useState<boolean | null>(
    null
  );
  const [embeddedSubtitles, setEmbeddedSubtitles] = useState<SubtitleCue[]>([]);
  const [readSubtitleEndTimes, setReadSubtitleEndTimes] = useState<Set<number>>(new Set());
  const [showSubtitleOverlay, setShowSubtitleOverlay] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
        onPause?.();
      } else {
        videoRef.current.play();
        onPlay?.();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const time = videoRef.current.currentTime;
      const videoDuration = videoRef.current.duration;

      // Clear error if video is playing successfully
      if (error) {
        setError(null);
      }

      // Calculate the end time based on startTime + duration prop
      const endTime = startTime + (duration || videoDuration);

      // If we have a duration prop, check if we've reached the end
      if (duration && time >= endTime) {
        console.log("🎬 Reached end of segment, pausing video");
        videoRef.current.pause();
        setCurrentTime(endTime);
        onTimeUpdate?.(endTime);
        return;
      }

      // Ensure currentTime doesn't exceed video duration
      const validTime = isNaN(videoDuration)
        ? time
        : Math.min(time, videoDuration);

      // If current time exceeds video duration, pause the video and reset to duration
      if (!isNaN(videoDuration) && time > videoDuration && videoDuration > 0) {
        console.warn("⚠️ Current time exceeds video duration, pausing video");
        videoRef.current.pause();
        videoRef.current.currentTime = videoDuration;
        setCurrentTime(videoDuration);
        onTimeUpdate?.(videoDuration);
        return;
      }

      setCurrentTime(validTime);
      onTimeUpdate?.(validTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      const videoDuration = videoRef.current.duration;
      const video = videoRef.current;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      // Calculate and set aspect ratio
      if (videoWidth > 0 && videoHeight > 0) {
        const aspectRatio = videoWidth / videoHeight;
        setVideoAspectRatio(aspectRatio);
      }

      // Clear any previous errors when video loads successfully
      setError(null);

      // Only set duration if it's valid and greater than 0
      if (!isNaN(videoDuration) && videoDuration > 0) {
        setVideoDuration(videoDuration);
        onDurationChange?.(videoDuration);

        // Seek to startTime if provided
        if (startTime > 0) {
          video.currentTime = startTime;
          setCurrentTime(startTime);
        }
      } else {
        console.warn("⚠️ Invalid video duration detected:", videoDuration);
        // Try to reload the video or wait for more metadata
        setTimeout(() => {
          if (videoRef.current && videoRef.current.duration !== videoDuration) {
            const newDuration = videoRef.current.duration;
            console.log("🔄 Retrying duration after delay:", newDuration);
            if (!isNaN(newDuration) && newDuration > 0) {
              setVideoDuration(newDuration);
              onDurationChange?.(newDuration);

              // Seek to startTime if provided
              if (startTime > 0) {
                videoRef.current.currentTime = startTime;
                setCurrentTime(startTime);
              }
            }
          }
        }, 1000);
      }
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (videoRef.current) {
      const time = parseFloat(e.target.value);
      videoRef.current.currentTime = time;
      setCurrentTime(time);
      onTimeUpdate?.(time);
    }
  };

  const formatTime = (time: number) => {
    // Ensure time is not negative and is a valid number
    const validTime = Math.max(0, isNaN(time) ? 0 : time);
    const minutes = Math.floor(validTime / 60);
    const seconds = Math.floor(validTime % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const handleError = (e: React.SyntheticEvent<HTMLVideoElement, Event>) => {
    const video = e.currentTarget;
    let errorMessage = "Failed to load video. Please try again.";

    if (video.error) {
      switch (video.error.code) {
        case MediaError.MEDIA_ERR_ABORTED:
          errorMessage = "Video loading was aborted.";
          break;
        case MediaError.MEDIA_ERR_NETWORK:
          errorMessage = "Network error while loading video.";
          break;
        case MediaError.MEDIA_ERR_DECODE:
          errorMessage = "Video decoding error.";
          break;
        case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
          errorMessage = "Video format not supported or source not available.";
          break;
        default:
          errorMessage = `Video error: ${video.error.message || "Unknown error"}`;
      }
      console.error("Video error:", {
        code: video.error.code,
        message: video.error.message,
        src: src,
      });
    } else {
      // If no error code, check if src is empty or invalid
      if (!src || src.trim() === "") {
        errorMessage = "Video source not available.";
      }
    }

    setError(errorMessage);
  };

  // Parse SRT file format
  const parseSRT = (srtContent: string): SubtitleCue[] => {
    const cues: SubtitleCue[] = [];
    const blocks = srtContent.trim().split(/\n\s*\n/);
    
    for (const block of blocks) {
      const lines = block.trim().split('\n');
      if (lines.length < 3) continue;
      
      // Skip index line (first line)
      const timeLine = lines[1];
      if (!timeLine) continue; // Skip if timeLine is missing
      
      const text = lines.slice(2).join(' ').replace(/<[^>]+>/g, ''); // Remove HTML tags
      
      // Parse time format: 00:00:00,000 --> 00:00:00,000
      const timeMatch = timeLine.match(/(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})/);
      if (timeMatch && timeMatch.length >= 9) {
        // Ensure all match groups exist before parsing
        const startHours = timeMatch[1];
        const startMinutes = timeMatch[2];
        const startSeconds = timeMatch[3];
        const startMillis = timeMatch[4];
        const endHours = timeMatch[5];
        const endMinutes = timeMatch[6];
        const endSeconds = timeMatch[7];
        const endMillis = timeMatch[8];
        
        if (startHours && startMinutes && startSeconds && startMillis &&
            endHours && endMinutes && endSeconds && endMillis) {
        const startTime = 
            parseInt(startHours) * 3600 +
            parseInt(startMinutes) * 60 +
            parseInt(startSeconds) +
            parseInt(startMillis) / 1000;
        const endTime = 
            parseInt(endHours) * 3600 +
            parseInt(endMinutes) * 60 +
            parseInt(endSeconds) +
            parseInt(endMillis) / 1000;
        
        cues.push({ startTime, endTime, text });
        }
      }
    }
    
    return cues;
  };

  // Load and parse embedded subtitles
  useEffect(() => {
    if (embeddedSubtitleUrl) {
      fetch(embeddedSubtitleUrl)
        .then(response => response.text())
        .then(content => {
          const parsed = parseSRT(content);
          setEmbeddedSubtitles(parsed);
          console.log(`📝 Loaded ${parsed.length} embedded subtitle cues`);
        })
        .catch(error => {
          console.error('Failed to load embedded subtitles:', error);
          setEmbeddedSubtitles([]);
        });
    } else {
      setEmbeddedSubtitles([]);
      setReadSubtitleEndTimes(new Set());
    }
  }, [embeddedSubtitleUrl]);

  // Track which subtitles have been read and update overlay visibility
  useEffect(() => {
    if (!videoRef.current || embeddedSubtitles.length === 0) {
      setShowSubtitleOverlay(false);
      return;
    }
    
    const video = videoRef.current;
    const updateReadSubtitles = () => {
      const currentTime = video.currentTime;
      const newReadTimes = new Set(readSubtitleEndTimes);
      
      embeddedSubtitles.forEach(cue => {
        // Mark as read if we've passed the end time
        if (currentTime > cue.endTime) {
          newReadTimes.add(cue.endTime);
        }
      });
      
      // Check if we're currently showing a subtitle (active)
      const activeSubtitle = embeddedSubtitles.find(
        cue => currentTime >= cue.startTime && currentTime <= cue.endTime
      );
      // Check if current time is past any subtitle's end time (read)
      const hasReadSubtitles = embeddedSubtitles.some(
        cue => currentTime > cue.endTime
      );
      // Show overlay when we have read subtitles but no active subtitle
      const shouldShow = hasReadSubtitles && !activeSubtitle;
      
      setShowSubtitleOverlay(shouldShow);
      
      if (newReadTimes.size !== readSubtitleEndTimes.size) {
        setReadSubtitleEndTimes(newReadTimes);
      }
    };
    
    video.addEventListener('timeupdate', updateReadSubtitles);
    return () => video.removeEventListener('timeupdate', updateReadSubtitles);
  }, [embeddedSubtitles, readSubtitleEndTimes]);

  // Clear error when src changes
  useEffect(() => {
    setError(null);
  }, [src]);

  // Update currentTime when startTime changes (when switching spots)
  useEffect(() => {
    if (startTime > 0 && videoRef.current) {
      videoRef.current.currentTime = startTime;
      setCurrentTime(startTime);
    }
  }, [startTime]);

  // Reset subtitle state when subtitleUrl changes and verify URL is accessible
  useEffect(() => {
    setSubtitleTrackReady(false);
    setSubtitleError(null);
    setSubtitleUrlValid(null);
    console.log("📝 Subtitle URL changed:", subtitleUrl);

    // Verify subtitle URL is accessible before creating track element
    // Only process if subtitleUrl is a non-empty string
    if (
      subtitleUrl &&
      typeof subtitleUrl === "string" &&
      subtitleUrl.trim().length > 0
    ) {
      fetch(subtitleUrl)
        .then(response => {
          if (response.ok) {
            console.log("✅ Subtitle URL is accessible:", subtitleUrl);
            console.log("Content-Type:", response.headers.get("content-type"));
            setSubtitleUrlValid(true);
            return response.text();
          } else {
            console.warn(
              "⚠️ Subtitle URL returned status:",
              response.status,
              subtitleUrl
            );
            setSubtitleUrlValid(false);
            setSubtitleError(`Subtitle file not found (${response.status})`);
            return null;
          }
        })
        .then(content => {
          if (content) {
            console.log(
              "📄 Subtitle content preview (first 500 chars):",
              content.substring(0, 500)
            );
            const lineCount = content.split("\n").length;
            console.log("📊 Subtitle file has", lineCount, "lines");
            if (content.trim().length === 0) {
              console.error("❌ Subtitle file is empty!");
              setSubtitleUrlValid(false);
              setSubtitleError("Subtitle file is empty");
            } else if (!content.includes("WEBVTT")) {
              console.warn(
                "⚠️ Subtitle content may not be valid WebVTT format"
              );
            }
          }
        })
        .catch(error => {
          console.error("❌ Failed to fetch subtitle URL:", error);
          setSubtitleUrlValid(false);
          setSubtitleError("Failed to load subtitle file");
        });
    } else {
      setSubtitleUrlValid(false);
    }
  }, [subtitleUrl]);

  // Enable subtitles when track loads (only if URL is valid)
  useEffect(() => {
    if (videoRef.current && subtitleUrl && subtitleUrlValid === true) {
      const video = videoRef.current;
      console.log("🎬 Setting up subtitle track for URL:", subtitleUrl);

      const enableSubtitles = () => {
        try {
          const tracks = video.textTracks;
          console.log(
            "🔍 Checking text tracks:",
            tracks.length,
            "tracks available"
          );

          if (tracks && tracks.length > 0) {
            // Find the subtitle track and enable it
            for (let i = 0; i < tracks.length; i++) {
              const track = tracks[i];
              if (!track) continue;

              try {
                const trackWithState = track as TextTrackWithState;

                console.log(`Track ${i}:`, {
                  kind: track.kind,
                  label: track.label,
                  language: track.language,
                  mode: track.mode,
                  readyState: trackWithState.readyState,
                  cues: track.cues?.length || 0,
                });

                if (track.kind === "subtitles" || track.kind === "captions") {
                  // Check if cues are actually loaded
                  const cueCount = track.cues?.length || 0;
                  const activeCues = track.activeCues?.length || 0;

                  console.log(`📊 Track ${i} cue status:`, {
                    totalCues: cueCount,
                    activeCues: activeCues,
                    readyState: trackWithState.readyState,
                    mode: track.mode,
                  });

                  // Enable track and verify cues exist
                  if (track.mode !== "showing") {
                    track.mode = "showing";
                    console.log("✅ Subtitles mode set to showing");
                  }

                  // Check for error conditions first
                  if (
                    trackWithState.readyState === undefined ||
                    trackWithState.readyState === null
                  ) {
                    console.warn(
                      "⚠️ Track readyState is undefined - subtitle file likely failed to load"
                    );
                    setSubtitleError("Subtitle track failed to load");
                    setSubtitleTrackReady(false);
                    return true; // Found track but it's in error state
                  }

                  // Only mark as ready if we have cues or track is fully loaded
                  if (cueCount > 0 || trackWithState.readyState === 2) {
                    setSubtitleTrackReady(true);
                    setSubtitleError(null);
                    console.log("✅ Subtitles ready with", cueCount, "cues");
                  } else if (trackWithState.readyState === 1) {
                    console.log("⏳ Waiting for cues to load (loading state)");
                  } else if (trackWithState.readyState === 0) {
                    console.warn(
                      "⚠️ Track readyState is NONE (0) - track not yet loaded"
                    );
                  } else {
                    console.warn(
                      "⚠️ Track has no cues yet. ReadyState:",
                      trackWithState.readyState
                    );
                  }

                  return true;
                }
              } catch (trackError) {
                console.warn(`⚠️ Error processing track ${i}:`, trackError);
                // Continue to next track instead of breaking
                continue;
              }
            }
          } else {
            console.log("⚠️ No text tracks found yet");
          }
          return false;
        } catch (error) {
          console.warn("⚠️ Error in enableSubtitles:", error);
          setSubtitleError("Failed to enable subtitles");
          setSubtitleTrackReady(false);
          return false;
        }
      };

      // Try multiple times with delays to catch when tracks are loaded
      const attempts = [0, 100, 300, 500, 1000, 2000, 3000];
      const timeouts: NodeJS.Timeout[] = [];
      let hasLoadedSuccessfully = false;

      attempts.forEach(delay => {
        const timeout = setTimeout(() => {
          try {
            const enabled = enableSubtitles();
            if (enabled && !subtitleError) {
              hasLoadedSuccessfully = true;
            }
            if (!enabled && delay === attempts[attempts.length - 1]) {
              console.warn(
                "⚠️ Could not enable subtitles after all attempts. Tracks:",
                video.textTracks?.length || 0
              );
              console.warn("Subtitle URL:", subtitleUrl);
            }
          } catch (error) {
            console.warn("⚠️ Error in subtitle enable attempt:", error);
            // Don't throw - just log and continue
          }
        }, delay);
        timeouts.push(timeout);
      });

      // Final timeout check - if subtitles haven't loaded after 10 seconds, give up
      const finalTimeout = setTimeout(() => {
        try {
          if (!hasLoadedSuccessfully && !subtitleTrackReady) {
            const tracks = video.textTracks;
            let hasErrorState = false;
            if (tracks && tracks.length > 0) {
              for (let i = 0; i < tracks.length; i++) {
                const track = tracks[i];
                if (!track) continue;
                
                try {
                  const trackWithState = track as TextTrackWithState;

                  if (track.kind === "subtitles" || track.kind === "captions") {
                    if (
                      trackWithState.readyState === undefined ||
                      trackWithState.readyState === null
                    ) {
                      hasErrorState = true;
                    } else if (
                      (track.cues?.length || 0) === 0 &&
                      trackWithState.readyState !== 1
                    ) {
                      hasErrorState = true;
                    }
                  }
                } catch (trackError) {
                  console.warn(`⚠️ Error checking track ${i} in finalTimeout:`, trackError);
                  // Continue to next track
                  continue;
                }
              }
            }
            if (hasErrorState && !subtitleError) {
              console.warn(
                "⚠️ Subtitle loading timeout - giving up after 10 seconds"
              );
              setSubtitleError("Subtitle track failed to load");
              setSubtitleTrackReady(false);
            }
          }
        } catch (error) {
          console.warn("⚠️ Error in finalTimeout:", error);
          // Don't throw - just log and continue
        }
      }, 10000);
      timeouts.push(finalTimeout);

      // Also listen for track changes
      const handleTrackChange = () => {
        try {
          console.log("📡 Track change event detected");
          enableSubtitles();
        } catch (error) {
          console.warn("⚠️ Error in handleTrackChange:", error);
          // Don't throw - just log and continue
        }
      };

      const handleLoadStart = () => {
        try {
          console.log("📡 Load start event - checking tracks");
          enableSubtitles();
        } catch (error) {
          console.warn("⚠️ Error in handleLoadStart:", error);
          // Don't throw - just log and continue
        }
      };

      if (video.textTracks) {
        video.textTracks.addEventListener("addtrack", handleTrackChange);
        video.textTracks.addEventListener("change", handleTrackChange);
        video.textTracks.addEventListener("removetrack", handleTrackChange);
      }

      // Listen for cue changes to verify cues are actually loaded
      const handleCueChange = () => {
        try {
          const tracks = video.textTracks;
          if (tracks && tracks.length > 0) {
            for (let i = 0; i < tracks.length; i++) {
              const track = tracks[i];
              if (!track) continue;

              try {
                if (
                  (track.kind === "subtitles" || track.kind === "captions") &&
                  track.mode === "showing"
                ) {
                  const cueCount = track.cues?.length || 0;
                  const activeCues = track.activeCues?.length || 0;
                  if (cueCount > 0) {
                    setSubtitleTrackReady(true);
                    setSubtitleError(null);
                    console.log("📝 Cue change detected:", {
                      totalCues: cueCount,
                      activeCues: activeCues,
                      currentTime: video.currentTime,
                    });
                  }
                }
              } catch (trackError) {
                console.warn(`⚠️ Error processing track ${i} in handleCueChange:`, trackError);
                // Continue to next track
                continue;
              }
            }
          }
        } catch (error) {
          console.warn("⚠️ Error in handleCueChange:", error);
          // Don't throw - just log and continue
        }
      };

      video.addEventListener("loadstart", handleLoadStart);
      video.addEventListener("loadedmetadata", handleTrackChange);
      video.addEventListener("loadeddata", handleTrackChange);
      video.addEventListener("canplay", handleTrackChange);
      video.addEventListener("timeupdate", handleCueChange);

      // Also listen directly on text tracks
      if (video.textTracks) {
        for (let i = 0; i < video.textTracks.length; i++) {
          const track = video.textTracks[i];
          if (track) {
            track.addEventListener("cuechange", handleCueChange);
          }
        }
      }

      return () => {
        // Clear all timeouts
        timeouts.forEach(timeout => {
          if (timeout) clearTimeout(timeout);
        });

        if (video.textTracks) {
          video.textTracks.removeEventListener("addtrack", handleTrackChange);
          video.textTracks.removeEventListener("change", handleTrackChange);
          video.textTracks.removeEventListener(
            "removetrack",
            handleTrackChange
          );
        }
        video.removeEventListener("loadstart", handleLoadStart);
        video.removeEventListener("loadedmetadata", handleTrackChange);
        video.removeEventListener("loadeddata", handleTrackChange);
        video.removeEventListener("canplay", handleTrackChange);
        video.removeEventListener("timeupdate", handleCueChange);

        // Remove cuechange listeners from tracks
        if (video.textTracks) {
          for (let i = 0; i < video.textTracks.length; i++) {
            const track = video.textTracks[i];
            if (track) {
              track.removeEventListener("cuechange", handleCueChange);
            }
          }
        }
      };
    }
  }, [subtitleUrl, subtitleLabel, subtitleUrlValid]);

  // Handle fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(
        !!(
          document.fullscreenElement ||
          (document as any).webkitFullscreenElement ||
          (document as any).msFullscreenElement ||
          (document as any).mozFullScreenElement
        )
      );
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
    document.addEventListener("msfullscreenchange", handleFullscreenChange);
    document.addEventListener("mozfullscreenchange", handleFullscreenChange);

    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener(
        "webkitfullscreenchange",
        handleFullscreenChange
      );
      document.removeEventListener(
        "msfullscreenchange",
        handleFullscreenChange
      );
      document.removeEventListener(
        "mozfullscreenchange",
        handleFullscreenChange
      );
    };
  }, []);

  // Don't render video if src is empty or invalid
  if (!src || src.trim() === "" || src === "undefined" || src === "null") {
    return (
      <div className="flex h-64 w-full items-center justify-center rounded-lg bg-gray-100 dark:bg-gray-800">
        <div className="text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-gray-200 dark:bg-gray-700">
            <svg
              className="h-8 w-8 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          </div>
          <p className="text-lg font-medium text-gray-900 dark:text-white">
            No Video Available
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            The translated video is not ready yet or the session is still
            processing.
          </p>
        </div>
      </div>
    );
  }

  // Handle empty or null src
  if (!src || src.trim() === "") {
    return (
      <div className="flex h-64 w-full items-center justify-center rounded-lg border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800">
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Video source not available
        </p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="video-container relative w-full overflow-hidden rounded-lg bg-black"
      style={
        videoAspectRatio
          ? {
              aspectRatio: videoAspectRatio.toString(),
              maxHeight: "300px",
            }
          : {
              maxHeight: "300px",
            }
      }
    >
      <video
        ref={videoRef}
        src={src}
        className={`${className} max-h-[300px]`}
        crossOrigin="anonymous"
        playsInline
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={() => {
          setIsPlaying(true);
          onPlay?.();
        }}
        onPause={() => {
          setIsPlaying(false);
          onPause?.();
        }}
        onEnded={() => {
          setIsPlaying(false);
          if (videoRef.current) {
            setCurrentTime(videoRef.current.duration);
            onTimeUpdate?.(videoRef.current.duration);
          }
        }}
        onError={handleError}
        aria-label={title}
      >
        {subtitleUrl && subtitleUrlValid === true && (
          <track
            kind="subtitles"
            srcLang="en"
            label={subtitleLabel || "Subtitles"}
            src={subtitleUrl}
            default
            onLoad={e => {
              // Enable subtitles programmatically when track loads
              const trackElement = e.target as HTMLTrackElement;
              if (trackElement.track) {
                const textTrack = trackElement.track;
                textTrack.mode = "showing";

                const textTrackWithState = textTrack as TextTrackWithState;

                // Log detailed track information
                console.log("✅ Track element loaded:", {
                  kind: textTrack.kind,
                  label: textTrack.label,
                  language: textTrack.language,
                  mode: textTrack.mode,
                  readyState: textTrackWithState.readyState,
                  cueCount: textTrack.cues?.length || 0,
                  activeCues: textTrack.activeCues?.length || 0,
                });

                // Wait a bit for cues to load, then check again
                setTimeout(() => {
                  const cueCount = textTrack.cues?.length || 0;
                  if (cueCount > 0) {
                    setSubtitleTrackReady(true);
                    setSubtitleError(null);
                    console.log(
                      "✅ Subtitles fully loaded with",
                      cueCount,
                      "cues"
                    );
                  } else {
                    console.warn(
                      "⚠️ Track loaded but has no cues. ReadyState:",
                      textTrackWithState.readyState
                    );
                    // Still mark as ready if track is in a valid state
                    if (
                      textTrackWithState.readyState !== undefined &&
                      textTrackWithState.readyState >= 1
                    ) {
                      setSubtitleTrackReady(true);
                    }
                  }
                }, 500);
              } else {
                console.warn("⚠️ Track loaded but track.track is null");
                setSubtitleError("Track loaded but invalid");
              }
            }}
            onError={e => {
              const trackElement = e.target as HTMLTrackElement;
              const textTrack = trackElement.track;
              let errorMsg = `Failed to load subtitle track`;

              // Check for track error state
              if (textTrack) {
                const textTrackWithState = textTrack as TextTrackWithState;

                console.error("❌ Track error state:", {
                  readyState: textTrackWithState.readyState,
                  kind: textTrack.kind,
                  label: textTrack.label,
                  error: textTrackWithState.error,
                });

                if (textTrackWithState.error) {
                  // TextTrack has an error property that may contain details
                  errorMsg = `Subtitle track error: ${textTrackWithState.error.message || "Unknown error"}`;
                } else if (
                  textTrackWithState.readyState === undefined ||
                  textTrackWithState.readyState === null
                ) {
                  errorMsg = "Subtitle track failed to initialize";
                } else if (
                  textTrackWithState.readyState === 0 &&
                  (textTrack.cues?.length || 0) === 0
                ) {
                  errorMsg = "Subtitle file not found or invalid";
                }
              } else {
                console.error(
                  "❌ Track element error - track.track is null",
                  e
                );
                errorMsg = "Subtitle track element failed to load";
              }

              console.error(
                "❌ Subtitle track error:",
                errorMsg,
                "URL:",
                subtitleUrl
              );
              setSubtitleError(errorMsg);
              setSubtitleTrackReady(false);
            }}
          />
        )}
      </video>

      {/* Embedded Subtitle Overlay - Hides read subtitles */}
      {showSubtitleOverlay && (
        <div 
          className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black via-black/90 to-transparent pointer-events-none transition-opacity duration-500 z-10"
          style={{
            opacity: 0.9,
          }}
        />
      )}

      {/* Error Overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
          <p className="px-4 text-center text-white">{error}</p>
        </div>
      )}

      {/* Subtitle Status Indicator (for debugging) */}
      {subtitleUrl && (
        <div className="absolute right-4 top-4 rounded-lg bg-black bg-opacity-60 px-3 py-2 text-xs text-white backdrop-blur-sm">
          {subtitleUrlValid === false || subtitleError ? (
            <span
              className="flex items-center gap-2"
              title={subtitleError || "Subtitle file not found"}
            >
              <span className="h-2 w-2 rounded-full bg-red-500"></span>
              Subtitles: Not Available
            </span>
          ) : subtitleTrackReady && subtitleUrlValid === true ? (
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-green-500"></span>
              Subtitles: ON
            </span>
          ) : subtitleUrlValid === null ? (
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse rounded-full bg-yellow-500"></span>
              Subtitles: Checking...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse rounded-full bg-yellow-500"></span>
              Subtitles: Loading...
            </span>
          )}
        </div>
      )}

      {/* Controls Overlay */}
      <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-0 transition-all duration-300 hover:bg-opacity-30">
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={togglePlay}
          className="flex h-16 w-16 items-center justify-center rounded-full bg-white bg-opacity-20 backdrop-blur-sm hover:bg-opacity-30"
          aria-label={isPlaying ? "Pause video" : "Play video"}
        >
          {isPlaying ? (
            <Pause className="h-8 w-8 text-white" />
          ) : (
            <Play
              className="h-8 w-8 text-white"
              style={{ marginLeft: "4px" }}
            />
          )}
        </motion.button>
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-4">
        <div className="flex items-center space-x-4">
          {/* Play/Pause Button */}
          <button
            onClick={togglePlay}
            className="text-white transition-colors hover:text-gray-300"
            aria-label={isPlaying ? "Pause video" : "Play video"}
          >
            {isPlaying ? (
              <Pause className="h-5 w-5" />
            ) : (
              <Play className="h-5 w-5" />
            )}
          </button>

          {/* Time Display */}
          <div className="text-sm text-white" aria-live="polite">
            {formatTime(currentTime)} / {formatTime(videoDuration || 0)}
          </div>

          {/* Progress Bar */}
          <div className="relative flex flex-1 items-center">
            {/* Background track */}
            <div className="absolute h-1 w-full rounded-lg bg-white bg-opacity-30"></div>
            {/* Filled portion */}
            <div
              className="absolute left-0 h-1 rounded-lg bg-blue-500 transition-all duration-150"
              style={{
                width: `${videoDuration > 0 ? (currentTime / (startTime + (duration || videoDuration || 0))) * 100 : 0}%`,
              }}
            ></div>
            {/* Slider */}
            <input
              type="range"
              min={startTime}
              max={startTime + (duration || videoDuration || 0)}
              value={currentTime}
              onChange={handleSeek}
              className="video-progress-slider relative z-10 w-full cursor-pointer appearance-none bg-transparent"
              style={{
                background: "transparent",
              }}
              aria-label="Video progress"
              aria-valuemin={startTime}
              aria-valuemax={startTime + (duration || videoDuration || 0)}
              aria-valuenow={currentTime}
              aria-valuetext={`${formatTime(currentTime)} of ${formatTime(videoDuration || 0)}`}
            />
          </div>

          {/* Volume Button */}
          <button
            onClick={toggleMute}
            className="text-white transition-colors hover:text-gray-300"
            aria-label={isMuted ? "Unmute video" : "Mute video"}
          >
            {isMuted ? (
              <VolumeX className="h-5 w-5" />
            ) : (
              <Volume2 className="h-5 w-5" />
            )}
          </button>

          {/* Fullscreen Button */}
          <button
            onClick={() => {
              if (containerRef.current) {
                const element = containerRef.current as FullscreenElement;
                const fullscreenDoc = document as FullscreenDocument;

                if (!isFullscreen) {
                  element.requestFullscreen?.() ||
                    element.webkitRequestFullscreen?.() ||
                    element.msRequestFullscreen?.() ||
                    element.mozRequestFullScreen?.();
                } else {
                  document.exitFullscreen?.() ||
                    fullscreenDoc.webkitExitFullscreen?.() ||
                    fullscreenDoc.msExitFullscreen?.() ||
                    fullscreenDoc.mozCancelFullScreen?.();
                }
              }
            }}
            className="text-white transition-colors hover:text-gray-300"
            aria-label={
              isFullscreen ? "Exit fullscreen mode" : "Enter fullscreen mode"
            }
          >
            <Maximize className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Title */}
      <div className="absolute left-4 top-4">
        <h3 className="rounded bg-black bg-opacity-50 px-3 py-1 text-lg font-semibold text-white">
          {title}
        </h3>
      </div>
    </div>
  );
}
