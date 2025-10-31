"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Play, Pause, Volume2, VolumeX, Maximize } from "lucide-react";

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
}

export function VideoPreview({
  src,
  title = "Video Preview",
  className = "w-full h-full object-cover",
  startTime = 0,
  duration,
  onTimeUpdate,
  onDurationChange,
  onPlay,
  onPause,
  subtitleUrl,
  subtitleLabel = "Subtitles",
}: VideoPreviewProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [videoAspectRatio, setVideoAspectRatio] = useState<number | null>(null);
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

  const handleError = () => {
    setError("Failed to load video. Please try again.");
  };

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

  // Enable subtitles when track loads
  useEffect(() => {
    if (videoRef.current && subtitleUrl) {
      const video = videoRef.current;
      
      const enableSubtitles = () => {
        const tracks = video.textTracks;
        console.log('🔍 Checking text tracks:', tracks.length, 'tracks available');
        
        if (tracks && tracks.length > 0) {
          // Find the subtitle track and enable it
          for (let i = 0; i < tracks.length; i++) {
            const track = tracks[i];
            console.log(`Track ${i}:`, {
              kind: track.kind,
              label: track.label,
              language: track.language,
              mode: track.mode,
              readyState: track.readyState,
              cues: track.cues?.length || 0
            });
            
            if (track.kind === 'subtitles' || track.kind === 'captions') {
              if (track.mode !== 'showing') {
                track.mode = 'showing';
                console.log('✅ Subtitles enabled:', subtitleLabel || 'Subtitles', 'Mode set to:', track.mode);
              } else {
                console.log('✅ Subtitles already showing');
              }
              return true;
            }
          }
        } else {
          console.log('⚠️ No text tracks found yet');
        }
        return false;
      };

      // Try multiple times with delays to catch when tracks are loaded
      const attempts = [0, 100, 300, 500, 1000, 2000, 3000];
      const timeouts: NodeJS.Timeout[] = [];
      
      attempts.forEach((delay) => {
        const timeout = setTimeout(() => {
          const enabled = enableSubtitles();
          if (!enabled && delay === attempts[attempts.length - 1]) {
            console.warn('⚠️ Could not enable subtitles after all attempts. Tracks:', video.textTracks?.length || 0);
            console.warn('Subtitle URL:', subtitleUrl);
          }
        }, delay);
        timeouts.push(timeout);
      });

      // Also listen for track changes
      const handleTrackChange = () => {
        console.log('📡 Track change event detected');
        enableSubtitles();
      };

      const handleLoadStart = () => {
        console.log('📡 Load start event - checking tracks');
        enableSubtitles();
      };

      if (video.textTracks) {
        video.textTracks.addEventListener('addtrack', handleTrackChange);
        video.textTracks.addEventListener('change', handleTrackChange);
        video.textTracks.addEventListener('removetrack', handleTrackChange);
      }

      video.addEventListener('loadstart', handleLoadStart);
      video.addEventListener('loadedmetadata', handleTrackChange);
      video.addEventListener('loadeddata', handleTrackChange);
      video.addEventListener('canplay', handleTrackChange);

      return () => {
        // Clear all timeouts
        timeouts.forEach(timeout => clearTimeout(timeout));
        
        if (video.textTracks) {
          video.textTracks.removeEventListener('addtrack', handleTrackChange);
          video.textTracks.removeEventListener('change', handleTrackChange);
          video.textTracks.removeEventListener('removetrack', handleTrackChange);
        }
        video.removeEventListener('loadstart', handleLoadStart);
        video.removeEventListener('loadedmetadata', handleTrackChange);
        video.removeEventListener('loadeddata', handleTrackChange);
        video.removeEventListener('canplay', handleTrackChange);
      };
    }
  }, [subtitleUrl, subtitleLabel]);

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

  return (
    <div
      ref={containerRef}
      className="video-container"
      style={
        videoAspectRatio
          ? {
              aspectRatio: videoAspectRatio.toString(),
              position: "relative",
            }
          : undefined
      }
    >
      <video
        ref={videoRef}
        src={src}
        className={className}
        crossOrigin="anonymous"
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
        {subtitleUrl && (
          <track
            kind="subtitles"
            srcLang="en"
            label={subtitleLabel || "Subtitles"}
            src={subtitleUrl}
            default
            onLoad={(e) => {
              // Enable subtitles programmatically when track loads
              const track = e.target as HTMLTrackElement;
              if (track.track) {
                track.track.mode = 'showing';
                console.log('✅ Subtitle track loaded and enabled:', track.label, 'Mode:', track.track.mode);
              } else {
                console.warn('⚠️ Track loaded but track.track is null');
              }
            }}
            onError={(e) => {
              console.error('❌ Failed to load subtitle track:', subtitleUrl, e);
            }}
          />
        )}
      </video>

      {/* Error Overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-75">
          <p className="px-4 text-center text-white">{error}</p>
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
                if (!isFullscreen) {
                  containerRef.current.requestFullscreen?.() ||
                    containerRef.current.webkitRequestFullscreen?.() ||
                    containerRef.current.msRequestFullscreen?.() ||
                    containerRef.current.mozRequestFullScreen?.();
                } else {
                  document.exitFullscreen?.() ||
                    (document as any).webkitExitFullscreen?.() ||
                    (document as any).msExitFullscreen?.() ||
                    (document as any).mozCancelFullScreen?.();
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
