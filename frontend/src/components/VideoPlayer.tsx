"use client";

import React, { useState, useRef } from "react";
import { STREAM_URL, TrackObject, selectTrackId, updateSettings } from "@/lib/api";
import { Play, Pause, Video, Sliders, Target, RefreshCw, Terminal, User, Car, Settings, Activity, ShieldCheck } from "lucide-react";

interface VideoPlayerProps {
  tracks: TrackObject[];
  selectedTrackId: number;
  onSelectTrack: (trackId: number) => void;
  confThreshold: number;
  nmsThreshold: number;
  tilingMode: number;
  isPlaying: boolean;
  sourceType: string;
  fps: number;
  latency: number;
  onOpenLogs: () => void;
  onOpenSetupModal: () => void;
}

export function VideoPlayer({
  tracks,
  selectedTrackId,
  onSelectTrack,
  confThreshold,
  nmsThreshold,
  tilingMode,
  isPlaying,
  sourceType,
  fps,
  latency,
  onOpenLogs,
  onOpenSetupModal,
}: VideoPlayerProps) {
  const [showSettings, setShowSettings] = useState(false);
  const [localConf, setLocalConf] = useState(confThreshold);
  const [localNms, setLocalNms] = useState(nmsThreshold);
  const [hoveredTrackId, setHoveredTrackId] = useState<number | null>(null);
  const [streamKey, setStreamKey] = useState(1);
  const [hasStreamError, setHasStreamError] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);

  const personCount = tracks.filter((t) => t.label === "person").length;
  const carCount = tracks.filter((t) => t.label === "car").length;

  const formattedFps = typeof fps === "number" ? fps.toFixed(2) : "0.00";
  const formattedLatency = typeof latency === "number" ? latency.toFixed(2) : "0.00";

  // Global Video Canvas Click Handler fallback
  const handleVideoClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();

    const scaleX = 1920 / rect.width;
    const scaleY = 1080 / rect.height;

    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    let closestTrackId = -1;
    let minDistance = Infinity;

    for (const trk of tracks) {
      const { x, y, w, h } = trk.box;

      if (clickX >= x - 25 && clickX <= x + w + 25 && clickY >= y - 25 && clickY <= y + h + 25) {
        closestTrackId = trk.track_id;
        break;
      }

      const cx = x + w / 2;
      const cy = y + h / 2;
      const dist = Math.hypot(clickX - cx, clickY - cy);
      if (dist < 250 && dist < minDistance) {
        minDistance = dist;
        closestTrackId = trk.track_id;
      }
    }

    if (closestTrackId !== -1) {
      onSelectTrack(closestTrackId);
      selectTrackId(closestTrackId);
    }
  };

  const handlePlayPause = async () => {
    await updateSettings({ is_playing: !isPlaying });
  };

  const handleConfChange = async (val: number) => {
    setLocalConf(val);
    await updateSettings({ conf_threshold: val });
  };

  const handleNmsChange = async (val: number) => {
    setLocalNms(val);
    await updateSettings({ nms_threshold: val });
  };

  const handleReconnectStream = () => {
    setHasStreamError(false);
    setStreamKey((prev) => prev + 1);
  };

  return (
    <div className="w-full space-y-3">
      {/* 1. Clean Unobstructed Video Frame Box */}
      <div className="relative w-full aspect-video rounded-3xl overflow-hidden border border-[var(--border-color)] bg-[#090d16] shadow-2xl group transition-all">
        {/* Live Video Stream Feed */}
        <div
          ref={containerRef}
          className="relative w-full h-full cursor-crosshair flex items-center justify-center select-none"
          onClick={handleVideoClick}
        >
          <img
            key={streamKey}
            src={`${STREAM_URL}?k=${streamKey}`}
            alt="Real-Time YOLO Object Tracking Stream"
            className="w-full h-full object-contain"
            onError={() => setHasStreamError(true)}
          />

          {/* Reconnect Overlay */}
          {hasStreamError && (
            <div className="absolute inset-0 bg-gray-950/90 backdrop-blur-md flex flex-col items-center justify-center text-center p-6 text-white z-20">
              <Video className="w-10 h-10 text-pink-500 mb-3 animate-bounce" />
              <h4 className="font-bold text-sm">Connecting to YOLO Tracking Stream...</h4>
              <p className="text-xs text-gray-400 max-w-sm mt-1 mb-4">
                Establishing real-time high-speed connection with C++ inference engine.
              </p>
              <button
                onClick={handleReconnectStream}
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center space-x-2 shadow-lg"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Reconnect Stream</span>
              </button>
            </div>
          )}

          {/* Interactive Bounding Box SVG Mouse Selection Layer */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 1920 1080">
            {tracks.map((trk) => {
              const isSelected = trk.track_id === selectedTrackId;
              const isHovered = trk.track_id === hoveredTrackId;

              return (
                <g key={trk.track_id} className="pointer-events-auto cursor-pointer">
                  {/* Invisible Click Surface padding for easy target selection */}
                  <rect
                    x={Math.max(0, trk.box.x - 10)}
                    y={Math.max(0, trk.box.y - 10)}
                    width={trk.box.w + 20}
                    height={trk.box.h + 20}
                    fill="transparent"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectTrack(trk.track_id);
                      selectTrackId(trk.track_id);
                    }}
                    onMouseEnter={() => setHoveredTrackId(trk.track_id)}
                    onMouseLeave={() => setHoveredTrackId(null)}
                  />

                  {/* Visible Interactive Bounding Box */}
                  <rect
                    x={trk.box.x}
                    y={trk.box.y}
                    width={trk.box.w}
                    height={trk.box.h}
                    fill={isSelected ? "rgba(255, 0, 220, 0.25)" : isHovered ? "rgba(0, 240, 255, 0.18)" : "transparent"}
                    stroke={isSelected ? "#FF00DC" : isHovered ? "#00F0FF" : "transparent"}
                    strokeWidth={isSelected ? 5 : isHovered ? 4 : 0}
                    rx="6"
                    className="transition-all duration-150"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectTrack(trk.track_id);
                      selectTrackId(trk.track_id);
                    }}
                    onMouseEnter={() => setHoveredTrackId(trk.track_id)}
                    onMouseLeave={() => setHoveredTrackId(null)}
                  />
                </g>
              );
            })}
          </svg>

          {/* Top-Left Click Hint & Class Badges */}
          <div className="absolute top-2.5 left-2.5 sm:top-4 sm:left-4 flex flex-wrap items-center gap-1.5 pointer-events-none z-10">
            <div className="bg-black/75 backdrop-blur-md px-2.5 py-1 sm:px-3.5 sm:py-1.5 rounded-lg sm:rounded-xl border border-white/15 text-[10px] sm:text-xs font-semibold text-white flex items-center space-x-1.5 shadow-xl">
              <Target className="w-3.5 h-3.5 text-pink-500 animate-pulse shrink-0" />
              <span className="hidden sm:inline">Click any box to lock crop view</span>
              <span className="sm:hidden">Tap box</span>
            </div>

            <div className="bg-cyan-500/20 backdrop-blur-md px-2 py-1 sm:px-3 sm:py-1.5 rounded-lg sm:rounded-xl border border-cyan-500/30 text-[10px] sm:text-xs font-bold text-cyan-300 flex items-center space-x-1 shadow-xl">
              <User className="w-3 h-3 shrink-0" />
              <span>Persons: {personCount}</span>
            </div>

            <div className="bg-amber-500/20 backdrop-blur-md px-2 py-1 sm:px-3 sm:py-1.5 rounded-lg sm:rounded-xl border border-amber-500/30 text-[10px] sm:text-xs font-bold text-amber-300 flex items-center space-x-1 shadow-xl">
              <Car className="w-3 h-3 shrink-0" />
              <span>Cars: {carCount}</span>
            </div>
          </div>

          {/* Top-Right Presentable FPS & Latency (ms) HUD Badges on Frame */}
          <div className="absolute top-2.5 right-2.5 sm:top-4 sm:right-4 flex items-center space-x-1.5 pointer-events-none z-10 font-mono">
            <div className="bg-black/80 backdrop-blur-md px-2 py-1 sm:px-3 sm:py-1.5 rounded-lg sm:rounded-xl border border-emerald-500/40 text-[10px] sm:text-xs font-bold text-emerald-400 flex items-center space-x-1 shadow-2xl">
              <Activity className="w-3 h-3 text-emerald-400 shrink-0" />
              <span>{formattedFps} FPS</span>
            </div>

            <div className="bg-black/80 backdrop-blur-md px-2 py-1 sm:px-3 sm:py-1.5 rounded-lg sm:rounded-xl border border-cyan-500/40 text-[10px] sm:text-xs font-bold text-cyan-300 flex items-center space-x-1 shadow-2xl">
              <ShieldCheck className="w-3 h-3 text-cyan-400 shrink-0" />
              <span>{formattedLatency} ms</span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Dedicated Control Toolbar Card Positioned BELOW Video Stream */}
      <div className="p-3 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-color)] text-[var(--text-primary)] flex flex-wrap items-center justify-between gap-2.5 shadow-lg">
        <div className="flex items-center space-x-2 overflow-x-auto py-0.5">
          <button
            onClick={handlePlayPause}
            className="p-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:scale-95 text-white transition-all shadow-md shrink-0"
            title={isPlaying ? "Pause Stream" : "Resume Stream"}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>

          <div className="h-6 w-px bg-[var(--border-color)] mx-1 shrink-0" />

          {/* Source Setup Modal Button */}
          <button
            onClick={onOpenSetupModal}
            className="px-3.5 py-2 rounded-xl text-xs font-extrabold flex items-center space-x-1.5 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:opacity-90 active:scale-95 text-white shadow-md transition-all shrink-0"
          >
            <Settings className="w-3.5 h-3.5" />
            <span>Source Setup (Video / RTSP)</span>
          </button>
        </div>

        <div className="flex items-center space-x-2 shrink-0">
          <button
            onClick={onOpenLogs}
            className="px-3 py-1.5 rounded-xl bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-500 dark:text-indigo-300 border border-indigo-500/30 text-xs font-semibold flex items-center space-x-1.5 transition-all"
            title="Open Live C++ System Diagnostics Logs"
          >
            <Terminal className="w-3.5 h-3.5 text-cyan-500" />
            <span>C++ Logs</span>
          </button>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className="px-3 py-1.5 rounded-xl bg-[var(--bg-main)] hover:bg-white/20 border border-[var(--border-color)] text-xs font-semibold flex items-center space-x-1.5 text-[var(--text-primary)] transition-colors"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Strictness</span>
          </button>
        </div>
      </div>

      {/* Strictness & Thresholds Popup */}
      {showSettings && (
        <div className="p-4 rounded-2xl bg-gray-900/95 backdrop-blur-2xl border border-white/20 text-white w-full max-w-sm ml-auto shadow-2xl space-y-4 animate-fade-in">
          <div className="flex items-center justify-between border-b border-white/10 pb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1">
              <Sliders className="w-3.5 h-3.5" /> Person & Car Strictness
            </span>
            <button onClick={() => setShowSettings(false)} className="text-gray-400 hover:text-white text-xs">
              Done
            </button>
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-300">Confidence Threshold:</span>
              <span className="font-mono text-cyan-400">{Math.round(localConf * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.20"
              max="0.85"
              step="0.05"
              value={localConf}
              onChange={(e) => handleConfChange(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-300">NMS IoU Deduplication:</span>
              <span className="font-mono text-cyan-400">{Math.round(localNms * 100)}%</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.60"
              step="0.05"
              value={localNms}
              onChange={(e) => handleNmsChange(parseFloat(e.target.value))}
              className="w-full accent-pink-400 cursor-pointer"
            />
          </div>
        </div>
      )}
    </div>
  );
}
