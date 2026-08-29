"use client";

import React, { useState, useEffect } from "react";
import { TrackObject, CROP_STREAM_URL } from "@/lib/api";
import { X, Target, Download, Eye, ArrowUpRight, Crosshair, Sparkles } from "lucide-react";

interface CroppedInspectorProps {
  selectedTrackId: number;
  selectedTrackObj: TrackObject | null;
  onDeselect: () => void;
}

export function CroppedInspector({
  selectedTrackId,
  selectedTrackObj,
  onDeselect,
}: CroppedInspectorProps) {
  const [streamError, setStreamError] = useState(false);
  const [key, setKey] = useState(0);

  useEffect(() => {
    setStreamError(false);
    setKey((prev) => prev + 1);
  }, [selectedTrackId]);

  const handleDownloadSnapshot = () => {
    if (selectedTrackId === -1) return;
    const img = document.getElementById("crop-stream-img") as HTMLImageElement;
    if (!img) return;

    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth || 400;
    canvas.height = img.naturalHeight || 400;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(img, 0, 0);
      const link = document.createElement("a");
      link.download = `tracked_person_${selectedTrackId}_snapshot.png`;
      link.href = canvas.toDataURL("image/png");
      link.click();
    }
  };

  if (selectedTrackId === -1) {
    return (
      <div className="hud-panel p-4 rounded-2xl w-full max-w-sm mx-auto shadow-2xl border border-[var(--border-color)] transition-all animate-fade-in">
        <div className="flex items-center space-x-2 mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
          <Crosshair className="w-4 h-4 text-indigo-500 dark:text-cyan-400" />
          <span>Interactive Target Inspector</span>
        </div>
        <div className="p-4 rounded-xl bg-[var(--bg-main)] border border-dashed border-[var(--border-color)] flex flex-col items-center justify-center text-center">
          <div className="w-10 h-10 rounded-full bg-indigo-500/10 flex items-center justify-center mb-2">
            <Target className="w-5 h-5 text-indigo-500 dark:text-cyan-400 animate-pulse" />
          </div>
          <p className="text-xs font-medium text-[var(--text-primary)]">
            No Object Selected
          </p>
          <p className="text-[11px] text-[var(--text-muted)] mt-1">
            Tap or click any tracked person or bounding box to view real-time cropped tracking on your phone or PC.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="hud-panel rounded-2xl w-full max-w-sm mx-auto overflow-hidden shadow-2xl border-2 border-pink-500/50 dark:border-pink-500/80 transition-all">
      {/* Header Bar */}
      <div className="px-4 py-3 bg-gradient-to-r from-pink-500/20 via-purple-500/20 to-indigo-500/20 border-b border-[var(--border-color)] flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-pink-500 animate-ping" />
          <span className="font-bold text-xs tracking-wider uppercase text-pink-600 dark:text-pink-400 flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5" /> Target #{selectedTrackId}
          </span>
        </div>
        <button
          onClick={onDeselect}
          className="p-1 rounded-lg hover:bg-black/20 text-[var(--text-muted)] hover:text-white transition-colors"
          title="Close Inspection"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Cropped Live Stream Display */}
      <div className="relative aspect-square w-full bg-black flex items-center justify-center overflow-hidden group">
        <img
          key={key}
          id="crop-stream-img"
          src={`${CROP_STREAM_URL}?t=${key}`}
          alt={`Cropped view of tracked object #${selectedTrackId}`}
          className="w-full h-full object-contain"
          onError={() => setStreamError(true)}
        />
        {streamError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-4 bg-gray-900/90 text-white">
            <Eye className="w-8 h-8 text-pink-400 mb-2" />
            <p className="text-xs font-semibold">Acquiring Target Crop...</p>
          </div>
        )}

        {/* Live Overlay HUD Badge */}
        <div className="absolute top-2 left-2 px-2.5 py-1 rounded-md bg-black/70 backdrop-blur-md text-[10px] font-mono text-cyan-300 border border-cyan-500/40">
          ROI CROP STREAM
        </div>

        {/* Quick Action Overlay */}
        <div className="absolute bottom-2 right-2">
          <button
            onClick={handleDownloadSnapshot}
            className="p-2.5 rounded-xl bg-pink-600 hover:bg-pink-500 active:scale-95 text-white text-xs flex items-center space-x-1 shadow-lg"
            title="Download Crop Snapshot"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Target Statistics & Telemetry */}
      <div className="p-3 bg-[var(--bg-card)] space-y-2 text-xs">
        <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-main)] border border-[var(--border-color)]">
          <span className="text-[var(--text-muted)]">Class Label:</span>
          <span className="font-semibold text-[var(--text-primary)] capitalize">
            {selectedTrackObj ? selectedTrackObj.label : "Person"}
          </span>
        </div>

        <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-main)] border border-[var(--border-color)]">
          <span className="text-[var(--text-muted)]">Confidence:</span>
          <span className="font-semibold text-emerald-500">
            {selectedTrackObj ? `${Math.round(selectedTrackObj.confidence * 100)}%` : "--"}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="p-2 rounded-lg bg-[var(--bg-main)] border border-[var(--border-color)]">
            <span className="text-[10px] text-[var(--text-muted)] block">Bounding Box [W x H]</span>
            <span className="font-mono text-xs font-medium text-[var(--text-primary)]">
              {selectedTrackObj ? `${selectedTrackObj.box.w} x ${selectedTrackObj.box.h} px` : "--"}
            </span>
          </div>

          <div className="p-2 rounded-lg bg-[var(--bg-main)] border border-[var(--border-color)]">
            <span className="text-[10px] text-[var(--text-muted)] block">Velocity Vector</span>
            <span className="font-mono text-xs font-medium text-indigo-500 dark:text-cyan-400 flex items-center gap-1">
              <ArrowUpRight className="w-3 h-3" />
              {selectedTrackObj ? `${selectedTrackObj.velocity.dx}, ${selectedTrackObj.velocity.dy}` : "0, 0"}
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between px-1 pt-1 text-[11px] text-[var(--text-muted)]">
          <span>Active Duration:</span>
          <span className="font-mono text-[var(--text-secondary)]">
            {selectedTrackObj ? `${selectedTrackObj.age} frames` : "--"}
          </span>
        </div>
      </div>
    </div>
  );
}
