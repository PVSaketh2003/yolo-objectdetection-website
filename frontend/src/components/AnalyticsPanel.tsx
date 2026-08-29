"use client";

import React from "react";
import { TrackObject, selectTrackId } from "@/lib/api";
import { Users, Activity, Crosshair, CheckCircle2, ChevronRight, User, Car } from "lucide-react";

interface AnalyticsPanelProps {
  tracks: TrackObject[];
  selectedTrackId: number;
  onSelectTrack: (trackId: number) => void;
  fps: number;
  latency: number;
}

export function AnalyticsPanel({
  tracks,
  selectedTrackId,
  onSelectTrack,
  fps,
  latency,
}: AnalyticsPanelProps) {
  const formattedFps = typeof fps === "number" ? fps.toFixed(2) : "0.00";
  const formattedLatency = typeof latency === "number" ? latency.toFixed(2) : "0.00";

  const personCount = tracks.filter((t) => t.label === "person").length;
  const carCount = tracks.filter((t) => t.label === "car").length;

  const handleSelectTrack = (id: number) => {
    onSelectTrack(id);
    selectTrackId(id);
  };

  return (
    <div className="w-full space-y-4">
      {/* Telemetry Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="glass-panel p-4 rounded-2xl border border-[var(--border-color)]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
              Active Targets (Person / Car)
            </span>
            <Users className="w-4 h-4 text-indigo-500 dark:text-cyan-400" />
          </div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-2xl font-bold tracking-tight text-[var(--text-primary)] font-mono">
              {tracks.length}
            </span>
            <div className="flex items-center space-x-2 text-[11px] font-semibold">
              <span className="text-cyan-500 flex items-center gap-1">
                <User className="w-3 h-3" /> {personCount}
              </span>
              <span className="text-amber-500 flex items-center gap-1">
                <Car className="w-3 h-3" /> {carCount}
              </span>
            </div>
          </div>
        </div>

        <div className="glass-panel p-4 rounded-2xl border border-[var(--border-color)]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
              Inference Speed
            </span>
            <Activity className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-2xl font-bold tracking-tight text-[var(--text-primary)] font-mono">
              {formattedFps} <span className="text-xs text-[var(--text-muted)] font-normal font-sans">FPS</span>
            </span>
            <span className="text-[11px] text-indigo-500 dark:text-cyan-400 font-medium">
              30 FPS Target
            </span>
          </div>
        </div>

        <div className="glass-panel p-4 rounded-2xl border border-[var(--border-color)]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
              Model Latency
            </span>
            <Crosshair className="w-4 h-4 text-pink-500" />
          </div>
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-2xl font-bold tracking-tight text-[var(--text-primary)] font-mono">
              {formattedLatency} <span className="text-xs text-[var(--text-muted)] font-normal font-sans">ms</span>
            </span>
            <span className="text-[11px] text-emerald-500 font-medium">
              Apple CoreML / NE
            </span>
          </div>
        </div>
      </div>

      {/* Track Inventory List with Person 👤 & Car 🚗 Icons */}
      <div className="glass-panel rounded-2xl p-4 border border-[var(--border-color)]">
        <div className="flex items-center justify-between mb-3 pb-2 border-b border-[var(--border-color)]">
          <div className="flex items-center space-x-2">
            <Crosshair className="w-4 h-4 text-indigo-500 dark:text-cyan-400" />
            <h3 className="font-bold text-sm text-[var(--text-primary)]">
              Tracked Person & Car Inventory ({tracks.length})
            </h3>
          </div>
          <span className="text-[11px] text-[var(--text-muted)]">
            Click row to lock top-right crop inspection
          </span>
        </div>

        {tracks.length === 0 ? (
          <div className="py-8 text-center text-xs text-[var(--text-muted)]">
            Searching frame for Persons and Cars...
          </div>
        ) : (
          <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
            {tracks.map((trk) => {
              const isSelected = trk.track_id === selectedTrackId;
              const isCar = trk.label === "car";

              return (
                <div
                  key={trk.track_id}
                  onClick={() => handleSelectTrack(trk.track_id)}
                  className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                    isSelected
                      ? "bg-gradient-to-r from-pink-500/10 via-purple-500/10 to-indigo-500/10 border-pink-500/80 shadow-md"
                      : "bg-[var(--bg-main)] border-[var(--border-color)] hover:border-indigo-500/40"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <span
                      className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-xs ${
                        isSelected
                          ? "bg-pink-500 text-white shadow-md shadow-pink-500/30"
                          : isCar
                          ? "bg-amber-500/10 text-amber-500 border border-amber-500/20"
                          : "bg-cyan-500/10 text-cyan-500 border border-cyan-500/20"
                      }`}
                    >
                      #{trk.track_id}
                    </span>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-bold text-xs text-[var(--text-primary)] capitalize flex items-center gap-1">
                          {isCar ? (
                            <Car className="w-3.5 h-3.5 text-amber-500" />
                          ) : (
                            <User className="w-3.5 h-3.5 text-cyan-400" />
                          )}
                          {trk.label}
                        </span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-mono font-medium">
                          {Math.round(trk.confidence * 100)}% Conf
                        </span>
                      </div>
                      <span className="text-[10px] text-[var(--text-muted)] font-mono block">
                        Box: [{trk.box.x}, {trk.box.y}, {trk.box.w}x{trk.box.h}]
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] text-[var(--text-muted)] font-mono">
                      {trk.age}f
                    </span>
                    <ChevronRight
                      className={`w-4 h-4 transition-transform ${
                        isSelected ? "text-pink-500 translate-x-0.5" : "text-[var(--text-muted)]"
                      }`}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
