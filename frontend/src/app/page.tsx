"use client";

import React, { useState, useEffect } from "react";
import { ThemeProvider } from "@/components/ThemeProvider";
import { Navbar } from "@/components/Navbar";
import { VideoPlayer } from "@/components/VideoPlayer";
import { CroppedInspector } from "@/components/CroppedInspector";
import { AnalyticsPanel } from "@/components/AnalyticsPanel";
import { LogsConsole } from "@/components/LogsConsole";
import { SourceSetupGate } from "@/components/SourceSetupGate";
import { fetchBackendStatus, BackendStatus, TrackObject } from "@/lib/api";
import { AlertCircle, Server } from "lucide-react";

export default function Home() {
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number>(-1);
  const [connected, setConnected] = useState<boolean>(true);
  const [showLogs, setShowLogs] = useState<boolean>(false);

  // Gate state: false = Initial Setup Screen, true = Studio Dashboard Workspace
  const [isSourceConfigured, setIsSourceConfigured] = useState<boolean>(false);

  // Poll backend status periodically
  useEffect(() => {
    let isMounted = true;
    const interval = setInterval(async () => {
      const data = await fetchBackendStatus();
      if (!isMounted) return;

      if (data) {
        setStatus(data);
        setConnected(true);
        if (data.selected_track_id !== undefined) {
          setSelectedTrackId(data.selected_track_id);
        }
      } else {
        setConnected(false);
      }
    }, 250);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const selectedTrackObj: TrackObject | null =
    status && selectedTrackId !== -1
      ? status.tracks.find((t) => t.track_id === selectedTrackId) || null
      : null;

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-[var(--bg-main)] text-[var(--text-primary)] transition-colors duration-300 flex flex-col antialiased selection:bg-indigo-500 selection:text-white">
        {!isSourceConfigured ? (
          /* INITIAL SETUP GATE SCREEN */
          <SourceSetupGate onLaunch={() => setIsSourceConfigured(true)} />
        ) : (
          /* MAIN TRACKER STUDIO DASHBOARD */
          <>
            {/* Navbar Header */}
            <Navbar status={status} onChangeSource={() => setIsSourceConfigured(false)} />

            {/* Backend Connection Error Banner */}
            {!connected && (
              <div className="w-full bg-amber-500/10 border-b border-amber-500/20 px-4 sm:px-6 py-2 flex flex-wrap items-center justify-between text-xs text-amber-600 dark:text-amber-400 gap-2">
                <div className="flex items-center space-x-2">
                  <AlertCircle className="w-4 h-4 text-amber-500 animate-bounce shrink-0" />
                  <span>
                    Connecting to C++ YOLO Engine... Ensure backend server is active.
                  </span>
                </div>
                <div className="flex items-center space-x-2 font-mono">
                  <Server className="w-3.5 h-3.5" />
                  <span>Offline Mode</span>
                </div>
              </div>
            )}

            {/* Main Workspace Responsive Layout */}
            <main className="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-6 space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 relative">
                {/* Main Video Stream Section */}
                <div className="lg:col-span-8 space-y-6">
                  <VideoPlayer
                    tracks={status?.tracks || []}
                    selectedTrackId={selectedTrackId}
                    onSelectTrack={setSelectedTrackId}
                    confThreshold={status?.conf_threshold || 0.35}
                    nmsThreshold={status?.nms_threshold || 0.45}
                    tilingMode={status?.tiling_mode || 0}
                    isPlaying={status?.is_playing ?? true}
                    sourceType={status?.source_type || "sample"}
                    fps={status?.fps || 0}
                    latency={status?.inference_ms || 0}
                    onOpenLogs={() => setShowLogs(true)}
                    onOpenSetupModal={() => setIsSourceConfigured(false)}
                  />

                  <AnalyticsPanel
                    tracks={status?.tracks || []}
                    selectedTrackId={selectedTrackId}
                    onSelectTrack={setSelectedTrackId}
                    fps={status?.fps || 0}
                    latency={status?.inference_ms || 0}
                  />
                </div>

                {/* Right/Bottom Inspector Section */}
                <div className="lg:col-span-4 relative">
                  <div className="sticky top-20 space-y-4">
                    <CroppedInspector
                      selectedTrackId={selectedTrackId}
                      selectedTrackObj={selectedTrackObj}
                      onDeselect={() => setSelectedTrackId(-1)}
                    />
                  </div>
                </div>
              </div>
            </main>

            {/* Diagnostic C++ System Logs Modal Console */}
            <LogsConsole isOpen={showLogs} onClose={() => setShowLogs(false)} />
          </>
        )}
      </div>
    </ThemeProvider>
  );
}
