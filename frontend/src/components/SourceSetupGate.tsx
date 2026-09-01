"use client";

import React, { useState, useRef } from "react";
import { changeVideoSource, uploadVideoParallelChunks } from "@/lib/api";
import { Upload, Radio, Video, Camera, Zap, Globe, HardDrive, ArrowRight, CheckCircle2, ShieldCheck } from "lucide-react";

interface SourceSetupGateProps {
  onLaunch: () => void;
}

export function SourceSetupGate({ onLaunch }: SourceSetupGateProps) {
  const [selectedMode, setSelectedMode] = useState<"upload" | "rtsp" | "presets">("upload");

  // RTSP Input State
  const [rtspUrl, setRtspUrl] = useState("rtsp://admin:123456@192.168.1.100:554/stream1");
  const [rtspLoading, setRtspLoading] = useState(false);
  const [rtspError, setRtspError] = useState<string | null>(null);

  // Upload State
  const [isUploading, setIsUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [uploadSpeedMbps, setUploadSpeedMbps] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedSourcePath, setUploadedSourcePath] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [manualVideoPath, setManualVideoPath] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  // Preset State
  const [selectedPreset, setSelectedPreset] = useState<{ type: "sample" | "test" | "webcam"; path: string; name: string } | null>({
    type: "test",
    path: "test/15690486_1920_1080_25fps.mp4",
    name: "HD Pedestrian Test Video #1",
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFileUpload = async (file: File) => {
    if (!file.type.startsWith("video/")) {
      setUploadError("Please select a valid video file (.mp4, .avi, .mov, etc.)");
      return;
    }

    setUploadError(null);
    setIsUploading(true);
    setUploadPct(0);
    setUploadSpeedMbps(0);

    const res = await uploadVideoParallelChunks(file, (pct, speed) => {
      setUploadPct(pct);
      setUploadSpeedMbps(speed);
    });

    setIsUploading(false);

    if (res.ok && res.videoSource) {
      setUploadedSourcePath(res.videoSource);
      setUploadedFileName(file.name);
    } else {
      setUploadError(res.error || "Failed parallel video upload");
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFileUpload(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processFileUpload(file);
  };

  const handleLaunchStudio = async () => {
    if (selectedMode === "upload") {
      const finalPath = manualVideoPath.trim() || uploadedSourcePath;
      if (!finalPath) {
        setUploadError("Please upload a video file or enter a valid video path first");
        return;
      }
      await changeVideoSource("file", finalPath);
    } else if (selectedMode === "rtsp") {
      if (!rtspUrl.trim()) {
        setRtspError("Please enter a valid RTSP or HTTP stream URL");
        return;
      }
      setRtspLoading(true);
      const ok = await changeVideoSource("rtsp", rtspUrl.trim());
      setRtspLoading(false);
      if (!ok) {
        setRtspError("Failed to connect to RTSP stream URL");
        return;
      }
    } else if (selectedMode === "presets" && selectedPreset) {
      await changeVideoSource(selectedPreset.type as any, selectedPreset.path);
    }

    onLaunch();
  };

  return (
    <div className="min-h-screen bg-[var(--bg-main)] text-[var(--text-primary)] flex flex-col justify-center items-center p-6 relative overflow-hidden">
      {/* Background Subtle Gradient Accents */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="w-full max-w-3xl space-y-8 z-10 animate-fade-in">
        {/* Brand Header */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs font-bold text-indigo-500 dark:text-cyan-400">
            <Zap className="w-3.5 h-3.5 animate-pulse" />
            <span>YOLO26s C++ Real-Time Studio</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
            Configure Your <span className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">Ingestion Stream</span>
          </h1>

          <p className="text-sm text-[var(--text-muted)] max-w-lg mx-auto">
            Choose your video stream source to begin real-time Person & Car tracking with interactive click-to-crop inspection.
          </p>
        </div>

        {/* Source Mode Selection Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Card 1: Upload Video */}
          <div
            onClick={() => setSelectedMode("upload")}
            className={`p-5 rounded-3xl border-2 transition-all cursor-pointer flex flex-col justify-between space-y-4 ${
              selectedMode === "upload"
                ? "bg-indigo-500/10 border-indigo-500 shadow-xl shadow-indigo-500/10 scale-[1.02]"
                : "bg-[var(--bg-card)] border-[var(--border-color)] hover:border-indigo-500/40"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="w-10 h-10 rounded-2xl bg-indigo-500/20 flex items-center justify-center">
                <Upload className="w-5 h-5 text-indigo-500 dark:text-cyan-400" />
              </div>
              {selectedMode === "upload" && <CheckCircle2 className="w-5 h-5 text-indigo-500" />}
            </div>

            <div>
              <h3 className="font-bold text-sm text-[var(--text-primary)]">Upload Video File</h3>
              <p className="text-[11px] text-[var(--text-muted)] mt-1">
                Sub-second 4MB parallel chunk uploader for MP4, AVI, MOV files
              </p>
            </div>
          </div>

          {/* Card 2: RTSP Stream */}
          <div
            onClick={() => setSelectedMode("rtsp")}
            className={`p-5 rounded-3xl border-2 transition-all cursor-pointer flex flex-col justify-between space-y-4 ${
              selectedMode === "rtsp"
                ? "bg-purple-500/10 border-purple-500 shadow-xl shadow-purple-500/10 scale-[1.02]"
                : "bg-[var(--bg-card)] border-[var(--border-color)] hover:border-purple-500/40"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="w-10 h-10 rounded-2xl bg-purple-500/20 flex items-center justify-center">
                <Radio className="w-5 h-5 text-purple-400" />
              </div>
              {selectedMode === "rtsp" && <CheckCircle2 className="w-5 h-5 text-purple-500" />}
            </div>

            <div>
              <h3 className="font-bold text-sm text-[var(--text-primary)]">Live RTSP / IP Camera</h3>
              <p className="text-[11px] text-[var(--text-muted)]">
                Direct C++ FFmpeg connection to live security cameras & network feeds
              </p>
            </div>
          </div>

          {/* Card 3: Demo Presets */}
          <div
            onClick={() => setSelectedMode("presets")}
            className={`p-5 rounded-3xl border-2 transition-all cursor-pointer flex flex-col justify-between space-y-4 ${
              selectedMode === "presets"
                ? "bg-pink-500/10 border-pink-500 shadow-xl shadow-pink-500/10 scale-[1.02]"
                : "bg-[var(--bg-card)] border-[var(--border-color)] hover:border-pink-500/40"
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="w-10 h-10 rounded-2xl bg-pink-500/20 flex items-center justify-center">
                <Video className="w-5 h-5 text-pink-400" />
              </div>
              {selectedMode === "presets" && <CheckCircle2 className="w-5 h-5 text-pink-500" />}
            </div>

            <div>
              <h3 className="font-bold text-sm text-[var(--text-primary)]">Demo Test Videos</h3>
              <p className="text-[11px] text-[var(--text-muted)]">
                Pre-configured 1080p pedestrian/traffic test videos & local Webcam
              </p>
            </div>
          </div>
        </div>

        {/* Dynamic Configuration Panel */}
        <div className="glass-panel p-6 rounded-3xl border border-[var(--border-color)] shadow-2xl space-y-4">
          {/* MODE 1: UPLOAD FILE CONFIGURATION */}
          {selectedMode === "upload" && (
            <div className="space-y-4">
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`p-8 rounded-2xl border-2 border-dashed transition-all cursor-pointer flex flex-col items-center justify-center text-center ${
                  isDragging
                    ? "border-pink-500 bg-pink-500/10 scale-[1.01]"
                    : "border-indigo-500/40 bg-[var(--bg-main)] hover:border-indigo-500 hover:bg-indigo-500/5"
                }`}
              >
                <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center mb-3">
                  <Upload className="w-7 h-7 text-indigo-500 dark:text-cyan-400 animate-bounce" />
                </div>
                <h4 className="font-bold text-sm text-[var(--text-primary)]">
                  {uploadedFileName ? `Selected: ${uploadedFileName}` : "Click or Drag & Drop Video File"}
                </h4>
                <p className="text-xs text-[var(--text-muted)] mt-1">
                  Supported formats: MP4, AVI, MOV, MKV (Parallel 4MB Chunked Transfer)
                </p>

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept="video/*"
                  className="hidden"
                />
              </div>

              <div className="flex items-center space-x-2 my-2">
                <div className="h-px bg-[var(--border-color)] flex-1" />
                <span className="text-[11px] font-bold text-[var(--text-muted)] uppercase tracking-wider">Or Enter Local Video Path</span>
                <div className="h-px bg-[var(--border-color)] flex-1" />
              </div>

              <div className="space-y-1">
                <input
                  type="text"
                  value={manualVideoPath}
                  onChange={(e) => {
                    setManualVideoPath(e.target.value);
                    setUploadError(null);
                  }}
                  placeholder="e.g. test/14365420-hd_1920_1080_60fps.mp4 or /path/to/video.mp4"
                  className="w-full px-4 py-2.5 rounded-xl bg-[var(--bg-main)] border border-[var(--border-color)] font-mono text-xs text-[var(--text-primary)] focus:outline-none focus:border-indigo-500"
                />
              </div>

              {isUploading && (
                <div className="p-4 rounded-2xl bg-[var(--bg-main)] border border-[var(--border-color)] space-y-2">
                  <div className="flex justify-between items-center text-xs font-semibold">
                    <span className="text-[var(--text-primary)]">Uploading Parallel Chunks directly to C++...</span>
                    <span className="font-mono text-cyan-400 font-bold">{uploadPct}% ({uploadSpeedMbps} Mbps)</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2.5 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 h-full transition-all duration-150"
                      style={{ width: `${uploadPct}%` }}
                    />
                  </div>
                </div>
              )}

              {uploadedSourcePath && !isUploading && (
                <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-xs font-semibold flex items-center space-x-2">
                  <ShieldCheck className="w-4 h-4" />
                  <span>Video file uploaded and verified by C++ engine! Ready to launch.</span>
                </div>
              )}

              {uploadError && (
                <div className="p-3 rounded-xl bg-pink-500/10 border border-pink-500/30 text-pink-500 text-xs font-medium">
                  {uploadError}
                </div>
              )}
            </div>
          )}

          {/* MODE 2: RTSP STREAM CONFIGURATION */}
          {selectedMode === "rtsp" && (
            <div className="space-y-4">
              <div className="p-4 rounded-2xl bg-[var(--bg-main)] border border-[var(--border-color)] space-y-3">
                <label className="text-xs font-bold text-[var(--text-primary)] flex items-center gap-1.5">
                  <Globe className="w-4 h-4 text-purple-400" />
                  Live RTSP / IP Camera Stream URL:
                </label>
                <input
                  type="text"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  placeholder="rtsp://admin:password@192.168.1.100:554/stream1"
                  className="w-full px-4 py-3 rounded-xl bg-[var(--bg-card)] border border-[var(--border-color)] font-mono text-xs text-[var(--text-primary)] focus:outline-none focus:border-purple-500"
                />
                <p className="text-[11px] text-[var(--text-muted)]">
                  C++ native FFmpeg hardware stream reader supports <code className="text-purple-400">rtsp://</code>, <code className="text-purple-400">http://</code>, and <code className="text-purple-400">rtmp://</code> feeds.
                </p>
              </div>

              {rtspError && (
                <div className="p-3 rounded-xl bg-pink-500/10 border border-pink-500/30 text-pink-500 text-xs font-medium">
                  {rtspError}
                </div>
              )}
            </div>
          )}

          {/* MODE 3: PRESET DEMO VIDEOS */}
          {selectedMode === "presets" && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                onClick={() => setSelectedPreset({ type: "test", path: "test/15690486_1920_1080_25fps.mp4", name: "HD Pedestrian Test Video #1" })}
                className={`p-4 rounded-2xl border text-left transition-all ${
                  selectedPreset?.path === "test/15690486_1920_1080_25fps.mp4"
                    ? "bg-pink-500/10 border-pink-500 shadow-md"
                    : "bg-[var(--bg-main)] border-[var(--border-color)] hover:border-pink-500/40"
                }`}
              >
                <div className="flex items-center space-x-2 mb-1">
                  <Video className="w-4 h-4 text-pink-500" />
                  <span className="font-bold text-xs text-[var(--text-primary)]">Pedestrians Test Video #1</span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)]">1920x1080 25fps High-Density Pedestrian Crowd</p>
              </button>

              <button
                onClick={() => setSelectedPreset({ type: "test", path: "test/14365420-hd_1920_1080_60fps.mp4", name: "60fps Traffic Test Video #2" })}
                className={`p-4 rounded-2xl border text-left transition-all ${
                  selectedPreset?.path === "test/14365420-hd_1920_1080_60fps.mp4"
                    ? "bg-cyan-500/10 border-cyan-500 shadow-md"
                    : "bg-[var(--bg-main)] border-[var(--border-color)] hover:border-cyan-500/40"
                }`}
              >
                <div className="flex items-center space-x-2 mb-1">
                  <Video className="w-4 h-4 text-cyan-400" />
                  <span className="font-bold text-xs text-[var(--text-primary)]">Traffic Test Video #2</span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)]">1920x1080 60fps High-Speed Highway Traffic</p>
              </button>

              <button
                onClick={() => setSelectedPreset({ type: "sample", path: "sample_assets/pedestrian_demo.mp4", name: "Synthetic Pedestrian Demo" })}
                className={`p-4 rounded-2xl border text-left transition-all ${
                  selectedPreset?.path === "sample_assets/pedestrian_demo.mp4"
                    ? "bg-indigo-500/10 border-indigo-500 shadow-md"
                    : "bg-[var(--bg-main)] border-[var(--border-color)] hover:border-indigo-500/40"
                }`}
              >
                <div className="flex items-center space-x-2 mb-1">
                  <HardDrive className="w-4 h-4 text-indigo-500" />
                  <span className="font-bold text-xs text-[var(--text-primary)]">Synthetic Benchmark Demo</span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)]">Pre-loaded synthetic pedestrian video</p>
              </button>

              <button
                onClick={() => setSelectedPreset({ type: "webcam", path: "0", name: "Live FaceTime / USB Webcam" })}
                className={`p-4 rounded-2xl border text-left transition-all ${
                  selectedPreset?.path === "0"
                    ? "bg-emerald-500/10 border-emerald-500 shadow-md"
                    : "bg-[var(--bg-main)] border-[var(--border-color)] hover:border-emerald-500/40"
                }`}
              >
                <div className="flex items-center space-x-2 mb-1">
                  <Camera className="w-4 h-4 text-emerald-500" />
                  <span className="font-bold text-xs text-[var(--text-primary)]">Live Mac Webcam</span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)]">FaceTime camera index 0 feed</p>
              </button>
            </div>
          )}

          {/* LAUNCH STUDIO BUTTON */}
          <button
            onClick={handleLaunchStudio}
            disabled={rtspLoading || isUploading}
            className="w-full py-4 rounded-2xl bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:opacity-90 text-white font-extrabold text-sm shadow-xl transition-all flex items-center justify-center space-x-2 group"
          >
            <span>🚀 Launch Interactive Tracker Studio</span>
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </div>
    </div>
  );
}
