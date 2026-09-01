"use client";

import React, { useState, useRef } from "react";
import { changeVideoSource, uploadVideoParallelChunks } from "@/lib/api";
import { Upload, Radio, Video, Camera, Sparkles, X, Check, Zap, Globe, HardDrive } from "lucide-react";

interface LandingSetupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSourceSelected: () => void;
}

export function LandingSetupModal({ isOpen, onClose, onSourceSelected }: LandingSetupModalProps) {
  const [activeTab, setActiveTab] = useState<"upload" | "rtsp" | "presets">("upload");

  // RTSP Input State
  const [rtspUrl, setRtspUrl] = useState("rtsp://admin:123456@192.168.1.100:554/stream1");
  const [rtspLoading, setRtspLoading] = useState(false);
  const [rtspError, setRtspError] = useState<string | null>(null);

  // Upload State
  const [isUploading, setIsUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [uploadSpeedMbps, setUploadSpeedMbps] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

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
      await changeVideoSource("file", res.videoSource);
      onSourceSelected();
      onClose();
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

  const handleConnectRtsp = async () => {
    if (!rtspUrl.trim()) {
      setRtspError("Please enter a valid RTSP or HTTP stream URL");
      return;
    }

    setRtspError(null);
    setRtspLoading(true);

    const success = await changeVideoSource("rtsp", rtspUrl.trim());
    setRtspLoading(false);

    if (success) {
      onSourceSelected();
      onClose();
    } else {
      setRtspError("Failed to connect to RTSP stream URL");
    }
  };

  const handleSelectPreset = async (type: "sample" | "test" | "webcam", path: string) => {
    await changeVideoSource(type as any, path);
    onSourceSelected();
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xl flex items-center justify-center p-4 animate-fade-in">
      <div className="w-full max-w-2xl bg-[var(--bg-card)] border border-[var(--border-color)] rounded-3xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-5 border-b border-[var(--border-color)] flex items-center justify-between bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-indigo-500 to-pink-500 p-0.5 shadow-lg">
              <div className="w-full h-full bg-[var(--bg-main)] rounded-[14px] flex items-center justify-center">
                <Zap className="w-5 h-5 text-indigo-500 dark:text-cyan-400" />
              </div>
            </div>
            <div>
              <h2 className="font-extrabold text-base tracking-tight text-[var(--text-primary)] flex items-center gap-2">
                YOLO26 <span className="bg-gradient-to-r from-indigo-500 to-pink-500 bg-clip-text text-transparent">Ingestion Setup</span>
              </h2>
              <p className="text-xs text-[var(--text-muted)]">
                Select your video input stream source to begin real-time Person & Car tracking
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl border border-[var(--border-color)] bg-[var(--bg-main)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Ingestion Source Tabs */}
        <div className="grid grid-cols-3 p-2 bg-[var(--bg-main)] border-b border-[var(--border-color)] gap-2">
          <button
            onClick={() => setActiveTab("upload")}
            className={`py-2.5 px-3 rounded-2xl text-xs font-bold flex items-center justify-center space-x-2 transition-all ${
              activeTab === "upload"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                : "text-[var(--text-secondary)] hover:bg-[var(--bg-card)]"
            }`}
          >
            <Upload className="w-4 h-4" />
            <span>Upload Video File</span>
          </button>

          <button
            onClick={() => setActiveTab("rtsp")}
            className={`py-2.5 px-3 rounded-2xl text-xs font-bold flex items-center justify-center space-x-2 transition-all ${
              activeTab === "rtsp"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                : "text-[var(--text-secondary)] hover:bg-[var(--bg-card)]"
            }`}
          >
            <Radio className="w-4 h-4" />
            <span>RTSP / Live Stream</span>
          </button>

          <button
            onClick={() => setActiveTab("presets")}
            className={`py-2.5 px-3 rounded-2xl text-xs font-bold flex items-center justify-center space-x-2 transition-all ${
              activeTab === "presets"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30"
                : "text-[var(--text-secondary)] hover:bg-[var(--bg-card)]"
            }`}
          >
            <Video className="w-4 h-4" />
            <span>Demo Videos / Webcam</span>
          </button>
        </div>

        {/* Tab Content Area */}
        <div className="p-6 space-y-4">
          {/* TAB 1: UPLOAD VIDEO FILE (Parallel Multi-Threaded Chunked Uploader) */}
          {activeTab === "upload" && (
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
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-pink-500/20 border border-indigo-500/30 flex items-center justify-center mb-3">
                  <Upload className="w-7 h-7 text-indigo-500 dark:text-cyan-400 animate-bounce" />
                </div>
                <h4 className="font-bold text-sm text-[var(--text-primary)]">
                  Click or Drag & Drop Video File Here
                </h4>
                <p className="text-xs text-[var(--text-muted)] mt-1 max-w-sm">
                  Powered by <strong className="text-indigo-400">Parallel Multi-Threaded Chunked Uploads</strong> for sub-second 100MB+ transfers directly to Python FastAPI.
                </p>

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept="video/*"
                  className="hidden"
                />
              </div>

              {/* Upload Progress Indicator */}
              {isUploading && (
                <div className="p-4 rounded-2xl bg-[var(--bg-main)] border border-[var(--border-color)] space-y-2 animate-fade-in">
                  <div className="flex justify-between items-center text-xs font-semibold">
                    <span className="text-[var(--text-primary)] flex items-center gap-2">
                      <Sparkles className="w-3.5 h-3.5 text-pink-500 animate-spin" />
                      Uploading Parallel Chunks to Python FastAPI Engine...
                    </span>
                    <span className="font-mono text-cyan-400 font-bold">{uploadPct}% ({uploadSpeedMbps} Mbps)</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2.5 overflow-hidden border border-white/10">
                    <div
                      className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 h-full transition-all duration-150"
                      style={{ width: `${uploadPct}%` }}
                    />
                  </div>
                </div>
              )}

              {uploadError && (
                <div className="p-3 rounded-xl bg-pink-500/10 border border-pink-500/30 text-pink-500 text-xs font-medium">
                  {uploadError}
                </div>
              )}
            </div>
          )}

          {/* TAB 2: RTSP / IP CAMERA LIVE STREAM */}
          {activeTab === "rtsp" && (
            <div className="space-y-4">
              <div className="p-4 rounded-2xl bg-[var(--bg-main)] border border-[var(--border-color)] space-y-3">
                <label className="text-xs font-bold text-[var(--text-primary)] flex items-center gap-1.5">
                  <Globe className="w-4 h-4 text-cyan-400" />
                  Enter RTSP / IP Camera / HTTP Stream URL:
                </label>
                <input
                  type="text"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  placeholder="rtsp://admin:password@192.168.1.100:554/h264Preview_01_main"
                  className="w-full px-4 py-2.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-color)] font-mono text-xs text-[var(--text-primary)] focus:outline-none focus:border-indigo-500"
                />
                <p className="text-[11px] text-[var(--text-muted)]">
                  Supports live <code className="text-indigo-400">rtsp://</code>, <code className="text-indigo-400">rtmp://</code>, <code className="text-indigo-400">http://</code>, and <code className="text-indigo-400">https://</code> stream URLs via OpenCV hardware reader.
                </p>
              </div>

              {rtspError && (
                <div className="p-3 rounded-xl bg-pink-500/10 border border-pink-500/30 text-pink-500 text-xs font-medium">
                  {rtspError}
                </div>
              )}

              <button
                onClick={handleConnectRtsp}
                disabled={rtspLoading}
                className="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-pink-500 hover:opacity-90 text-white font-bold text-xs shadow-lg transition-all flex items-center justify-center space-x-2"
              >
                {rtspLoading ? (
                  <span>Connecting to Stream...</span>
                ) : (
                  <>
                    <Radio className="w-4 h-4" />
                    <span>Connect Live Stream</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* TAB 3: DEMO PRESET VIDEOS & WEBCAM */}
          {activeTab === "presets" && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                onClick={() => handleSelectPreset("test", "test/15690486_1920_1080_25fps.mp4")}
                className="p-4 rounded-2xl bg-[var(--bg-main)] border border-[var(--border-color)] hover:border-pink-500 text-left transition-all group"
              >
                <div className="flex items-center space-x-3 mb-1">
                  <Video className="w-5 h-5 text-pink-500 group-hover:scale-110 transition-transform" />
                  <span className="font-bold text-xs text-[var(--text-primary)]">Test Video #1 (HD Pedestrians)</span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)]">1920x1080 25fps High-Density Pedestrian Surveillance</p>
              </button>

              <button
                onClick={() => handleSelectPreset("test", "test/14365420-hd_1920_1080_60fps.mp4")}
                className="p-4 rounded-2xl bg-[var(--bg-main)] border border-[var(--border-color)] hover:border-cyan-400 text-left transition-all group"
              >
                <div className="flex items-center space-x-3 mb-1">
                  <Video className="w-5 h-5 text-cyan-400 group-hover:scale-110 transition-transform" />
                  <span className="font-bold text-xs text-[var(--text-primary)]">Test Video #2 (60fps Traffic)</span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)]">1920x1080 60fps High-Speed Car & Traffic Monitoring</p>
              </button>

              <button
                onClick={() => handleSelectPreset("sample", "sample_assets/pedestrian_demo.mp4")}
                className="p-4 rounded-2xl bg-[var(--bg-main)] border border-[var(--border-color)] hover:border-indigo-500 text-left transition-all group"
              >
                <div className="flex items-center space-x-3 mb-1">
                  <HardDrive className="w-5 h-5 text-indigo-500 group-hover:scale-110 transition-transform" />
                  <span className="font-bold text-xs text-[var(--text-primary)]">Synthetic Pedestrian Demo</span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)]">Pre-loaded synthetic benchmark video file</p>
              </button>

              <button
                onClick={() => handleSelectPreset("webcam", "0")}
                className="p-4 rounded-2xl bg-[var(--bg-main)] border border-[var(--border-color)] hover:border-emerald-500 text-left transition-all group"
              >
                <div className="flex items-center space-x-3 mb-1">
                  <Camera className="w-5 h-5 text-emerald-500 group-hover:scale-110 transition-transform" />
                  <span className="font-bold text-xs text-[var(--text-primary)]">Live Mac Webcam</span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)]">Connect to local FaceTime / USB camera feed</p>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
