"use client";

import React from "react";
import { useTheme } from "./ThemeProvider";
import { Sun, Moon, Cpu, ShieldCheck, Zap, Activity, Settings } from "lucide-react";
import { BackendStatus } from "@/lib/api";

interface NavbarProps {
  status: BackendStatus | null;
  onChangeSource?: () => void;
}

export function Navbar({ status, onChangeSource }: NavbarProps) {
  const { theme, toggleTheme } = useTheme();

  const formattedFps = status && typeof status.fps === "number" ? status.fps.toFixed(2) : "0.00";
  const formattedLatency = status && typeof status.inference_ms === "number" ? status.inference_ms.toFixed(2) : "0.00";

  return (
    <header className="w-full min-h-[64px] border-b border-[var(--border-color)] bg-[var(--bg-card)] backdrop-blur-xl sticky top-0 z-40 px-4 md:px-6 py-2 flex flex-wrap items-center justify-between gap-3 shadow-sm transition-all">
      {/* Brand & Logo */}
      <div className="flex items-center space-x-3 shrink-0">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 p-0.5 shadow-md shadow-indigo-500/20">
          <div className="w-full h-full bg-[var(--bg-main)] rounded-[10px] flex items-center justify-center">
            <Zap className="w-4 h-4 text-indigo-500 dark:text-cyan-400 animate-pulse" />
          </div>
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="font-bold text-base tracking-tight text-[var(--text-primary)]">
              YOLO26 <span className="bg-gradient-to-r from-indigo-500 to-pink-500 bg-clip-text text-transparent">Tracker Studio</span>
            </h1>
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-600 dark:text-cyan-400 border border-indigo-500/20">
              OFFLINE C++
            </span>
          </div>
          <p className="text-[11px] text-[var(--text-muted)] hidden sm:block">
            Interactive Real-Time Object Tracking & Inspection
          </p>
        </div>
      </div>

      {/* Center Dynamic Metrics Badges */}
      <div className="flex items-center space-x-2 md:space-x-4 overflow-x-auto py-1">
        <div className="flex items-center space-x-1.5 text-xs text-[var(--text-secondary)] bg-[var(--bg-main)] px-3 py-1.5 rounded-xl border border-[var(--border-color)] shrink-0">
          <Cpu className="w-3.5 h-3.5 text-indigo-500 dark:text-cyan-400" />
          <span>Model: <strong className="text-[var(--text-primary)]">YOLO26s (ONNX)</strong></span>
        </div>

        <div className="flex items-center space-x-1.5 text-xs text-[var(--text-secondary)] bg-[var(--bg-main)] px-3 py-1.5 rounded-xl border border-[var(--border-color)] shrink-0">
          <Activity className="w-3.5 h-3.5 text-emerald-500" />
          <span>Engine FPS: <strong className="text-[var(--text-primary)] font-mono">{formattedFps}</strong></span>
        </div>

        <div className="flex items-center space-x-1.5 text-xs text-[var(--text-secondary)] bg-[var(--bg-main)] px-3 py-1.5 rounded-xl border border-[var(--border-color)] shrink-0">
          <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
          <span>Latency: <strong className="text-[var(--text-primary)] font-mono">{formattedLatency} ms</strong></span>
        </div>
      </div>

      {/* Right Controls: Change Source & Theme Switcher */}
      <div className="flex items-center space-x-2 shrink-0">
        {onChangeSource && (
          <button
            onClick={onChangeSource}
            className="px-3 py-1.5 rounded-xl border border-[var(--border-color)] bg-[var(--bg-main)] text-xs font-semibold text-[var(--text-primary)] hover:border-indigo-500 transition-all flex items-center space-x-1.5"
            title="Change Ingestion Stream Source"
          >
            <Settings className="w-3.5 h-3.5 text-indigo-500 dark:text-cyan-400" />
            <span>Change Source</span>
          </button>
        )}

        <button
          onClick={toggleTheme}
          className="p-2 rounded-xl border border-[var(--border-color)] bg-[var(--bg-main)] text-[var(--text-primary)] hover:border-indigo-500 transition-all flex items-center justify-center group"
          title={`Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`}
        >
          {theme === "dark" ? (
            <Sun className="w-4 h-4 text-amber-400 group-hover:rotate-45 transition-transform" />
          ) : (
            <Moon className="w-4 h-4 text-indigo-600 group-hover:-rotate-12 transition-transform" />
          )}
        </button>
      </div>
    </header>
  );
}
