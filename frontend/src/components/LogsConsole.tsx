"use client";

import React, { useState, useEffect } from "react";
import { SystemLogEntry, fetchSystemLogs } from "@/lib/api";
import { Terminal, RefreshCw, AlertTriangle, CheckCircle, Info, ShieldAlert, X } from "lucide-react";

interface LogsConsoleProps {
  isOpen: boolean;
  onClose: () => void;
}

export function LogsConsole({ isOpen, onClose }: LogsConsoleProps) {
  const [logs, setLogs] = useState<SystemLogEntry[]>([]);
  const [filterLevel, setFilterLevel] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [isRefreshing, setIsRefreshing] = useState(false);

  const loadLogs = async () => {
    setIsRefreshing(true);
    const data = await fetchSystemLogs();
    setLogs(data);
    setIsRefreshing(false);
  };

  useEffect(() => {
    if (isOpen) {
      loadLogs();
      const interval = setInterval(loadLogs, 2000);
      return () => clearInterval(interval);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const filteredLogs = logs.filter((log) => {
    const matchesLevel = filterLevel === "ALL" || log.level === filterLevel;
    const matchesSearch =
      log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.module.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesLevel && matchesSearch;
  });

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-md flex items-center justify-center p-4 animate-fade-in">
      <div className="w-full max-w-4xl max-h-[85vh] bg-[var(--bg-card)] border border-[var(--border-color)] rounded-3xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-[var(--border-color)] flex items-center justify-between bg-[var(--bg-main)]">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
              <Terminal className="w-5 h-5 text-indigo-500 dark:text-cyan-400" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-[var(--text-primary)] flex items-center gap-2">
                Python FastAPI Diagnostic Console
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-mono">
                  Live Stream
                </span>
              </h3>
              <p className="text-[11px] text-[var(--text-muted)]">
                Real-time Python FastAPI backend event logs, warnings, and hardware execution trace
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={loadLogs}
              className={`p-2 rounded-xl border border-[var(--border-color)] bg-[var(--bg-card)] text-[var(--text-primary)] hover:border-indigo-500 transition-colors ${
                isRefreshing ? "animate-spin" : ""
              }`}
              title="Refresh Logs"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-xl border border-[var(--border-color)] bg-[var(--bg-card)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Filter Controls */}
        <div className="p-4 bg-[var(--bg-main)]/50 border-b border-[var(--border-color)] flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-1 overflow-x-auto py-1">
            {["ALL", "INFO", "WARN", "ERROR", "CRITICAL"].map((lvl) => (
              <button
                key={lvl}
                onClick={() => setFilterLevel(lvl)}
                className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                  filterLevel === lvl
                    ? "bg-indigo-600 text-white shadow-md"
                    : "bg-[var(--bg-card)] border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-indigo-500/40"
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>

          <input
            type="text"
            placeholder="Search module or message..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="px-3 py-1.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-color)] text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-indigo-500 w-full sm:w-64"
          />
        </div>

        {/* Log Stream Output Container */}
        <div className="p-4 overflow-y-auto font-mono text-xs space-y-2 bg-[#090d16] text-gray-200 flex-1 min-h-[350px]">
          {filteredLogs.length === 0 ? (
            <div className="py-12 text-center text-gray-500 italic">
              No logs match the selected filter query.
            </div>
          ) : (
            filteredLogs.map((log, i) => {
              let badgeColor = "text-cyan-400 bg-cyan-500/10 border-cyan-500/30";
              let Icon = Info;
              if (log.level === "WARN") {
                badgeColor = "text-amber-400 bg-amber-500/10 border-amber-500/30";
                Icon = AlertTriangle;
              } else if (log.level === "ERROR" || log.level === "CRITICAL") {
                badgeColor = "text-pink-500 bg-pink-500/10 border-pink-500/30";
                Icon = ShieldAlert;
              }

              return (
                <div
                  key={i}
                  className="p-2.5 rounded-xl bg-gray-900/80 border border-gray-800 flex items-start space-x-3 hover:border-gray-700 transition-colors"
                >
                  <span className="text-[10px] text-gray-500 shrink-0 pt-0.5">
                    {log.timestamp}
                  </span>

                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded border shrink-0 flex items-center gap-1 ${badgeColor}`}
                  >
                    <Icon className="w-3 h-3" />
                    {log.level}
                  </span>

                  <span className="text-[11px] font-bold text-indigo-300 shrink-0">
                    [{log.module}]
                  </span>

                  <span className="text-xs text-gray-200 break-all leading-relaxed">
                    {log.message}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
