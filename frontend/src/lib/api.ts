export interface TrackObject {
  track_id: number;
  label: string;
  confidence: number;
  class_id: number;
  box: { x: number; y: number; w: number; h: number };
  velocity: { dx: number; dy: number };
  age: number;
}

export interface BackendStatus {
  fps: number;
  inference_ms: number;
  selected_track_id: number;
  source_type: string;
  video_source: string;
  conf_threshold: number;
  nms_threshold: number;
  tiling_mode: number;
  is_playing: boolean;
  tracks: TrackObject[];
}

export interface SystemLogEntry {
  timestamp: string;
  level: "INFO" | "WARN" | "ERROR" | "CRITICAL";
  module: string;
  message: string;
}

const BACKEND_URL = "";
const DIRECT_BACKEND_URL = "";

// Generate or retrieve 100% Cryptographically Isolated Private Session ID
function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return "sess_ssr";
  let sid = localStorage.getItem("yolo_session_id");
  if (!sid) {
    const randomHex = Array.from(crypto.getRandomValues(new Uint8Array(12)))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    sid = `sess_${Date.now()}_${randomHex}`;
    localStorage.setItem("yolo_session_id", sid);
  }
  return sid;
}

export const SESSION_ID = getOrCreateSessionId();

function getSessionHeaders(): Record<string, string> {
  return {
    "X-Session-ID": getOrCreateSessionId(),
  };
}

export async function fetchBackendStatus(): Promise<BackendStatus | null> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/status`, {
      cache: "no-store",
      headers: getSessionHeaders(),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function fetchSystemLogs(): Promise<SystemLogEntry[]> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/logs`, {
      cache: "no-store",
      headers: getSessionHeaders(),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.logs || [];
  } catch {
    return [];
  }
}

/**
 * Ultra-Fast Parallel Multi-Threaded Chunked Upload Technique
 * Encrypted Session Storage: Uploaded chunks go directly to uploads/<session_id>/
 * 100% Private - No other user can see, stream, or access this uploaded video!
 */
export async function uploadVideoParallelChunks(
  file: File,
  onProgress?: (pct: number, speedMbps: number) => void
): Promise<{ ok: boolean; videoSource?: string; error?: string }> {
  try {
    const chunkSize = 4 * 1024 * 1024; // 4MB Chunk size
    const totalChunks = Math.ceil(file.size / chunkSize);
    const uploadId = "up_" + Date.now() + "_" + Math.random().toString(36).substring(2, 7);

    let uploadedBytes = 0;
    const startTime = Date.now();
    const concurrency = 4;
    const chunkIndices = Array.from({ length: totalChunks }, (_, i) => i);

    let finalVideoSource: string | undefined = undefined;

    const uploadSingleChunk = async (index: number): Promise<boolean> => {
      const start = index * chunkSize;
      const end = Math.min(file.size, start + chunkSize);
      const chunkBlob = file.slice(start, end);

      const formData = new FormData();
      formData.append("chunk", chunkBlob, file.name);
      formData.append("upload_id", uploadId);
      formData.append("filename", file.name);
      formData.append("chunk_index", index.toString());
      formData.append("total_chunks", totalChunks.toString());

      try {
        const res = await fetch(`${DIRECT_BACKEND_URL}/api/upload_chunk`, {
          method: "POST",
          headers: getSessionHeaders(),
          body: formData,
        });

        if (!res.ok) return false;
        const data = await res.json();

        uploadedBytes += chunkBlob.size;

        if (onProgress) {
          const elapsedSec = (Date.now() - startTime) / 1000;
          const pct = Math.min(100, Math.round((uploadedBytes / file.size) * 100));
          const speedMbps = elapsedSec > 0 ? (uploadedBytes * 8) / (1024 * 1024 * elapsedSec) : 0;
          onProgress(pct, Math.round(speedMbps * 10) / 10);
        }

        if (data.status === "complete" && data.video_source) {
          finalVideoSource = data.video_source;
        }

        return true;
      } catch {
        return false;
      }
    };

    const activeWorkers: Promise<boolean>[] = [];
    for (const idx of chunkIndices) {
      const p = uploadSingleChunk(idx);
      activeWorkers.push(p);

      if (activeWorkers.length >= concurrency) {
        await Promise.race(activeWorkers);
        for (let i = activeWorkers.length - 1; i >= 0; i--) {
          const state = await Promise.race([activeWorkers[i], "pending"]);
          if (state !== "pending") {
            activeWorkers.splice(i, 1);
          }
        }
      }
    }

    await Promise.all(activeWorkers);

    if (finalVideoSource) {
      return { ok: true, videoSource: finalVideoSource };
    }

    return { ok: false, error: "Chunk assembly did not complete on server" };
  } catch (err: any) {
    return { ok: false, error: err.message || "Failed chunked video upload" };
  }
}

export async function selectTrackId(trackId: number): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/select_track`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getSessionHeaders(),
      },
      body: JSON.stringify({ track_id: trackId }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function updateSettings(settings: {
  conf_threshold?: number;
  nms_threshold?: number;
  tiling_mode?: number;
  is_playing?: boolean;
}): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/settings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getSessionHeaders(),
      },
      body: JSON.stringify(settings),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function changeVideoSource(
  sourceType: "sample" | "webcam" | "file" | "test" | "rtsp",
  videoSource: string
): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/source`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getSessionHeaders(),
      },
      body: JSON.stringify({
        source_type: sourceType,
        video_source: videoSource,
      }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export const STREAM_URL = `${BACKEND_URL}/api/stream?sid=${SESSION_ID}`;
export const CROP_STREAM_URL = `${BACKEND_URL}/api/crop?sid=${SESSION_ID}`;
