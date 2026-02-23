#!/usr/bin/env python3
"""
Web UI for the playlist downloader with built-in Winamp-style player.
Run this and open http://localhost:8080 in your browser.

Usage:
    python playlist_server.py --no-browser
    python playlist_server.py --port 9000
    python playlist_server.py -o /path/to/music
"""

import argparse
import json
import mimetypes
import platform
import queue
import random
import subprocess as _subprocess
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

from playlist_dl import (
    PLAYLIST_URL,
    DEFAULT_OUTPUT_DIR,
    PAUSE_BETWEEN_TRACKS,
    PAUSE_BETWEEN_BATCHES,
    MAX_RETRIES,
    BATCH_SIZE,
    get_playlist_entries,
    load_progress,
    save_progress,
    download_track,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app_state = {
    "entries": [],
    "completed": set(),
    "failed": set(),
    "output_dir": None,
    "downloading": False,
    "cancel_flag": False,
    "current_id": None,
}

sse_clients: list[queue.Queue] = []
sse_lock = threading.Lock()


def broadcast_sse(event: str, data: dict):
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)


def download_worker(ids: list[str]):
    state = app_state
    state["downloading"] = True
    state["cancel_flag"] = False
    state["failed"] -= set(ids)

    total = len(ids)
    done_count = 0
    fail_list = []
    batch_size = BATCH_SIZE
    total_batches = (total + batch_size - 1) // batch_size

    broadcast_sse("start", {"total": total})

    for batch_num in range(total_batches):
        if state["cancel_flag"]:
            break
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total)
        batch = ids[batch_start:batch_end]

        broadcast_sse("batch", {
            "batch": batch_num + 1,
            "total_batches": total_batches,
            "tracks_in_batch": len(batch),
        })

        for i, vid_id in enumerate(batch):
            if state["cancel_flag"]:
                break
            title = next((e["title"] for e in state["entries"] if e["id"] == vid_id), vid_id)
            state["current_id"] = vid_id

            broadcast_sse("track_start", {
                "id": vid_id, "title": title,
                "index": done_count + 1, "total": total,
            })

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                if state["cancel_flag"]:
                    break
                if attempt > 1:
                    broadcast_sse("retry", {"id": vid_id, "attempt": attempt})
                    time.sleep(10 * attempt)
                if download_track(vid_id, state["output_dir"]):
                    success = True
                    break

            if success:
                done_count += 1
                state["completed"].add(vid_id)
                save_progress(state["output_dir"], state["completed"])
                broadcast_sse("track_done", {
                    "id": vid_id, "title": title,
                    "done": done_count, "total": total,
                })
            else:
                state["failed"].add(vid_id)
                fail_list.append(vid_id)
                broadcast_sse("track_fail", {"id": vid_id, "title": title})

            if not state["cancel_flag"] and i < len(batch) - 1:
                time.sleep(random.uniform(*PAUSE_BETWEEN_TRACKS))

        if not state["cancel_flag"] and batch_num < total_batches - 1:
            pause = random.uniform(*PAUSE_BETWEEN_BATCHES)
            broadcast_sse("batch_pause", {"seconds": round(pause)})
            slept = 0.0
            while slept < pause and not state["cancel_flag"]:
                time.sleep(min(1.0, pause - slept))
                slept += 1.0

    state["downloading"] = False
    state["current_id"] = None
    broadcast_sse("done", {
        "downloaded": done_count,
        "failed": len(fail_list),
        "cancelled": state["cancel_flag"],
    })


def find_file_by_id(video_id: str, extensions: tuple[str, ...]) -> Path | None:
    """Find a file containing [video_id] in its name with one of the given extensions."""
    output_dir = app_state["output_dir"]
    tag = f"[{video_id}]"
    for f in output_dir.iterdir():
        if tag in f.name and f.suffix.lstrip(".") in extensions:
            return f
    return None


def find_audio_file(video_id: str) -> Path | None:
    return find_file_by_id(video_id, ("opus", "m4a", "mp3", "ogg", "webm"))


def find_thumb_file(video_id: str) -> Path | None:
    return find_file_by_id(video_id, ("jpg", "png", "webp"))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = unquote(self.path)
        if path == "/":
            self._serve_html()
        elif path == "/api/tracks":
            self._api_tracks()
        elif path == "/api/progress":
            self._api_sse()
        elif path.startswith("/api/audio/") and path.endswith("/thumb"):
            vid_id = path[len("/api/audio/"):-len("/thumb")]
            self._api_thumb(vid_id)
        elif path.startswith("/api/audio/"):
            vid_id = path[len("/api/audio/"):]
            self._api_audio(vid_id)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/download":
            self._api_download()
        elif self.path == "/api/cancel":
            self._api_cancel()
        elif self.path == "/api/refresh":
            self._api_refresh()
        elif self.path.startswith("/api/open-folder"):
            self._api_open_folder()
        else:
            self.send_error(404)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _api_tracks(self):
        state = app_state
        tracks = []
        for i, e in enumerate(state["entries"]):
            vid_id = e["id"]
            if vid_id == state.get("current_id"):
                status = "downloading"
            elif vid_id in state["completed"]:
                status = "downloaded"
            elif vid_id in state["failed"]:
                status = "failed"
            else:
                status = "pending"
            tracks.append({
                "index": i + 1, "id": vid_id,
                "title": e["title"], "status": status,
            })
        self._json_response({
            "tracks": tracks, "total": len(tracks),
            "downloaded": len(state["completed"]),
            "failed": len(state["failed"]),
            "downloading": state["downloading"],
        })

    def _api_download(self):
        if app_state["downloading"]:
            self._json_response({"error": "Download already in progress"}, 409)
            return
        body = self._read_json_body()
        ids = body.get("ids", [])
        if not ids:
            self._json_response({"error": "No track IDs provided"}, 400)
            return
        ids = [vid for vid in ids if vid not in app_state["completed"]]
        if not ids:
            self._json_response({"error": "All selected tracks already downloaded"}, 400)
            return
        threading.Thread(target=download_worker, args=(ids,), daemon=True).start()
        self._json_response({"started": len(ids)})

    def _api_cancel(self):
        app_state["cancel_flag"] = True
        self._json_response({"ok": True})

    def _api_refresh(self):
        try:
            entries = get_playlist_entries(PLAYLIST_URL)
            app_state["entries"] = entries
            app_state["completed"] = load_progress(app_state["output_dir"])
            self._json_response({"ok": True, "count": len(entries)})
        except Exception as exc:
            self._json_response({"error": str(exc)}, 500)

    def _api_open_folder(self):
        """Open the download folder in the OS file manager, highlighting a file if given."""
        body = self._read_json_body()
        video_id = body.get("id")

        target_file = find_audio_file(video_id) if video_id else None
        output_dir = app_state["output_dir"]

        try:
            system = platform.system()
            is_wsl = system == "Linux" and "microsoft" in platform.uname().release.lower()

            if is_wsl:
                if target_file:
                    win_path = _subprocess.run(
                        ["wslpath", "-w", str(target_file)],
                        capture_output=True, text=True
                    ).stdout.strip()
                else:
                    win_path = _subprocess.run(
                        ["wslpath", "-w", str(output_dir)],
                        capture_output=True, text=True
                    ).stdout.strip()
                # explorer.exe on WSL needs the whole thing as one shell command
                _subprocess.Popen(
                    f'explorer.exe /select,"{win_path}"',
                    shell=True
                )
            elif system == "Windows":
                import os
                if target_file:
                    # Use native Windows API — most reliable way
                    win_path = str(target_file).replace("/", "\\")
                    _subprocess.run(
                        f'explorer /select,"{win_path}"',
                        shell=True
                    )
                else:
                    os.startfile(str(output_dir))
            elif system == "Darwin":
                if target_file:
                    _subprocess.Popen(["open", "-R", str(target_file)])
                else:
                    _subprocess.Popen(["open", str(output_dir)])
            else:
                _subprocess.Popen(["xdg-open", str(output_dir)])

            self._json_response({"ok": True})
        except Exception as exc:
            self._json_response({"error": str(exc)}, 500)

    def _api_audio(self, video_id: str):
        """Serve audio file with HTTP Range support for seeking."""
        fpath = find_audio_file(video_id)
        if not fpath:
            self.send_error(404, "Audio file not found")
            return

        file_size = fpath.stat().st_size
        content_type = mimetypes.guess_type(str(fpath))[0] or "audio/ogg"

        try:
            range_header = self.headers.get("Range")
            if range_header:
                range_spec = range_header.replace("bytes=", "")
                parts = range_spec.split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(fpath, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(fpath, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _api_thumb(self, video_id: str):
        """Serve thumbnail image."""
        fpath = find_thumb_file(video_id)
        if not fpath:
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(fpath))[0] or "image/jpeg"
        data = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def _api_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q: queue.Queue = queue.Queue(maxsize=256)
        with sse_lock:
            sse_clients.append(q)
        try:
            self.wfile.write(b": heartbeat\n\n")
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=15)
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    def _serve_html(self):
        body = HTML_PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Embedded HTML — full page with Winamp player
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Playlist Downloader</title>
<style>
/* ===== RESET & VARS ===== */
:root {
  --bg: #0f0f0f;
  --bg2: #1a1a2e;
  --bg3: #16213e;
  --surface: #1e1e3a;
  --border: #2a2a4a;
  --text: #e0e0e0;
  --text2: #8888aa;
  --accent: #7c3aed;
  --accent2: #a78bfa;
  --green: #22c55e;
  --green-dim: #16a34a33;
  --red: #ef4444;
  --red-dim: #ef444433;
  --yellow: #eab308;
  --yellow-dim: #eab30833;
  --blue: #3b82f6;
  --blue-dim: #3b82f633;
  /* winamp colors */
  --wa-bg: #232323;
  --wa-bg2: #2a2a2a;
  --wa-border: #0a0a0a;
  --wa-border-light: #3a3a3a;
  --wa-lcd: #000000;
  --wa-green: #00ff00;
  --wa-green2: #00cc00;
  --wa-green-dim: #003300;
  --wa-text: #cccccc;
  --wa-highlight: #0078d7;
  --wa-width: 275px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
  min-height: 100vh;
}
a { color: var(--accent2); }

/* ===== PAGE LAYOUT ===== */
.page-layout {
  display: flex;
  min-height: 100vh;
}
.main-content {
  flex: 1;
  min-width: 0;
  padding: 1rem;
  margin-right: var(--wa-width);
  transition: margin-right 0.3s;
}
.main-content.player-hidden {
  margin-right: 0;
}

/* ===== HEADER ===== */
header {
  background: linear-gradient(135deg, var(--bg2), var(--bg3));
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1rem;
}
header h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; }
header h1 span { color: var(--accent2); }

.stats {
  display: flex; gap: 1.5rem; flex-wrap: wrap;
  font-size: 0.9rem; color: var(--text2);
}
.stats .stat { display: flex; align-items: center; gap: 0.4rem; }
.stats .dot {
  width: 10px; height: 10px; border-radius: 50%; display: inline-block;
}
.dot-green { background: var(--green); }
.dot-grey { background: var(--text2); }
.dot-red { background: var(--red); }
.stats .num { color: var(--text); font-weight: 600; }

/* ===== CONTROLS ===== */
.controls {
  display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;
  margin-bottom: 1rem;
}
input[type="text"] {
  background: var(--surface); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 0.5rem 0.75rem; font-size: 0.9rem;
  flex: 1; min-width: 180px;
  outline: none; transition: border-color 0.2s;
}
input[type="text"]:focus { border-color: var(--accent); }
input[type="text"]::placeholder { color: var(--text2); }

.btn {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text);
  font-size: 0.85rem; cursor: pointer; white-space: nowrap;
  transition: all 0.15s;
}
.btn:hover { border-color: var(--accent); background: var(--bg3); }
.btn:active { transform: scale(0.97); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
.btn.primary:hover { background: #6d28d9; }
.btn.primary:disabled { opacity: 0.4; cursor: not-allowed; }
.btn.danger { border-color: var(--red); color: var(--red); }
.btn.danger:hover { background: var(--red-dim); }
.btn.sm { padding: 0.3rem 0.6rem; font-size: 0.8rem; }
.btn-group { display: flex; gap: 0.35rem; }

.filters { display: flex; gap: 0.35rem; flex-wrap: wrap; }
.pill {
  padding: 0.35rem 0.75rem; border-radius: 20px;
  border: 1px solid var(--border); background: transparent;
  color: var(--text2); font-size: 0.8rem; cursor: pointer;
  transition: all 0.15s;
}
.pill:hover { border-color: var(--accent); color: var(--text); }
.pill.active { background: var(--accent); border-color: var(--accent); color: #fff; }

/* ===== PROGRESS ===== */
.progress-panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem; margin-bottom: 1rem;
  display: none;
}
.progress-panel.active { display: block; }
.progress-bar-wrap {
  background: var(--bg); border-radius: 8px; height: 8px;
  overflow: hidden; margin: 0.75rem 0;
}
.progress-bar {
  height: 100%; background: linear-gradient(90deg, var(--accent), var(--green));
  border-radius: 8px; width: 0%; transition: width 0.4s ease;
}
.progress-text {
  font-size: 0.85rem; color: var(--text2);
  display: flex; justify-content: space-between;
}
.progress-log {
  max-height: 160px; overflow-y: auto; font-size: 0.8rem;
  font-family: 'Cascadia Code', 'Fira Code', monospace;
  background: var(--bg); border-radius: 8px; padding: 0.5rem 0.75rem;
  margin-top: 0.75rem; color: var(--text2); line-height: 1.7;
}
.progress-log .ok { color: var(--green); }
.progress-log .fail { color: var(--red); }
.progress-log .info { color: var(--blue); }
.progress-log .warn { color: var(--yellow); }

/* ===== TABLE ===== */
.table-wrap {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; overflow: hidden;
}
table { width: 100%; border-collapse: collapse; }
thead { position: sticky; top: 0; z-index: 2; }
th {
  background: var(--bg3); text-align: left;
  padding: 0.65rem 0.75rem; font-size: 0.8rem;
  font-weight: 600; color: var(--text2);
  border-bottom: 1px solid var(--border);
  user-select: none;
}
th:first-child { width: 42px; text-align: center; }
td {
  padding: 0.5rem 0.75rem; font-size: 0.85rem;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
td:first-child { text-align: center; }
tr:last-child td { border-bottom: none; }
tr:hover { background: #ffffff06; }
tr.row-downloaded { background: var(--green-dim); }
tr.row-downloading { background: var(--blue-dim); }
tr.row-failed { background: var(--red-dim); }
tr.row-playing { background: #7c3aed22; }

.badge {
  display: inline-block; padding: 0.15rem 0.5rem;
  border-radius: 12px; font-size: 0.75rem; font-weight: 600;
}
.badge-downloaded { background: var(--green-dim); color: var(--green); }
.badge-pending { background: #ffffff11; color: var(--text2); }
.badge-downloading { background: var(--blue-dim); color: var(--blue); }
.badge-failed { background: var(--red-dim); color: var(--red); }

input[type="checkbox"] {
  width: 16px; height: 16px; accent-color: var(--accent);
  cursor: pointer;
}
.col-num { width: 55px; color: var(--text2); text-align: center; }
.col-status { width: 110px; }
.col-id { width: 120px; font-family: monospace; font-size: 0.8rem; color: var(--text2); }
.col-play { width: 58px; text-align: center; white-space: nowrap; }

.play-btn, .folder-btn {
  background: none; border: none; cursor: pointer;
  font-size: 1.1rem; padding: 0;
  opacity: 0.7; transition: opacity 0.15s;
}
.play-btn { color: var(--green); }
.folder-btn { color: var(--text2); font-size: 0.95rem; margin-left: 4px; }
.play-btn:hover, .folder-btn:hover { opacity: 1; }
.col-play { width: 58px; text-align: center; white-space: nowrap; }

.scroll-body { max-height: 70vh; overflow-y: auto; }

.loading {
  display: flex; align-items: center; justify-content: center;
  padding: 4rem; color: var(--text2); font-size: 1.1rem; gap: 0.75rem;
}
.spinner {
  width: 24px; height: 24px; border: 3px solid var(--border);
  border-top-color: var(--accent); border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }


/* ============================================================
   WINAMP PLAYER SIDEBAR
   ============================================================ */
.winamp {
  position: fixed;
  right: 0; top: 0;
  width: var(--wa-width);
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--wa-bg);
  border-left: 2px solid var(--wa-border);
  z-index: 100;
  font-family: 'Arial', 'Helvetica', sans-serif;
  user-select: none;
  transition: transform 0.3s;
}
.winamp.hidden {
  transform: translateX(100%);
}

/* Toggle tab on the left edge */
.wa-toggle {
  position: fixed;
  right: var(--wa-width);
  top: 50%;
  transform: translateY(-50%);
  width: 24px; height: 60px;
  background: var(--wa-bg);
  border: 1px solid var(--wa-border-light);
  border-right: none;
  border-radius: 4px 0 0 4px;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  color: var(--wa-green);
  font-size: 0.7rem;
  z-index: 101;
  transition: right 0.3s;
}
.wa-toggle.shifted { right: 0; }
.wa-toggle:hover { background: var(--wa-bg2); }

/* ---- Title bar ---- */
.wa-titlebar {
  background: linear-gradient(180deg, #3a3a5c, #1e1e3a 40%, #2a2a4a);
  padding: 3px 4px;
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--wa-border);
  min-height: 24px;
  cursor: default;
}
.wa-title-text {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #fff;
  text-shadow: 0 0 4px rgba(120,90,255,0.5);
}
.wa-title-buttons { display: flex; gap: 2px; }
.wa-title-btn {
  width: 9px; height: 9px; border-radius: 1px;
  border: 1px solid #555;
  background: #333;
  cursor: pointer;
  font-size: 0;
}
.wa-title-btn:hover { background: #555; }

/* ---- Visualizer ---- */
.wa-visualizer {
  background: #000;
  height: 50px;
  border-bottom: 1px solid var(--wa-border);
  position: relative;
  overflow: hidden;
}
.wa-visualizer canvas {
  width: 100%;
  height: 100%;
  display: block;
}

/* ---- LCD display ---- */
.wa-lcd {
  background: #000;
  padding: 6px 8px;
  border-bottom: 1px solid var(--wa-border);
}
.wa-lcd-title {
  height: 16px;
  overflow: hidden;
  position: relative;
}
.wa-lcd-marquee {
  font-family: 'Courier New', monospace;
  font-size: 11px;
  font-weight: 700;
  color: var(--wa-green);
  white-space: nowrap;
  position: absolute;
  animation: wa-scroll 12s linear infinite;
  text-shadow: 0 0 6px rgba(0,255,0,0.4);
}
@keyframes wa-scroll {
  0% { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}
.wa-lcd-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 4px;
}
.wa-lcd-time {
  font-family: 'Courier New', monospace;
  font-size: 18px;
  font-weight: 700;
  color: var(--wa-green);
  letter-spacing: 1px;
  text-shadow: 0 0 8px rgba(0,255,0,0.3);
}
.wa-lcd-kbps {
  font-family: 'Courier New', monospace;
  font-size: 9px;
  color: var(--wa-green2);
}
.wa-lcd-status {
  font-family: 'Courier New', monospace;
  font-size: 9px;
  color: var(--wa-green2);
}

/* ---- Seek bar ---- */
.wa-seek-wrap {
  padding: 4px 8px;
  background: var(--wa-bg2);
  border-bottom: 1px solid var(--wa-border);
}
.wa-seek {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 6px;
  background: #111;
  border-radius: 3px;
  outline: none;
  cursor: pointer;
  border: 1px solid #333;
}
.wa-seek::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 12px; height: 12px;
  background: linear-gradient(180deg, #888, #444);
  border: 1px solid #222;
  border-radius: 2px;
  cursor: pointer;
}
.wa-seek::-moz-range-thumb {
  width: 12px; height: 12px;
  background: linear-gradient(180deg, #888, #444);
  border: 1px solid #222;
  border-radius: 2px;
  cursor: pointer;
}

/* ---- Transport buttons ---- */
.wa-transport {
  display: flex;
  justify-content: center;
  gap: 2px;
  padding: 6px 8px;
  background: var(--wa-bg);
  border-bottom: 1px solid var(--wa-border);
}
.wa-btn {
  width: 36px; height: 26px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(180deg, #4a4a4a, #2a2a2a 50%, #333);
  border: 1px solid;
  border-color: #555 #222 #222 #555;
  border-radius: 2px;
  cursor: pointer;
  color: #ccc;
  font-size: 11px;
  transition: all 0.08s;
}
.wa-btn:hover { background: linear-gradient(180deg, #5a5a5a, #3a3a3a 50%, #444); color: #fff; }
.wa-btn:active {
  border-color: #222 #555 #555 #222;
  background: linear-gradient(180deg, #2a2a2a, #3a3a3a);
  transform: translateY(1px);
}
.wa-btn.active { color: var(--wa-green); }

/* ---- Volume row ---- */
.wa-volume-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  background: var(--wa-bg2);
  border-bottom: 1px solid var(--wa-border);
  font-size: 9px;
  color: var(--wa-text);
}
.wa-volume-row label { width: 28px; text-align: right; }
.wa-vol-slider {
  -webkit-appearance: none;
  appearance: none;
  flex: 1;
  height: 4px;
  background: #111;
  border-radius: 2px;
  outline: none;
  cursor: pointer;
  border: 1px solid #333;
}
.wa-vol-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 10px; height: 10px;
  background: linear-gradient(180deg, #888, #444);
  border: 1px solid #222;
  border-radius: 2px;
  cursor: pointer;
}
.wa-vol-slider::-moz-range-thumb {
  width: 10px; height: 10px;
  background: linear-gradient(180deg, #888, #444);
  border: 1px solid #222;
  border-radius: 2px;
  cursor: pointer;
}

/* ---- Shuffle/Repeat row ---- */
.wa-modes {
  display: flex;
  justify-content: center;
  gap: 4px;
  padding: 4px 8px;
  background: var(--wa-bg);
  border-bottom: 1px solid var(--wa-border);
}
.wa-mode-btn {
  padding: 2px 10px;
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 2px;
  color: #666;
  cursor: pointer;
  transition: all 0.15s;
}
.wa-mode-btn:hover { color: #999; border-color: #555; }
.wa-mode-btn.on { color: var(--wa-green); border-color: var(--wa-green-dim); background: #0a1a0a; }

/* ---- Playlist panel ---- */
.wa-playlist-header {
  background: linear-gradient(180deg, #2a2a4a, #1a1a3a);
  padding: 3px 8px;
  display: flex; justify-content: space-between; align-items: center;
  border-bottom: 1px solid var(--wa-border);
}
.wa-playlist-header span {
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #aaa;
}
.wa-playlist-count {
  font-size: 9px;
  color: var(--wa-green2);
  font-family: 'Courier New', monospace;
}

.wa-playlist {
  flex: 1;
  overflow-y: auto;
  background: #0a0a0a;
}
.wa-playlist::-webkit-scrollbar { width: 8px; }
.wa-playlist::-webkit-scrollbar-track { background: #111; }
.wa-playlist::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
.wa-playlist::-webkit-scrollbar-thumb:hover { background: #444; }

.wa-pl-item {
  padding: 3px 8px;
  font-size: 11px;
  color: var(--wa-text);
  cursor: pointer;
  display: flex;
  gap: 6px;
  border-bottom: 1px solid #111;
  transition: background 0.1s;
}
.wa-pl-item:hover { background: #1a1a2a; }
.wa-pl-item.current {
  background: #1a1a3a;
  color: #fff;
}
.wa-pl-item.current .wa-pl-num { color: var(--wa-green); }
.wa-pl-num {
  color: #555;
  font-family: 'Courier New', monospace;
  font-size: 10px;
  min-width: 24px;
  text-align: right;
}
.wa-pl-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.wa-pl-remove {
  color: #444;
  cursor: pointer;
  font-size: 10px;
  padding: 0 2px;
}
.wa-pl-remove:hover { color: var(--red); }

/* ---- Responsive ---- */
@media (max-width: 900px) {
  .main-content { margin-right: 0; }
  .winamp { display: none; }
  .wa-toggle { display: none; }
}
@media (max-width: 700px) {
  .col-id { display: none; }
  th:nth-child(6), td:nth-child(6) { display: none; }
  .controls { flex-direction: column; }
}
</style>
</head>
<body>
<div class="page-layout">

<!-- ============ MAIN CONTENT (left) ============ -->
<div class="main-content" id="main-content">
  <header>
    <h1><span>&#9835;</span> Playlist Downloader</h1>
    <div class="stats" id="stats">
      <div class="stat"><span class="dot dot-grey"></span> Total: <span class="num" id="stat-total">-</span></div>
      <div class="stat"><span class="dot dot-green"></span> Downloaded: <span class="num" id="stat-done">-</span></div>
      <div class="stat"><span class="dot dot-grey"></span> Remaining: <span class="num" id="stat-remaining">-</span></div>
      <div class="stat"><span class="dot dot-red"></span> Failed: <span class="num" id="stat-failed">-</span></div>
    </div>
  </header>

  <div class="controls">
    <input type="text" id="search" placeholder="Search tracks...">
    <div class="filters">
      <button class="pill active" data-filter="all">All</button>
      <button class="pill" data-filter="downloaded">Downloaded</button>
      <button class="pill" data-filter="pending">Not Downloaded</button>
      <button class="pill" data-filter="failed">Failed</button>
    </div>
    <div class="btn-group">
      <button class="btn sm" id="btn-sel-all" title="Select all visible">Select All</button>
      <button class="btn sm" id="btn-sel-none" title="Deselect all">None</button>
      <button class="btn sm" id="btn-sel-invert" title="Invert selection">Invert</button>
    </div>
    <button class="btn primary" id="btn-download">Download Selected</button>
    <button class="btn danger" id="btn-cancel" style="display:none">Cancel</button>
    <button class="btn sm" id="btn-refresh" title="Re-fetch playlist from YouTube">Refresh</button>
  </div>

  <div class="progress-panel" id="progress-panel">
    <div class="progress-text">
      <span id="progress-label">Starting...</span>
      <span id="progress-pct">0%</span>
    </div>
    <div class="progress-bar-wrap"><div class="progress-bar" id="progress-bar"></div></div>
    <div class="progress-log" id="progress-log"></div>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th><input type="checkbox" id="check-all"></th>
          <th class="col-num">#</th>
          <th class="col-play"></th>
          <th>Title</th>
          <th class="col-status">Status</th>
          <th class="col-id">Video ID</th>
        </tr>
      </thead>
    </table>
    <div class="scroll-body" id="scroll-body">
      <table>
        <tbody id="tbody"></tbody>
      </table>
      <div class="loading" id="loading">
        <div class="spinner"></div> Loading playlist...
      </div>
    </div>
  </div>
</div>

<!-- ============ WINAMP PLAYER (right sidebar) ============ -->
<div class="winamp" id="winamp">
  <!-- Title bar -->
  <div class="wa-titlebar">
    <span class="wa-title-text">&#9835; Winamp</span>
    <div class="wa-title-buttons">
      <div class="wa-title-btn" id="wa-shade" title="Shade mode"></div>
    </div>
  </div>

  <!-- Visualizer -->
  <div class="wa-visualizer">
    <canvas id="wa-viz-canvas"></canvas>
  </div>

  <!-- LCD display -->
  <div class="wa-lcd">
    <div class="wa-lcd-title">
      <div class="wa-lcd-marquee" id="wa-marquee">No track loaded</div>
    </div>
    <div class="wa-lcd-info">
      <span class="wa-lcd-time" id="wa-time">00:00</span>
      <span class="wa-lcd-status" id="wa-status">Stopped</span>
      <span class="wa-lcd-kbps" id="wa-kbps"></span>
    </div>
  </div>

  <!-- Seek bar -->
  <div class="wa-seek-wrap">
    <input type="range" class="wa-seek" id="wa-seek" min="0" max="1000" value="0">
  </div>

  <!-- Transport controls -->
  <div class="wa-transport">
    <div class="wa-btn" id="wa-prev" title="Previous">&#9198;</div>
    <div class="wa-btn" id="wa-play" title="Play">&#9654;</div>
    <div class="wa-btn" id="wa-pause" title="Pause">&#10074;&#10074;</div>
    <div class="wa-btn" id="wa-stop" title="Stop">&#9632;</div>
    <div class="wa-btn" id="wa-next" title="Next">&#9197;</div>
  </div>

  <!-- Volume -->
  <div class="wa-volume-row">
    <label>VOL</label>
    <input type="range" class="wa-vol-slider" id="wa-volume" min="0" max="100" value="80">
    <span id="wa-vol-val" style="width:24px">80%</span>
  </div>

  <!-- Shuffle / Repeat -->
  <div class="wa-modes">
    <div class="wa-mode-btn" id="wa-shuffle" title="Shuffle">SHUF</div>
    <div class="wa-mode-btn" id="wa-repeat" title="Repeat all">REP</div>
    <div class="wa-mode-btn" id="wa-repeat-one" title="Repeat one">REP1</div>
  </div>

  <!-- Playlist panel -->
  <div class="wa-playlist-header">
    <span>Playlist</span>
    <span class="wa-playlist-count" id="wa-pl-count">0 tracks</span>
  </div>
  <div class="wa-playlist" id="wa-playlist"></div>
</div>

<!-- Toggle tab -->
<div class="wa-toggle" id="wa-toggle" title="Toggle player">&#9835;</div>

</div><!-- /page-layout -->

<script>
(function(){
  const $ = id => document.getElementById(id);

  /* ==================================================================
     DOWNLOAD MANAGER (same as before, adapted)
     ================================================================== */
  let allTracks = [];
  let selected = new Set();
  let filter = 'all';
  let searchTerm = '';
  let downloading = false;

  const tbody = $('tbody');
  const loading = $('loading');
  const searchInput = $('search');
  const progressPanel = $('progress-panel');
  const progressBar = $('progress-bar');
  const progressLabel = $('progress-label');
  const progressPct = $('progress-pct');
  const progressLog = $('progress-log');
  const btnDownload = $('btn-download');
  const btnCancel = $('btn-cancel');

  async function fetchTracks() {
    loading.style.display = 'flex';
    tbody.innerHTML = '';
    try {
      const r = await fetch('/api/tracks');
      const data = await r.json();
      allTracks = data.tracks;
      downloading = data.downloading;
      updateStats();
      renderTable();
    } catch(e) {
      loading.innerHTML = '<span style="color:var(--red)">Failed to load tracks: ' + e.message + '</span>';
    }
  }

  function updateStats() {
    const done = allTracks.filter(t => t.status === 'downloaded').length;
    const failed = allTracks.filter(t => t.status === 'failed').length;
    $('stat-total').textContent = allTracks.length;
    $('stat-done').textContent = done;
    $('stat-remaining').textContent = allTracks.length - done;
    $('stat-failed').textContent = failed;
  }

  function visibleTracks() {
    return allTracks.filter(t => {
      if (filter !== 'all' && t.status !== filter) return false;
      if (searchTerm && !t.title.toLowerCase().includes(searchTerm)) return false;
      return true;
    });
  }

  function renderTable() {
    loading.style.display = 'none';
    const tracks = visibleTracks();
    const fragments = [];
    for (const t of tracks) {
      const checked = selected.has(t.id) ? 'checked' : '';
      let rowClass = t.status !== 'pending' ? 'row-' + t.status : '';
      if (player.currentTrack && player.currentTrack.id === t.id) rowClass += ' row-playing';
      const badgeClass = 'badge-' + t.status;
      const badgeText = t.status === 'downloaded' ? 'Downloaded'
                      : t.status === 'downloading' ? 'Downloading...'
                      : t.status === 'failed' ? 'Failed' : 'Pending';
      const titleEsc = escHtml(t.title);
      const playCell = t.status === 'downloaded'
        ? `<td class="col-play"><button class="play-btn" data-play-id="${t.id}" title="Play">&#9654;</button><button class="folder-btn" data-folder-id="${t.id}" title="Open folder">&#128193;</button></td>`
        : `<td class="col-play"></td>`;
      fragments.push(
        `<tr class="${rowClass}" data-id="${t.id}">` +
        `<td><input type="checkbox" data-id="${t.id}" ${checked}></td>` +
        `<td class="col-num">${t.index}</td>` +
        playCell +
        `<td>${titleEsc}</td>` +
        `<td class="col-status"><span class="badge ${badgeClass}">${badgeText}</span></td>` +
        `<td class="col-id">${t.id}</td>` +
        `</tr>`
      );
    }
    tbody.innerHTML = fragments.join('');
    updateDownloadBtn();
  }

  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function updateDownloadBtn() {
    const count = [...selected].filter(id => {
      const t = allTracks.find(t => t.id === id);
      return t && t.status !== 'downloaded';
    }).length;
    btnDownload.textContent = count > 0 ? `Download Selected (${count})` : 'Download Selected';
    btnDownload.disabled = count === 0 || downloading;
    btnCancel.style.display = downloading ? '' : 'none';
  }

  // Selection via checkbox
  tbody.addEventListener('change', e => {
    if (e.target.type === 'checkbox') {
      const id = e.target.dataset.id;
      e.target.checked ? selected.add(id) : selected.delete(id);
      updateDownloadBtn();
    }
  });

  // Play button + folder button clicks
  tbody.addEventListener('click', e => {
    const playBtn = e.target.closest('.play-btn');
    if (playBtn) {
      e.stopPropagation();
      const id = playBtn.dataset.playId;
      const t = allTracks.find(t => t.id === id);
      if (t) player.playNow(t);
      return;
    }

    const folderBtn = e.target.closest('.folder-btn');
    if (folderBtn) {
      e.stopPropagation();
      fetch('/api/open-folder', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: folderBtn.dataset.folderId}),
      });
      return;
    }

    const tr = e.target.closest('tr');
    if (!tr || e.target.type === 'checkbox') return;
    const cb = tr.querySelector('input[type="checkbox"]');
    if (cb) { cb.checked = !cb.checked; cb.dispatchEvent(new Event('change', {bubbles:true})); }
  });

  // Double-click to play
  tbody.addEventListener('dblclick', e => {
    const tr = e.target.closest('tr');
    if (!tr) return;
    const id = tr.dataset.id;
    const t = allTracks.find(t => t.id === id);
    if (t && t.status === 'downloaded') {
      player.playNow(t);
    }
  });

  $('check-all').addEventListener('change', e => {
    const vis = visibleTracks();
    if (e.target.checked) vis.forEach(t => selected.add(t.id));
    else vis.forEach(t => selected.delete(t.id));
    renderTable();
  });
  $('btn-sel-all').addEventListener('click', () => { visibleTracks().forEach(t => selected.add(t.id)); renderTable(); });
  $('btn-sel-none').addEventListener('click', () => { selected.clear(); renderTable(); });
  $('btn-sel-invert').addEventListener('click', () => {
    visibleTracks().forEach(t => { selected.has(t.id) ? selected.delete(t.id) : selected.add(t.id); });
    renderTable();
  });

  document.querySelectorAll('.pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filter = btn.dataset.filter;
      renderTable();
    });
  });
  searchInput.addEventListener('input', () => { searchTerm = searchInput.value.trim().toLowerCase(); renderTable(); });

  btnDownload.addEventListener('click', async () => {
    const ids = [...selected].filter(id => {
      const t = allTracks.find(t => t.id === id);
      return t && t.status !== 'downloaded';
    });
    if (!ids.length) return;
    try {
      const r = await fetch('/api/download', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ids}),
      });
      const data = await r.json();
      if (data.error) { alert(data.error); return; }
      downloading = true;
      progressPanel.classList.add('active');
      progressLog.innerHTML = '';
      progressBar.style.width = '0%';
      updateDownloadBtn();
    } catch(e) { alert('Failed to start download: ' + e.message); }
  });

  btnCancel.addEventListener('click', async () => {
    if (!confirm('Cancel the current download?')) return;
    await fetch('/api/cancel', {method: 'POST'});
  });

  $('btn-refresh').addEventListener('click', async () => {
    $('btn-refresh').disabled = true;
    $('btn-refresh').textContent = 'Refreshing...';
    try { await fetch('/api/refresh', {method: 'POST'}); await fetchTracks(); }
    finally { $('btn-refresh').disabled = false; $('btn-refresh').textContent = 'Refresh'; }
  });

  // SSE
  function connectSSE() {
    const es = new EventSource('/api/progress');
    es.addEventListener('start', e => { logMsg(`Starting download of ${JSON.parse(e.data).total} tracks...`, 'info'); });
    es.addEventListener('batch', e => { const d = JSON.parse(e.data); logMsg(`Batch ${d.batch}/${d.total_batches} (${d.tracks_in_batch} tracks)`, 'info'); });
    es.addEventListener('track_start', e => {
      const d = JSON.parse(e.data);
      progressLabel.textContent = `Downloading: ${d.title}`;
      const pct = Math.round(((d.index - 1) / d.total) * 100);
      progressBar.style.width = pct + '%'; progressPct.textContent = pct + '%';
      logMsg(`[${d.index}/${d.total}] ${d.title}`, '');
      const t = allTracks.find(t => t.id === d.id);
      if (t) { t.status = 'downloading'; renderRow(d.id); }
    });
    es.addEventListener('track_done', e => {
      const d = JSON.parse(e.data);
      const pct = Math.round((d.done / d.total) * 100);
      progressBar.style.width = pct + '%'; progressPct.textContent = pct + '%';
      progressLabel.textContent = `Downloaded: ${d.title}`;
      logMsg(`  Done: ${d.title}`, 'ok');
      const t = allTracks.find(t => t.id === d.id);
      if (t) { t.status = 'downloaded'; selected.delete(d.id); renderRow(d.id); }
      updateStats(); updateDownloadBtn();
    });
    es.addEventListener('track_fail', e => {
      const d = JSON.parse(e.data);
      logMsg(`  FAILED: ${d.title}`, 'fail');
      const t = allTracks.find(t => t.id === d.id);
      if (t) { t.status = 'failed'; renderRow(d.id); }
      updateStats();
    });
    es.addEventListener('retry', e => { logMsg(`  Retry attempt ${JSON.parse(e.data).attempt}...`, 'warn'); });
    es.addEventListener('batch_pause', e => { logMsg(`Batch complete. Pausing ${JSON.parse(e.data).seconds}s...`, 'info'); });
    es.addEventListener('done', e => {
      const d = JSON.parse(e.data);
      downloading = false;
      progressBar.style.width = '100%'; progressPct.textContent = '100%';
      const msg = d.cancelled ? 'Cancelled.' : `Done! ${d.downloaded} downloaded, ${d.failed} failed.`;
      progressLabel.textContent = msg;
      logMsg(msg, d.failed > 0 ? 'warn' : 'ok');
      updateDownloadBtn(); updateStats();
    });
    es.onerror = () => {};
  }

  function logMsg(text, cls) {
    const div = document.createElement('div');
    if (cls) div.className = cls;
    div.textContent = text;
    progressLog.appendChild(div);
    progressLog.scrollTop = progressLog.scrollHeight;
  }

  function renderRow(id) {
    const t = allTracks.find(t => t.id === id);
    if (!t) return;
    const row = tbody.querySelector(`tr[data-id="${id}"]`);
    if (!row) return;
    let cls = t.status !== 'pending' ? 'row-' + t.status : '';
    if (player.currentTrack && player.currentTrack.id === id) cls += ' row-playing';
    row.className = cls;
    const badge = row.querySelector('.badge');
    if (badge) {
      badge.className = 'badge badge-' + t.status;
      badge.textContent = t.status === 'downloaded' ? 'Downloaded'
                        : t.status === 'downloading' ? 'Downloading...'
                        : t.status === 'failed' ? 'Failed' : 'Pending';
    }
    // Update play button
    const playCell = row.children[2];
    if (t.status === 'downloaded' && !playCell.querySelector('.play-btn')) {
      playCell.innerHTML = `<button class="play-btn" data-play-id="${t.id}" title="Play">&#9654;</button>`;
    }
  }


  /* ==================================================================
     WINAMP PLAYER
     ================================================================== */
  const player = {
    audio: new Audio(),
    audioCtx: null,
    analyser: null,
    source: null,
    playlist: [],       // [{id, title}, ...]
    currentIndex: -1,
    currentTrack: null,
    shuffle: false,
    repeat: false,      // repeat all
    repeatOne: false,
    seeking: false,
    vizAnimId: null,

    init() {
      this.audio.volume = 0.8;
      this.audio.addEventListener('timeupdate', () => this.onTimeUpdate());
      this.audio.addEventListener('ended', () => this.onEnded());
      this.audio.addEventListener('play', () => this.onPlayState());
      this.audio.addEventListener('pause', () => this.onPlayState());
      this.audio.addEventListener('loadedmetadata', () => this.onMeta());

      // Seek bar
      $('wa-seek').addEventListener('input', () => { this.seeking = true; });
      $('wa-seek').addEventListener('change', e => {
        if (this.audio.duration) {
          this.audio.currentTime = (e.target.value / 1000) * this.audio.duration;
        }
        this.seeking = false;
      });

      // Volume
      $('wa-volume').addEventListener('input', e => {
        this.audio.volume = e.target.value / 100;
        $('wa-vol-val').textContent = e.target.value + '%';
      });

      // Transport
      $('wa-prev').addEventListener('click', () => this.prev());
      $('wa-play').addEventListener('click', () => this.play());
      $('wa-pause').addEventListener('click', () => this.togglePause());
      $('wa-stop').addEventListener('click', () => this.stop());
      $('wa-next').addEventListener('click', () => this.next());

      // Modes
      $('wa-shuffle').addEventListener('click', () => {
        this.shuffle = !this.shuffle;
        $('wa-shuffle').classList.toggle('on', this.shuffle);
      });
      $('wa-repeat').addEventListener('click', () => {
        this.repeat = !this.repeat;
        if (this.repeat) this.repeatOne = false;
        $('wa-repeat').classList.toggle('on', this.repeat);
        $('wa-repeat-one').classList.toggle('on', this.repeatOne);
      });
      $('wa-repeat-one').addEventListener('click', () => {
        this.repeatOne = !this.repeatOne;
        if (this.repeatOne) this.repeat = false;
        $('wa-repeat-one').classList.toggle('on', this.repeatOne);
        $('wa-repeat').classList.toggle('on', this.repeat);
      });

      // Toggle sidebar
      $('wa-toggle').addEventListener('click', () => this.toggleSidebar());

      this.initVisualizer();
    },

    toggleSidebar() {
      const wa = $('winamp');
      const main = $('main-content');
      const toggle = $('wa-toggle');
      wa.classList.toggle('hidden');
      main.classList.toggle('player-hidden');
      toggle.classList.toggle('shifted');
      toggle.innerHTML = wa.classList.contains('hidden') ? '&#9835;' : '&#10005;';
    },

    initVisualizer() {
      this.vizCanvas = $('wa-viz-canvas');
      this.vizCtx = this.vizCanvas.getContext('2d');
    },

    ensureAudioContext() {
      if (this.audioCtx) return;
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 128;
      this.source = this.audioCtx.createMediaElementSource(this.audio);
      this.source.connect(this.analyser);
      this.analyser.connect(this.audioCtx.destination);
      this.startVisualizer();
    },

    startVisualizer() {
      const canvas = this.vizCanvas;
      const ctx = this.vizCtx;
      const analyser = this.analyser;
      const bufLen = analyser.frequencyBinCount;
      const dataArr = new Uint8Array(bufLen);

      const draw = () => {
        this.vizAnimId = requestAnimationFrame(draw);
        const w = canvas.width = canvas.clientWidth;
        const h = canvas.height = canvas.clientHeight;
        analyser.getByteFrequencyData(dataArr);

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, w, h);

        const barCount = 32;
        const barWidth = Math.floor(w / barCount) - 1;
        const step = Math.floor(bufLen / barCount);

        for (let i = 0; i < barCount; i++) {
          const val = dataArr[i * step] / 255;
          const barH = val * h;

          // Classic green gradient
          const grad = ctx.createLinearGradient(0, h - barH, 0, h);
          grad.addColorStop(0, '#00ff00');
          grad.addColorStop(0.6, '#00cc00');
          grad.addColorStop(1, '#006600');
          ctx.fillStyle = grad;

          const x = i * (barWidth + 1);
          ctx.fillRect(x, h - barH, barWidth, barH);

          // Peak dot
          if (val > 0.05) {
            ctx.fillStyle = '#00ff66';
            ctx.fillRect(x, h - barH - 3, barWidth, 2);
          }
        }
      };
      draw();
    },

    playNow(track) {
      // Add to playlist if not there, then play it
      const idx = this.playlist.findIndex(t => t.id === track.id);
      if (idx >= 0) {
        this.currentIndex = idx;
      } else {
        this.playlist.push({id: track.id, title: track.title});
        this.currentIndex = this.playlist.length - 1;
      }
      this.loadAndPlay();
    },

    addToQueue(track) {
      if (!this.playlist.find(t => t.id === track.id)) {
        this.playlist.push({id: track.id, title: track.title});
        this.renderPlaylist();
      }
    },

    loadAndPlay() {
      const track = this.playlist[this.currentIndex];
      if (!track) return;
      this.currentTrack = track;
      this.audio.src = `/api/audio/${track.id}`;
      this.audio.load();

      this.ensureAudioContext();
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume();

      this.audio.play();
      $('wa-marquee').textContent = track.title;
      $('wa-status').textContent = 'Playing';
      $('wa-play').classList.add('active');
      this.renderPlaylist();
      renderTable(); // update row-playing highlight
    },

    play() {
      if (this.currentTrack) {
        this.ensureAudioContext();
        if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
        this.audio.play();
      } else if (this.playlist.length) {
        this.currentIndex = 0;
        this.loadAndPlay();
      }
    },

    togglePause() {
      if (this.audio.paused) {
        this.ensureAudioContext();
        this.audio.play();
      } else {
        this.audio.pause();
      }
    },

    stop() {
      this.audio.pause();
      this.audio.currentTime = 0;
      $('wa-status').textContent = 'Stopped';
      $('wa-play').classList.remove('active');
      $('wa-time').textContent = '00:00';
      $('wa-seek').value = 0;
    },

    next() {
      if (!this.playlist.length) return;
      if (this.shuffle) {
        this.currentIndex = Math.floor(Math.random() * this.playlist.length);
      } else {
        this.currentIndex++;
        if (this.currentIndex >= this.playlist.length) {
          this.currentIndex = this.repeat ? 0 : this.playlist.length - 1;
          if (!this.repeat) { this.stop(); return; }
        }
      }
      this.loadAndPlay();
    },

    prev() {
      if (!this.playlist.length) return;
      // If past 3 seconds, restart current track
      if (this.audio.currentTime > 3) {
        this.audio.currentTime = 0;
        return;
      }
      if (this.shuffle) {
        this.currentIndex = Math.floor(Math.random() * this.playlist.length);
      } else {
        this.currentIndex--;
        if (this.currentIndex < 0) this.currentIndex = this.repeat ? this.playlist.length - 1 : 0;
      }
      this.loadAndPlay();
    },

    onEnded() {
      if (this.repeatOne) {
        this.audio.currentTime = 0;
        this.audio.play();
      } else {
        this.next();
      }
    },

    onTimeUpdate() {
      if (this.seeking) return;
      const t = this.audio.currentTime;
      const d = this.audio.duration || 0;
      $('wa-time').textContent = fmtTime(t) + ' / ' + fmtTime(d);
      if (d > 0) $('wa-seek').value = Math.round((t / d) * 1000);
    },

    onPlayState() {
      if (this.audio.paused) {
        $('wa-status').textContent = 'Paused';
        $('wa-play').classList.remove('active');
        $('wa-pause').classList.add('active');
      } else {
        $('wa-status').textContent = 'Playing';
        $('wa-play').classList.add('active');
        $('wa-pause').classList.remove('active');
      }
    },

    onMeta() {
      $('wa-kbps').textContent = 'OPUS';
    },

    renderPlaylist() {
      const el = $('wa-playlist');
      $('wa-pl-count').textContent = this.playlist.length + ' tracks';
      const frags = [];
      for (let i = 0; i < this.playlist.length; i++) {
        const t = this.playlist[i];
        const cls = i === this.currentIndex ? 'wa-pl-item current' : 'wa-pl-item';
        frags.push(
          `<div class="${cls}" data-pli="${i}">` +
          `<span class="wa-pl-num">${i + 1}.</span>` +
          `<span class="wa-pl-title">${escHtml(t.title)}</span>` +
          `<span class="wa-pl-remove" data-rm="${i}" title="Remove">&times;</span>` +
          `</div>`
        );
      }
      el.innerHTML = frags.join('');

      // Scroll current into view
      const cur = el.querySelector('.current');
      if (cur) cur.scrollIntoView({block: 'nearest'});
    },

    removeFromPlaylist(index) {
      this.playlist.splice(index, 1);
      if (index < this.currentIndex) this.currentIndex--;
      else if (index === this.currentIndex) {
        if (this.currentIndex >= this.playlist.length) this.currentIndex = this.playlist.length - 1;
        if (this.playlist.length > 0) this.loadAndPlay();
        else this.stop();
      }
      this.renderPlaylist();
    }
  };

  // Playlist click handlers
  $('wa-playlist').addEventListener('click', e => {
    const rm = e.target.closest('.wa-pl-remove');
    if (rm) {
      player.removeFromPlaylist(parseInt(rm.dataset.rm));
      return;
    }
    const item = e.target.closest('.wa-pl-item');
    if (item) {
      player.currentIndex = parseInt(item.dataset.pli);
      player.loadAndPlay();
    }
  });

  function fmtTime(s) {
    if (!s || !isFinite(s)) return '00:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
  }

  // ---- Init ----
  player.init();
  fetchTracks();
  connectSSE();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Web UI for playlist downloader")
    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-p", "--port", type=int, default=8080,
                        help="Port to serve on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    parser.add_argument("--url", type=str, default=PLAYLIST_URL,
                        help="Override playlist URL")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    app_state["output_dir"] = output_dir

    print(f"Output directory: {output_dir}")
    print(f"Fetching playlist...")
    entries = get_playlist_entries(args.url)
    app_state["entries"] = entries
    app_state["completed"] = load_progress(output_dir)

    print(f"Loaded {len(entries)} tracks, {len(app_state['completed'])} already downloaded.")

    class ThreadedServer(HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

        def process_request(self, request, client_address):
            t = threading.Thread(target=self.process_request_thread,
                                 args=(request, client_address), daemon=True)
            t.start()

        def process_request_thread(self, request, client_address):
            try:
                self.finish_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
            finally:
                self.shutdown_request(request)

    server = ThreadedServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\nServer running at {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
