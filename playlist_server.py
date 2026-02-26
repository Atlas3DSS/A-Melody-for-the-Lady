#!/usr/bin/env python3
"""
Web UI for the playlist downloader with built-in Winamp-style player.
Run this and open http://localhost:5050 in your browser.

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
from urllib.parse import unquote, parse_qs

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

FAILED_FILE_NAME = ".failed_tracks.json"


def load_failed(output_dir: Path) -> set[str]:
    """Load the set of failed video IDs from disk."""
    fpath = output_dir / FAILED_FILE_NAME
    if fpath.exists():
        try:
            data = json.loads(fpath.read_text())
            return set(data) if isinstance(data, list) else set()
        except Exception:
            return set()
    return set()


def save_failed(output_dir: Path, failed: set[str]):
    """Save the set of failed video IDs to disk."""
    fpath = output_dir / FAILED_FILE_NAME
    fpath.write_text(json.dumps(list(failed), indent=2))

try:
    import numpy as np
    from recommender import find_similar, load_discovery_embeddings
    _HAS_RECOMMENDER = True
except ImportError:
    _HAS_RECOMMENDER = False

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
    "loading": False,  # True while fetching playlist from YouTube
    "skip_loading": False,  # Flag to skip YouTube playlist loading
    "playlist_url": PLAYLIST_URL,  # Current playlist URL
    "embeddings": None,
    "embeddings_mtime": 0,
    "discovery_embeddings": None,
    "discovery_mtime": 0,
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


def get_embeddings():
    """Load embeddings from disk, cached and auto-refreshed on file change."""
    if not _HAS_RECOMMENDER:
        return None
    emb_path = app_state["output_dir"] / "embeddings.npz"
    if not emb_path.exists():
        return None
    mtime = emb_path.stat().st_mtime
    if app_state["embeddings"] is None or mtime > app_state["embeddings_mtime"]:
        data = np.load(emb_path, allow_pickle=True)
        app_state["embeddings"] = {
            "ids": list(data["ids"]),
            "titles": list(data["titles"]),
            "vae": data["vae"],
            "clap": data["clap"],
        }
        app_state["embeddings_mtime"] = mtime
    return app_state["embeddings"]


def get_discovery_embeddings():
    """Load discovery embeddings from disk, cached and auto-refreshed."""
    if not _HAS_RECOMMENDER:
        return None
    emb_path = app_state["output_dir"] / "discovery_embeddings.npz"
    if not emb_path.exists():
        return None
    mtime = emb_path.stat().st_mtime
    if app_state["discovery_embeddings"] is None or mtime > app_state["discovery_mtime"]:
        app_state["discovery_embeddings"] = load_discovery_embeddings(
            app_state["output_dir"])
        app_state["discovery_mtime"] = mtime
    return app_state["discovery_embeddings"]


def download_worker(ids: list[str]):
    state = app_state
    state["downloading"] = True
    state["cancel_flag"] = False
    # Clear failed status for tracks we're about to retry
    state["failed"] -= set(ids)
    save_failed(state["output_dir"], state["failed"])

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
                ok, is_unavailable = download_track(vid_id, state["output_dir"])
                if ok:
                    success = True
                    break
                if is_unavailable:
                    # Video unavailable, no point retrying
                    break

            if success:
                done_count += 1
                state["completed"].add(vid_id)
                state["failed"].discard(vid_id)  # Remove from failed if was there
                save_progress(state["output_dir"], state["completed"])
                save_failed(state["output_dir"], state["failed"])
                broadcast_sse("track_done", {
                    "id": vid_id, "title": title,
                    "done": done_count, "total": total,
                })
            else:
                state["failed"].add(vid_id)
                save_failed(state["output_dir"], state["failed"])
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


def scan_downloaded_tracks() -> list[dict]:
    """Scan output directory for downloaded audio files and return track entries."""
    import re
    output_dir = app_state["output_dir"]
    audio_exts = (".mp3", ".opus", ".m4a", ".ogg", ".webm")
    tracks = []
    seen_ids = set()

    # Pattern to extract video ID from filename: "Title [VIDEO_ID].ext"
    id_pattern = re.compile(r'\[([a-zA-Z0-9_-]{11})\]')

    for f in output_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in audio_exts:
            continue
        match = id_pattern.search(f.stem)
        if match:
            vid_id = match.group(1)
            if vid_id in seen_ids:
                continue
            seen_ids.add(vid_id)
            # Extract title (everything before the [ID])
            title = f.stem[:match.start()].strip()
            if not title:
                title = vid_id
            tracks.append({"id": vid_id, "title": title})

    return tracks


def find_audio_file(video_id: str) -> Path | None:
    return find_file_by_id(video_id, ("mp3", "opus", "m4a", "ogg", "webm"))


def find_thumb_file(video_id: str) -> Path | None:
    return find_file_by_id(video_id, ("jpg", "png", "webp"))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        raw = unquote(self.path)
        path = raw.split("?")[0]
        qs = raw.split("?", 1)[1] if "?" in raw else ""
        if path == "/":
            self._serve_html()
        elif path == "/api/tracks":
            self._api_tracks()
        elif path == "/api/progress":
            self._api_sse()
        elif path.startswith("/api/similar/"):
            vid_id = path[len("/api/similar/"):]
            self._api_similar(vid_id, qs)
        elif path == "/api/search-semantic":
            self._api_semantic_search(qs)
        elif path == "/api/embeddings-status":
            self._api_embeddings_status()
        elif path.startswith("/api/audio/") and path.endswith("/thumb"):
            vid_id = path[len("/api/audio/"):-len("/thumb")]
            self._api_thumb(vid_id)
        elif path.startswith("/api/audio/"):
            vid_id = path[len("/api/audio/"):]
            self._api_audio(vid_id)
        elif path == "/api/config":
            self._api_config()
        elif path == "/api/tags":
            self._api_tags()
        elif path.startswith("/api/playlist/"):
            seed_id = path[len("/api/playlist/"):]
            self._api_playlist(seed_id, qs)
        elif path == "/api/duplicates":
            self._api_duplicates(qs)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/download":
            self._api_download()
        elif self.path == "/api/cancel":
            self._api_cancel()
        elif self.path == "/api/refresh":
            self._api_refresh()
        elif self.path == "/api/skip-loading":
            self._api_skip_loading()
        elif self.path == "/api/clear-failed":
            self._api_clear_failed()
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

    def _api_config(self):
        """Return current config including playlist URL."""
        self._json_response({
            "playlist_url": app_state["playlist_url"],
        })

    def _api_tracks(self):
        state = app_state
        tracks = []

        # If playlist is loaded, use it; otherwise scan directory for downloaded files
        if state["entries"]:
            # Use full playlist
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
        else:
            # Playlist not loaded yet - scan directory for downloaded tracks
            scanned = scan_downloaded_tracks()
            for i, e in enumerate(scanned):
                vid_id = e["id"]
                tracks.append({
                    "index": i + 1, "id": vid_id,
                    "title": e["title"], "status": "downloaded",
                })

        self._json_response({
            "tracks": tracks, "total": len(tracks),
            "downloaded": len(state["completed"]),
            "failed": len(state["failed"]),
            "downloading": state["downloading"],
            "loading": state.get("loading", False),
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
        body = self._read_json_body()
        url = body.get("url", app_state["playlist_url"])
        if url:
            app_state["playlist_url"] = url

        def fetch_async():
            app_state["loading"] = True
            broadcast_sse("playlist_loading", {})
            try:
                entries = get_playlist_entries(url)
                app_state["entries"] = entries
                app_state["completed"] = load_progress(app_state["output_dir"])
                app_state["loading"] = False
                broadcast_sse("playlist_loaded", {"count": len(entries)})
            except Exception as e:
                app_state["loading"] = False
                broadcast_sse("playlist_error", {"error": str(e)})

        threading.Thread(target=fetch_async, daemon=True).start()
        self._json_response({"ok": True, "message": "Fetching playlist..."})

    def _api_skip_loading(self):
        """Skip waiting for YouTube playlist and just use local files."""
        app_state["skip_loading"] = True
        app_state["loading"] = False
        broadcast_sse("playlist_skipped", {})
        self._json_response({"ok": True})

    def _api_clear_failed(self):
        """Clear all failed tracks so they can be retried."""
        count = len(app_state["failed"])
        app_state["failed"].clear()
        save_failed(app_state["output_dir"], app_state["failed"])
        self._json_response({"ok": True, "cleared": count})

    def _api_open_folder(self):
        """Open the download folder in the OS file manager, highlighting a file if given."""
        body = self._read_json_body()
        video_id = body.get("id")

        target_file = find_audio_file(video_id) if video_id else None
        output_dir = app_state["output_dir"]

        try:
            # WSL — use explorer.exe
            if "microsoft" in platform.uname().release.lower():
                if target_file:
                    win_path = _subprocess.run(
                        ["wslpath", "-w", str(target_file)],
                        capture_output=True, text=True
                    ).stdout.strip()
                    _subprocess.Popen(["explorer.exe", "/select,", win_path])
                else:
                    win_path = _subprocess.run(
                        ["wslpath", "-w", str(output_dir)],
                        capture_output=True, text=True
                    ).stdout.strip()
                    _subprocess.Popen(["explorer.exe", win_path])
            elif platform.system() == "Darwin":
                if target_file:
                    _subprocess.Popen(["open", "-R", str(target_file)])
                else:
                    _subprocess.Popen(["open", str(output_dir)])
            elif platform.system() == "Windows":
                if target_file:
                    _subprocess.Popen(["explorer.exe", "/select,", str(target_file)])
                else:
                    _subprocess.Popen(["explorer.exe", str(output_dir)])
            else:
                _subprocess.Popen(["xdg-open", str(output_dir)])

            self._json_response({"ok": True})
        except Exception as exc:
            self._json_response({"error": str(exc)}, 500)

    def _api_similar(self, video_id: str, qs: str):
        """Return top-N similar tracks by blended embedding similarity."""
        if not _HAS_RECOMMENDER:
            self._json_response(
                {"error": "Recommender not available (numpy/recommender.py missing)"}, 501)
            return
        params = parse_qs(qs)
        n = int(params.get("n", ["20"])[0])
        alpha = float(params.get("alpha", ["0.5"])[0])
        emb_data = get_embeddings()
        if emb_data is None:
            self._json_response(
                {"error": "No embeddings found. Run 'python recommender.py embed' first."}, 404)
            return
        disc_data = get_discovery_embeddings()
        results = find_similar(emb_data, video_id, n=n, alpha=alpha,
                               discovery_data=disc_data)
        query_title = ""
        if video_id in emb_data["ids"]:
            query_title = emb_data["titles"][emb_data["ids"].index(video_id)]
        elif disc_data and video_id in disc_data["ids"]:
            query_title = disc_data["titles"][disc_data["ids"].index(video_id)]
        self._json_response({
            "query_id": video_id, "query_title": query_title, "results": results,
        })

    def _api_semantic_search(self, qs: str):
        """Search tracks by text description using CLAP text encoder."""
        params = parse_qs(qs)
        q = params.get("q", [""])[0]
        n = int(params.get("n", ["20"])[0])
        if not q:
            self._json_response({"error": "No query provided"}, 400)
            return
        emb_data = get_embeddings()
        if emb_data is None:
            self._json_response(
                {"error": "No embeddings found. Run 'python recommender.py embed' first."}, 404)
            return
        try:
            from recommender import search_by_text, load_clap
            if app_state.get("clap_model") is None:
                app_state["clap_model"], app_state["clap_processor"] = load_clap(
                    device="cuda")
            results = search_by_text(
                emb_data, app_state["clap_model"], app_state["clap_processor"],
                q, n=n, device="cuda")
            self._json_response({"query": q, "results": results})
        except ImportError:
            self._json_response(
                {"error": "Semantic search requires torch and transformers. Activate the dev venv."}, 501)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_embeddings_status(self):
        """Return status of the embeddings file."""
        emb_data = get_embeddings() if _HAS_RECOMMENDER else None
        if emb_data:
            self._json_response({"available": True, "count": len(emb_data["ids"])})
        else:
            self._json_response({"available": False, "count": 0})

    def _api_tags(self):
        """Return all track tags from track_tags.json."""
        tags_file = OUTPUT_DIR / "track_tags.json"
        if tags_file.exists():
            try:
                tags = json.loads(tags_file.read_text())
                self._json_response({"available": True, "tags": tags})
            except Exception:
                self._json_response({"available": False, "tags": {}})
        else:
            self._json_response({"available": False, "tags": {}})

    def _api_playlist(self, seed_id: str, qs: str):
        """Generate smart playlist from seed track."""
        if not _HAS_RECOMMENDER:
            self._json_response({"error": "Recommender not available"}, 400)
            return

        # Parse query params
        params = {}
        for part in qs.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v

        length = int(params.get("length", 60))
        drift = float(params.get("drift", 0.3))

        emb_data = get_embeddings()
        if not emb_data:
            self._json_response({"error": "Embeddings not loaded"}, 400)
            return

        from recommender import generate_playlist
        playlist = generate_playlist(emb_data, seed_id, length_minutes=length, drift=drift)

        if not playlist:
            self._json_response({"error": "Seed track not found"}, 404)
            return

        self._json_response({"playlist": playlist})

    def _api_duplicates(self, qs: str):
        """Find duplicate tracks."""
        if not _HAS_RECOMMENDER:
            self._json_response({"error": "Recommender not available"}, 400)
            return

        params = {}
        for part in qs.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v

        threshold = float(params.get("threshold", 0.98))
        limit = int(params.get("limit", 50))

        emb_data = get_embeddings()
        if not emb_data:
            self._json_response({"error": "Embeddings not loaded"}, 400)
            return

        from recommender import find_duplicates
        duplicates = find_duplicates(emb_data, threshold=threshold)

        results = []
        for d in duplicates[:limit]:
            results.append({
                "id1": d[0], "id2": d[1],
                "similarity": round(d[2], 4),
                "title1": d[3], "title2": d[4]
            })

        self._json_response({"duplicates": results, "total": len(duplicates)})

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
/* ===== RESET & VARS - NEON BANANA THEME ===== */
:root {
  --bg: #0a0a12;
  --bg2: #12121f;
  --bg3: #1a1a2e;
  --surface: #15152a;
  --border: #2a2a4a;
  --text: #f0f0f0;
  --text2: #9090b0;
  --accent: #f0e130;
  --accent2: #ffe066;
  --accent-glow: rgba(240, 225, 48, 0.4);
  --neon-pink: #ff00ff;
  --neon-cyan: #00ffff;
  --neon-green: #39ff14;
  --green: #39ff14;
  --green-dim: rgba(57, 255, 20, 0.15);
  --red: #ff3366;
  --red-dim: rgba(255, 51, 102, 0.15);
  --yellow: #f0e130;
  --yellow-dim: rgba(240, 225, 48, 0.15);
  --blue: #00d4ff;
  --blue-dim: rgba(0, 212, 255, 0.15);
  /* winamp neon colors */
  --wa-bg: #0d0d15;
  --wa-bg2: #151522;
  --wa-border: #000000;
  --wa-border-light: #2a2a4a;
  --wa-lcd: #050508;
  --wa-green: #39ff14;
  --wa-green2: #00ff88;
  --wa-green-dim: #0a2010;
  --wa-yellow: #f0e130;
  --wa-pink: #ff00ff;
  --wa-cyan: #00ffff;
  --wa-text: #e0e0e0;
  --wa-highlight: #f0e130;
  --wa-width: 320px;
  --glow-yellow: 0 0 20px rgba(240, 225, 48, 0.6), 0 0 40px rgba(240, 225, 48, 0.3);
  --glow-green: 0 0 15px rgba(57, 255, 20, 0.5), 0 0 30px rgba(57, 255, 20, 0.2);
  --glow-pink: 0 0 15px rgba(255, 0, 255, 0.5), 0 0 30px rgba(255, 0, 255, 0.2);
  --glow-cyan: 0 0 15px rgba(0, 255, 255, 0.5), 0 0 30px rgba(0, 255, 255, 0.2);
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
  position: relative;
  overflow: hidden;
}
header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, var(--accent-glow) 0%, transparent 50%);
  opacity: 0.1;
  pointer-events: none;
}
header h1 {
  font-size: 1.6rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  text-shadow: var(--glow-yellow);
}
header h1 span {
  color: var(--accent);
  text-shadow: var(--glow-yellow);
}

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
  padding: 0.5rem 1rem; border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface); color: var(--text);
  font-size: 0.85rem; cursor: pointer; white-space: nowrap;
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}
.btn::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
  transition: left 0.5s;
}
.btn:hover::before { left: 100%; }
.btn:hover {
  border-color: var(--accent);
  background: var(--bg3);
  box-shadow: 0 0 15px var(--accent-glow);
}
.btn:active { transform: scale(0.97); }
.btn.primary {
  background: linear-gradient(135deg, var(--accent), #d4c520);
  border-color: var(--accent);
  color: #000;
  font-weight: 700;
  text-shadow: none;
}
.btn.primary:hover {
  box-shadow: var(--glow-yellow);
  background: linear-gradient(135deg, #ffe066, var(--accent));
}
.btn.primary:disabled { opacity: 0.4; cursor: not-allowed; box-shadow: none; }
.btn.danger { border-color: var(--red); color: var(--red); }
.btn.danger:hover { background: var(--red-dim); box-shadow: var(--glow-pink); }
.btn.neon-green { border-color: var(--neon-green); color: var(--neon-green); }
.btn.neon-green:hover { box-shadow: var(--glow-green); background: var(--green-dim); }
.btn.neon-cyan { border-color: var(--neon-cyan); color: var(--neon-cyan); }
.btn.neon-cyan:hover { box-shadow: var(--glow-cyan); background: var(--blue-dim); }
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
.col-play { width: 84px; text-align: center; white-space: nowrap; }

.play-btn, .folder-btn {
  background: none; border: none; cursor: pointer;
  font-size: 1.1rem; padding: 0;
  opacity: 0.7; transition: opacity 0.15s;
}
.play-btn { color: var(--green); }
.folder-btn { color: var(--text2); font-size: 0.95rem; margin-left: 4px; }
.play-btn:hover, .folder-btn:hover { opacity: 1; }
.col-play { width: 84px; text-align: center; white-space: nowrap; }

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

/* ---- Title bar - Neon ---- */
.wa-titlebar {
  background: linear-gradient(180deg, #2a2a4a 0%, #15152a 40%, #1a1a35 100%);
  padding: 4px 6px;
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid #000;
  min-height: 28px;
  cursor: default;
  position: relative;
}
.wa-titlebar::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--wa-yellow), transparent);
  opacity: 0.3;
}
.wa-title-text {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--wa-yellow);
  text-shadow: 0 0 10px rgba(240, 225, 48, 0.6);
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

/* ---- Visualizer - Neon EQ ---- */
.wa-visualizer {
  background: linear-gradient(180deg, #000 0%, #0a0a15 100%);
  height: 100px;
  border-bottom: 1px solid var(--wa-border);
  position: relative;
  overflow: hidden;
}
.wa-visualizer::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background:
    repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.3) 3px, rgba(0,0,0,0.3) 4px),
    repeating-linear-gradient(90deg, transparent, transparent 3px, rgba(0,0,0,0.2) 3px, rgba(0,0,0,0.2) 4px);
  pointer-events: none;
  z-index: 1;
}
.wa-visualizer canvas {
  width: 100%;
  height: 100%;
  display: block;
  position: relative;
  z-index: 0;
}

/* ---- LCD display - Neon ---- */
.wa-lcd {
  background: linear-gradient(180deg, #050510 0%, #0a0a18 100%);
  padding: 8px 10px;
  border-bottom: 1px solid var(--wa-border);
  position: relative;
}
.wa-lcd::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse at center top, rgba(240, 225, 48, 0.05) 0%, transparent 60%);
  pointer-events: none;
}
.wa-lcd-title {
  height: 18px;
  overflow: hidden;
  position: relative;
}
.wa-lcd-marquee {
  font-family: 'Courier New', 'Consolas', monospace;
  font-size: 12px;
  font-weight: 700;
  color: var(--wa-yellow);
  white-space: nowrap;
  position: absolute;
  animation: wa-scroll 12s linear infinite;
  text-shadow: 0 0 10px rgba(240, 225, 48, 0.6), 0 0 20px rgba(240, 225, 48, 0.3);
}
@keyframes wa-scroll {
  0% { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}
.wa-lcd-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 6px;
}
.wa-lcd-time {
  font-family: 'Courier New', 'Consolas', monospace;
  font-size: 22px;
  font-weight: 700;
  color: var(--wa-yellow);
  letter-spacing: 2px;
  text-shadow: 0 0 15px rgba(240, 225, 48, 0.5), 0 0 30px rgba(240, 225, 48, 0.2);
}
.wa-lcd-kbps {
  font-family: 'Courier New', monospace;
  font-size: 10px;
  color: var(--neon-cyan);
  text-shadow: 0 0 8px rgba(0, 255, 255, 0.4);
}
.wa-lcd-status {
  font-family: 'Courier New', monospace;
  font-size: 10px;
  color: var(--neon-green);
  text-shadow: 0 0 8px rgba(57, 255, 20, 0.4);
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

/* ===== SIMILAR BUTTON ===== */
.similar-btn {
  background: none; border: none; cursor: pointer;
  font-size: 0.85rem; padding: 0; margin-left: 4px;
  opacity: 0.5; transition: opacity 0.15s;
  color: var(--accent2);
}
.similar-btn:hover { opacity: 1; }

/* ===== MODAL ===== */
.modal-overlay {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.75); z-index: 200;
  display: none; align-items: center; justify-content: center;
}
.modal-overlay.active { display: flex; }
.modal {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 12px; width: 620px; max-width: 92vw;
  max-height: 80vh; display: flex; flex-direction: column;
  overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.modal-header {
  padding: 1rem 1.25rem; border-bottom: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
  background: var(--bg3);
}
.modal-header h2 {
  font-size: 0.95rem; font-weight: 600;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  flex: 1; margin-right: 1rem;
}
.modal-close {
  cursor: pointer; background: none; border: none;
  color: var(--text2); font-size: 1.3rem; padding: 0 4px;
  transition: color 0.15s;
}
.modal-close:hover { color: var(--text); }
.modal-body { flex: 1; overflow-y: auto; padding: 0; }
.modal-loading {
  display: flex; align-items: center; justify-content: center;
  padding: 3rem; color: var(--text2); gap: 0.5rem;
}
.modal-error { padding: 2rem; text-align: center; color: var(--red); }

.sim-result {
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.55rem 1rem; border-bottom: 1px solid var(--border);
  transition: background 0.1s; cursor: default;
}
.sim-result:hover { background: #ffffff08; }
.sim-result:last-child { border-bottom: none; }
.sim-rank { color: var(--text2); font-size: 0.8rem; width: 26px; text-align: right; flex-shrink: 0; }
.sim-score-bar {
  width: 50px; height: 6px; background: var(--bg);
  border-radius: 3px; overflow: hidden; flex-shrink: 0;
}
.sim-score-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, var(--accent), var(--green));
}
.sim-score {
  font-family: monospace; font-size: 0.75rem; color: var(--accent2);
  width: 40px; text-align: center; flex-shrink: 0;
}
.sim-title {
  flex: 1; font-size: 0.85rem;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.sim-actions { display: flex; gap: 4px; flex-shrink: 0; }
.sim-actions button {
  background: none; border: 1px solid var(--border); border-radius: 4px;
  color: var(--text2); cursor: pointer; font-size: 0.75rem;
  padding: 2px 6px; transition: all 0.15s;
}
.sim-actions button:hover { border-color: var(--accent); color: var(--text); background: var(--surface); }

/* ===== SEMANTIC SEARCH ===== */
.semantic-row {
  display: flex; gap: 0.5rem; align-items: center;
  margin-bottom: 1rem;
}
.semantic-row input[type="text"] { flex: 1; }

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
    <input type="text" id="playlist-url" placeholder="YouTube playlist URL..." style="flex: 2; min-width: 280px;">
    <button class="btn primary" id="btn-load-playlist" title="Fetch playlist from YouTube">Load Playlist</button>
  </div>
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
    <button class="btn sm" id="btn-clear-failed" title="Reset failed tracks to pending so they can be retried">Clear Failed</button>
  </div>

  <div class="semantic-row" id="semantic-row" style="display:none">
    <input type="text" id="semantic-input" placeholder="Describe a vibe... (e.g. &quot;dark techno heavy bass&quot;, &quot;melodic ambient&quot;)">
    <button class="btn sm" id="btn-semantic">Semantic Search</button>
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
        <div class="spinner"></div> <span id="loading-text">Loading playlist...</span>
        <button class="btn sm" id="btn-skip-loading" style="margin-left: 1rem; display: none;">Use Local Files</button>
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

<!-- Similar tracks modal -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal">
    <div class="modal-header">
      <h2 id="modal-title">Similar Tracks</h2>
      <button class="modal-close" id="modal-close">&times;</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
  </div>
</div>

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

  let isLoading = false;
  const btnSkipLoading = $('btn-skip-loading');
  const playlistUrlInput = $('playlist-url');
  const btnLoadPlaylist = $('btn-load-playlist');

  async function fetchTracks() {
    try {
      const r = await fetch('/api/tracks');
      const data = await r.json();
      allTracks = data.tracks;
      downloading = data.downloading;
      isLoading = data.loading;
      updateStats();

      // Always render tracks if we have any (even while still loading from YouTube)
      if (allTracks.length > 0) {
        renderTable();
        if (isLoading) {
          // Show a note that we're still loading more, with skip button
          loading.style.display = 'flex';
          loading.querySelector('.spinner').style.display = '';
          $('loading-text').textContent = 'Loading full playlist from YouTube...';
          btnSkipLoading.style.display = '';
        } else {
          loading.style.display = 'none';
          btnSkipLoading.style.display = 'none';
        }
      } else if (isLoading) {
        // No tracks yet and still loading - show skip button
        loading.style.display = 'flex';
        loading.querySelector('.spinner').style.display = '';
        $('loading-text').textContent = 'Fetching playlist from YouTube...';
        btnSkipLoading.style.display = '';
        tbody.innerHTML = '';
      } else {
        // Not loading and no tracks - empty state
        loading.style.display = 'flex';
        loading.querySelector('.spinner').style.display = 'none';
        $('loading-text').textContent = 'No local tracks found. Enter a playlist URL and click "Load Playlist" to fetch from YouTube.';
        $('loading-text').style.color = 'var(--text2)';
        btnSkipLoading.style.display = 'none';
        tbody.innerHTML = '';
      }
    } catch(e) {
      loading.querySelector('.spinner').style.display = 'none';
      $('loading-text').textContent = 'Failed to load tracks: ' + e.message;
      $('loading-text').style.color = 'var(--red)';
      btnSkipLoading.style.display = 'none';
    }
  }

  btnSkipLoading.addEventListener('click', async () => {
    btnSkipLoading.disabled = true;
    btnSkipLoading.textContent = 'Skipping...';
    await fetch('/api/skip-loading', {method: 'POST'});
    isLoading = false;
    await fetchTracks();
    btnSkipLoading.disabled = false;
    btnSkipLoading.textContent = 'Use Local Files';
  });

  // Fetch config to populate playlist URL field
  async function fetchConfig() {
    try {
      const r = await fetch('/api/config');
      const data = await r.json();
      playlistUrlInput.value = data.playlist_url || '';
    } catch(e) {
      console.error('Failed to fetch config:', e);
    }
  }

  // Load Playlist button - fetch from YouTube
  btnLoadPlaylist.addEventListener('click', async () => {
    const url = playlistUrlInput.value.trim();
    if (!url) {
      alert('Please enter a playlist URL');
      return;
    }
    btnLoadPlaylist.disabled = true;
    btnLoadPlaylist.textContent = 'Loading...';
    try {
      await fetch('/api/refresh', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url}),
      });
      // The SSE events will handle updating the UI
    } catch(e) {
      alert('Failed to start playlist fetch: ' + e.message);
      btnLoadPlaylist.disabled = false;
      btnLoadPlaylist.textContent = 'Load Playlist';
    }
  });

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
    // Only hide loading if playlist is fully loaded
    if (!isLoading) {
      loading.style.display = 'none';
    }
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
        ? `<td class="col-play"><button class="play-btn" data-play-id="${t.id}" title="Play">&#9654;</button><button class="folder-btn" data-folder-id="${t.id}" title="Open folder">&#128193;</button><button class="similar-btn" data-similar-id="${t.id}" title="Find similar">&#8776;</button></td>`
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

    const simBtn = e.target.closest('.similar-btn');
    if (simBtn) {
      e.stopPropagation();
      fetchSimilar(simBtn.dataset.similarId);
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

  $('btn-clear-failed').addEventListener('click', async () => {
    const failedCount = allTracks.filter(t => t.status === 'failed').length;
    if (failedCount === 0) { alert('No failed tracks to clear.'); return; }
    if (!confirm(`Reset ${failedCount} failed track(s) to pending?`)) return;
    $('btn-clear-failed').disabled = true;
    try {
      await fetch('/api/clear-failed', {method: 'POST'});
      // Update local state
      allTracks.forEach(t => { if (t.status === 'failed') t.status = 'pending'; });
      renderTable();
      updateStats();
    } finally {
      $('btn-clear-failed').disabled = false;
    }
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
    es.addEventListener('playlist_loading', e => {
      isLoading = true;
      loading.style.display = 'flex';
      loading.querySelector('.spinner').style.display = '';
      $('loading-text').textContent = 'Fetching playlist from YouTube...';
      $('loading-text').style.color = '';
      btnSkipLoading.style.display = '';
      btnLoadPlaylist.disabled = true;
      btnLoadPlaylist.textContent = 'Loading...';
    });
    es.addEventListener('playlist_loaded', e => {
      isLoading = false;
      loading.style.display = 'none';
      btnLoadPlaylist.disabled = false;
      btnLoadPlaylist.textContent = 'Load Playlist';
      fetchTracks();
    });
    es.addEventListener('playlist_error', e => {
      isLoading = false;
      const d = JSON.parse(e.data);
      // If we have downloaded tracks, just show warning; otherwise show error
      loading.querySelector('.spinner').style.display = 'none';
      btnSkipLoading.style.display = 'none';
      btnLoadPlaylist.disabled = false;
      btnLoadPlaylist.textContent = 'Load Playlist';
      if (allTracks.length > 0) {
        $('loading-text').textContent = 'Could not fetch full playlist: ' + d.error;
        $('loading-text').style.color = 'var(--yellow)';
      } else {
        $('loading-text').textContent = 'Failed to fetch playlist: ' + d.error;
        $('loading-text').style.color = 'var(--red)';
      }
      // Refetch to show local tracks
      fetchTracks();
    });
    es.addEventListener('playlist_skipped', () => {
      isLoading = false;
      loading.style.display = 'none';
      btnSkipLoading.style.display = 'none';
      btnLoadPlaylist.disabled = false;
      btnLoadPlaylist.textContent = 'Load Playlist';
      fetchTracks();
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
      playCell.innerHTML = `<button class="play-btn" data-play-id="${t.id}" title="Play">&#9654;</button><button class="folder-btn" data-folder-id="${t.id}" title="Open folder">&#128193;</button><button class="similar-btn" data-similar-id="${t.id}" title="Find similar">&#8776;</button>`;
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
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.7;
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

      // Peak hold values for each bar
      const peaks = new Float32Array(64).fill(0);
      const peakDecay = 0.97;
      const peakDropSpeed = 0.015;

      // Color palette - neon banana
      const colors = {
        low: { r: 255, g: 0, b: 255 },      // magenta for bass
        mid: { r: 240, g: 225, b: 48 },     // yellow for mids
        high: { r: 0, g: 255, b: 255 },     // cyan for highs
        peak: { r: 255, g: 255, b: 255 }    // white peaks
      };

      const draw = () => {
        this.vizAnimId = requestAnimationFrame(draw);
        const w = canvas.width = canvas.clientWidth * 2; // 2x for retina
        const h = canvas.height = canvas.clientHeight * 2;
        canvas.style.width = canvas.clientWidth + 'px';
        canvas.style.height = canvas.clientHeight + 'px';

        analyser.getByteFrequencyData(dataArr);

        // Clear with subtle fade for trail effect
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fillRect(0, 0, w, h);

        const barCount = 48;
        const gap = 2;
        const barWidth = Math.floor((w - gap * barCount) / barCount);
        const step = Math.floor(bufLen / barCount);

        for (let i = 0; i < barCount; i++) {
          // Average nearby frequencies for smoother visualization
          let sum = 0;
          for (let j = 0; j < step; j++) {
            sum += dataArr[i * step + j] || 0;
          }
          const val = (sum / step) / 255;
          const barH = val * h * 0.9;

          // Update peak
          if (val > peaks[i]) {
            peaks[i] = val;
          } else {
            peaks[i] = Math.max(peaks[i] * peakDecay - peakDropSpeed, 0);
          }

          const x = i * (barWidth + gap);
          const freq = i / barCount; // 0 = low, 1 = high

          // Create gradient based on frequency range
          const grad = ctx.createLinearGradient(x, h, x, h - barH);

          if (freq < 0.33) {
            // Bass - magenta to pink
            grad.addColorStop(0, 'rgba(255, 0, 128, 0.9)');
            grad.addColorStop(0.5, 'rgba(255, 0, 255, 0.8)');
            grad.addColorStop(1, 'rgba(255, 100, 255, 0.6)');
          } else if (freq < 0.66) {
            // Mids - yellow to orange
            grad.addColorStop(0, 'rgba(255, 180, 0, 0.9)');
            grad.addColorStop(0.5, 'rgba(240, 225, 48, 0.8)');
            grad.addColorStop(1, 'rgba(255, 255, 100, 0.6)');
          } else {
            // Highs - cyan to white
            grad.addColorStop(0, 'rgba(0, 200, 255, 0.9)');
            grad.addColorStop(0.5, 'rgba(0, 255, 255, 0.8)');
            grad.addColorStop(1, 'rgba(150, 255, 255, 0.6)');
          }

          ctx.fillStyle = grad;

          // Draw bar with rounded top
          const radius = Math.min(barWidth / 2, 4);
          ctx.beginPath();
          ctx.moveTo(x, h);
          ctx.lineTo(x, h - barH + radius);
          ctx.arcTo(x, h - barH, x + radius, h - barH, radius);
          ctx.arcTo(x + barWidth, h - barH, x + barWidth, h - barH + radius, radius);
          ctx.lineTo(x + barWidth, h);
          ctx.fill();

          // Glow effect
          ctx.shadowBlur = 15;
          ctx.shadowColor = freq < 0.33 ? '#ff00ff' : freq < 0.66 ? '#f0e130' : '#00ffff';
          ctx.fill();
          ctx.shadowBlur = 0;

          // Peak indicator with glow
          if (peaks[i] > 0.05) {
            const peakY = h - peaks[i] * h * 0.9;
            ctx.fillStyle = '#ffffff';
            ctx.shadowBlur = 8;
            ctx.shadowColor = '#ffffff';
            ctx.fillRect(x, peakY - 3, barWidth, 3);
            ctx.shadowBlur = 0;
          }

          // Reflection effect (subtle)
          const reflectGrad = ctx.createLinearGradient(x, h, x, h + barH * 0.3);
          reflectGrad.addColorStop(0, 'rgba(255, 255, 255, 0.1)');
          reflectGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
          ctx.fillStyle = reflectGrad;
          ctx.fillRect(x, h, barWidth, barH * 0.3);
        }

        // Add subtle scanlines
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.lineWidth = 1;
        for (let y = 0; y < h; y += 4) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(w, y);
          ctx.stroke();
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

  /* ==================================================================
     RECOMMENDATION ENGINE
     ================================================================== */
  let embeddingsAvailable = false;

  async function checkEmbeddings() {
    try {
      const r = await fetch('/api/embeddings-status');
      const data = await r.json();
      embeddingsAvailable = data.available;
      if (embeddingsAvailable) {
        $('semantic-row').style.display = '';
      }
    } catch(e) {}
  }

  async function fetchSimilar(videoId) {
    showModal('Finding similar tracks...', true);
    try {
      const r = await fetch(`/api/similar/${videoId}?n=20`);
      const data = await r.json();
      if (data.error) { showModalError(data.error); return; }
      showModalResults(`Similar to: ${data.query_title}`, data.results);
    } catch(e) { showModalError('Failed to fetch: ' + e.message); }
  }

  async function doSemanticSearch(query) {
    if (!query.trim()) return;
    showModal('Searching...', true);
    try {
      const r = await fetch(`/api/search-semantic?q=${encodeURIComponent(query)}&n=20`);
      const data = await r.json();
      if (data.error) { showModalError(data.error); return; }
      showModalResults(`Search: "${data.query}"`, data.results);
    } catch(e) { showModalError('Failed to fetch: ' + e.message); }
  }

  function showModal(title, isLoading) {
    $('modal-title').textContent = title;
    $('modal-body').innerHTML = isLoading
      ? '<div class="modal-loading"><div class="spinner"></div> Loading...</div>'
      : '';
    $('modal-overlay').classList.add('active');
  }

  function showModalError(msg) {
    $('modal-body').innerHTML = `<div class="modal-error">${escHtml(msg)}</div>`;
  }

  function showModalResults(title, results) {
    $('modal-title').textContent = title;
    if (!results.length) {
      $('modal-body').innerHTML = '<div class="modal-error">No results found.</div>';
      return;
    }
    const frags = [];
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      const pct = Math.round(r.score * 100);
      const isLibrary = r.source !== 'discovery';
      const isDownloaded = allTracks.some(t => t.id === r.id && t.status === 'downloaded');
      let actionBtns = '';
      if (isDownloaded) {
        actionBtns += `<button data-mplay="${r.id}" title="Play">&#9654;</button>`;
        actionBtns += `<button data-mqueue="${r.id}" title="Queue">+</button>`;
      } else if (!isLibrary) {
        actionBtns += `<button data-myt="${r.id}" title="Listen on YouTube">&#127911;</button>`;
      }
      actionBtns += `<button data-msim="${r.id}" title="Find similar">&#8776;</button>`;
      const tag = isLibrary ? '' : '<span style="color:var(--yellow);font-size:0.7rem;margin-left:4px">NEW</span>';
      frags.push(
        `<div class="sim-result">` +
        `<span class="sim-rank">${i + 1}.</span>` +
        `<div class="sim-score-bar"><div class="sim-score-fill" style="width:${pct}%"></div></div>` +
        `<span class="sim-score">${pct}%</span>` +
        `<span class="sim-title" title="${escHtml(r.title)}">${escHtml(r.title)}${tag}</span>` +
        `<span class="sim-actions">${actionBtns}</span>` +
        `</div>`
      );
    }
    $('modal-body').innerHTML = frags.join('');
  }

  // Modal close
  $('modal-close').addEventListener('click', () => $('modal-overlay').classList.remove('active'));
  $('modal-overlay').addEventListener('click', e => {
    if (e.target === $('modal-overlay')) $('modal-overlay').classList.remove('active');
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') $('modal-overlay').classList.remove('active');
  });

  // Modal button handlers
  $('modal-body').addEventListener('click', e => {
    const playBtn = e.target.closest('[data-mplay]');
    if (playBtn) {
      const t = allTracks.find(t => t.id === playBtn.dataset.mplay);
      if (t) player.playNow(t);
      return;
    }
    const queueBtn = e.target.closest('[data-mqueue]');
    if (queueBtn) {
      const t = allTracks.find(t => t.id === queueBtn.dataset.mqueue);
      if (t) { player.addToQueue(t); queueBtn.textContent = '\u2713'; }
      return;
    }
    const ytBtn = e.target.closest('[data-myt]');
    if (ytBtn) {
      window.open(`https://www.youtube.com/watch?v=${ytBtn.dataset.myt}`, '_blank');
      return;
    }
    const simBtn = e.target.closest('[data-msim]');
    if (simBtn) { fetchSimilar(simBtn.dataset.msim); return; }
  });

  // Semantic search handlers
  $('btn-semantic').addEventListener('click', () => {
    doSemanticSearch($('semantic-input').value);
  });
  $('semantic-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSemanticSearch($('semantic-input').value);
  });

  // ---- Init ----
  player.init();
  fetchConfig();
  fetchTracks();
  connectSSE();
  checkEmbeddings();
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
    parser.add_argument("-p", "--port", type=int, default=5050,
                        help="Port to serve on (default: 5050)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    parser.add_argument("--url", type=str, default=PLAYLIST_URL,
                        help="Override playlist URL")
    parser.add_argument("--auto-fetch", action="store_true",
                        help="Automatically fetch YouTube playlist on startup (default: local only)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    app_state["output_dir"] = output_dir
    app_state["playlist_url"] = args.url

    print(f"Output directory: {output_dir}")

    # Load completed and failed progress immediately (fast, local files)
    app_state["completed"] = load_progress(output_dir)
    app_state["failed"] = load_failed(output_dir)
    print(f"Already completed: {len(app_state['completed'])}, previously failed: {len(app_state['failed'])}")

    def fetch_playlist_async():
        """Fetch playlist in background so UI can load immediately."""
        print("Fetching playlist...")
        try:
            entries = get_playlist_entries(args.url)
            app_state["entries"] = entries
            app_state["loading"] = False
            print(f"Loaded {len(entries)} tracks, {len(app_state['completed'])} already downloaded.")
            broadcast_sse("playlist_loaded", {"count": len(entries)})
        except Exception as e:
            app_state["loading"] = False
            print(f"Error fetching playlist: {e}")
            broadcast_sse("playlist_error", {"error": str(e)})

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
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
                pass
            except Exception:
                self.handle_error(request, client_address)
            finally:
                self.shutdown_request(request)

    server = ThreadedServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\nServer running at {url}")
    print("Press Ctrl+C to stop.\n")

    # Start playlist fetch in background only if --auto-fetch is set
    if args.auto_fetch:
        app_state["loading"] = True
        threading.Thread(target=fetch_playlist_async, daemon=True).start()
    else:
        print("Local mode: use 'Load Playlist' button in UI to fetch from YouTube")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
