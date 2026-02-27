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
import os
import platform
import queue
import random
import re as regex
import subprocess as _subprocess
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, parse_qs

# Load .env file for API keys
def load_dotenv(path: Path):
    """Load environment variables from .env file."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)

load_dotenv(Path(__file__).parent / ".env")

# Gemini integration for YouTube search
# Try GEMINI_API_KEY first, fall back to GOOGLE_API_KEY
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
_gemini_client = None

def get_gemini_client():
    """Lazy-load Gemini client."""
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        try:
            from google import genai
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        except ImportError:
            print("google-genai not installed. YouTube suggestions disabled.")
    return _gemini_client

def optimize_music_query(raw_input: str) -> str:
    """
    Use Gemini to transform a track title or vibe description into optimized YouTube search terms.
    """
    client = get_gemini_client()
    if not client:
        return raw_input

    try:
        from google.genai import types

        prompt = f"""You are a music expert. Given this input (either a song title or a vibe description),
generate an optimized search query to find similar music on YouTube.

Input: "{raw_input}"

Think about:
- What genre/subgenre is this? (e.g., dark ambient, lo-fi hip hop, acid techno)
- What's the mood/energy? (e.g., melancholic, uplifting, aggressive, dreamy)
- What are similar artists or tracks?
- What era or scene does this belong to?

Return ONLY a concise search query (max 10 words) that would find similar music.
No explanation, just the search terms. Examples:
- "dark ambient drone music like lustmord"
- "90s acid techno rave classics"
- "chill lo-fi beats jazz hop study music"
"""

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7)
        )
        optimized = response.text.strip().strip('"').strip("'")
        # Sanity check - if response is too long or weird, use original
        if len(optimized) > 100 or '\n' in optimized:
            return raw_input
        print(f"Query optimized: '{raw_input}' -> '{optimized}'")
        return optimized
    except Exception as e:
        print(f"Query optimization failed: {e}")
        return raw_input


def search_youtube_via_gemini(query: str, num_results: int = 8, optimize: bool = True) -> tuple[list[dict], str, str]:
    """
    Use Gemini with Google Search to find YouTube music videos.
    Returns tuple of (results, error_message, optimized_query).
    results is list of {id, title, channel, description}.
    """
    client = get_gemini_client()
    if not client:
        return [], "Gemini API not configured", query

    # Optimize the query first
    original_query = query
    if optimize:
        query = optimize_music_query(query)

    try:
        from google.genai import types

        prompt = f"""Find {num_results} YouTube music videos matching this music style/vibe: "{query}"

Search for actual YouTube videos. For each result, extract:
- The YouTube video ID (the 11-character code from the URL like "dQw4w9WgXcQ")
- The video title
- The channel name
- A brief description

Return ONLY a JSON array with this exact format, no other text:
[
  {{"id": "VIDEO_ID", "title": "Video Title", "channel": "Channel Name", "description": "Brief description"}}
]

Focus on finding high-quality music videos, official uploads, or well-known covers."""

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3,
            )
        )

        # Extract JSON from response
        text = response.text.strip()
        # Find JSON array in response
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            results = json.loads(json_str)
            # Validate and clean results
            valid_results = []
            for r in results:
                if isinstance(r, dict) and 'id' in r and 'title' in r:
                    vid_id = r['id']
                    # Validate YouTube ID format (11 chars, alphanumeric + _ -)
                    if regex.match(r'^[a-zA-Z0-9_-]{11}$', vid_id):
                        valid_results.append({
                            'id': vid_id,
                            'title': r.get('title', 'Unknown'),
                            'channel': r.get('channel', ''),
                            'description': r.get('description', ''),
                            'source': 'youtube'
                        })
            return valid_results[:num_results], "", query
        return [], "No results found", query
    except Exception as e:
        error_str = str(e)
        if "PERMISSION_DENIED" in error_str or "leaked" in error_str.lower():
            return [], "API key issue - please update .env", query
        elif "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            return [], "API quota exceeded - try again later", query
        print(f"Gemini search error: {e}")
        return [], f"Search failed: {str(e)[:50]}", query

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


FEEDBACK_FILE_NAME = "user_feedback.json"

def load_feedback(output_dir: Path) -> list[dict]:
    """Load user feedback data from disk."""
    fpath = output_dir / FEEDBACK_FILE_NAME
    if fpath.exists():
        try:
            return json.loads(fpath.read_text())
        except Exception:
            return []
    return []


def save_feedback(output_dir: Path, entry: dict):
    """Append a feedback entry to the feedback file."""
    fpath = output_dir / FEEDBACK_FILE_NAME
    feedback = load_feedback(output_dir)
    feedback.append(entry)
    # Keep last 10000 entries to prevent unbounded growth
    if len(feedback) > 10000:
        feedback = feedback[-10000:]
    fpath.write_text(json.dumps(feedback, indent=2))


try:
    import numpy as np
    from recommender import find_similar, load_discovery_embeddings, load_segment_embeddings
    _HAS_RECOMMENDER = True
except ImportError:
    _HAS_RECOMMENDER = False

try:
    from umap import UMAP
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

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
    "segment_embeddings": None,
    "segment_mtime": 0,
    "sonic_map": None,  # Cached 2D UMAP projection
    "sonic_map_mtime": 0,
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


def get_segment_embeddings():
    """Load segment embeddings from disk, cached and auto-refreshed."""
    if not _HAS_RECOMMENDER:
        return None
    emb_path = app_state["output_dir"] / "segment_embeddings.npz"
    if not emb_path.exists():
        return None
    mtime = emb_path.stat().st_mtime
    if app_state["segment_embeddings"] is None or mtime > app_state["segment_mtime"]:
        app_state["segment_embeddings"] = load_segment_embeddings(
            app_state["output_dir"])
        app_state["segment_mtime"] = mtime
    return app_state["segment_embeddings"]


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
    """Find a file by video_id (or local_ hash ID) with one of the given extensions."""
    import hashlib
    output_dir = app_state["output_dir"]

    # Standard YouTube ID - look for [video_id] in filename
    if not video_id.startswith("local_"):
        tag = f"[{video_id}]"
        for f in output_dir.iterdir():
            if tag in f.name and f.suffix.lstrip(".") in extensions:
                return f
        return None

    # Local file ID - regenerate hash to find matching file
    target_hash = video_id[6:]  # Strip "local_" prefix
    for f in output_dir.iterdir():
        if f.suffix.lstrip(".") not in extensions:
            continue
        # Check if this file's hash matches
        stem = f.stem
        h = hashlib.md5(stem.encode('utf-8')).hexdigest()[:11]
        if h == target_hash:
            return f
    return None


def generate_file_id(filepath: Path) -> str:
    """Generate a stable ID for a file without a video ID, using filename hash."""
    import hashlib
    stem = filepath.stem
    h = hashlib.md5(stem.encode('utf-8')).hexdigest()[:11]
    return f"local_{h}"


def scan_downloaded_tracks() -> list[dict]:
    """Scan output directory for downloaded audio files and return track entries.

    Handles both YouTube downloads (with [video_id] in filename) and arbitrary
    audio files dropped into the folder.
    """
    import re
    output_dir = app_state["output_dir"]
    audio_exts = (".mp3", ".opus", ".m4a", ".ogg", ".webm")
    tracks = []
    seen_ids = set()

    # Pattern to extract video ID from filename: "Title [VIDEO_ID].ext"
    id_pattern = re.compile(r'\[([a-zA-Z0-9_-]+)\]')

    for f in sorted(output_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in audio_exts:
            continue

        match = id_pattern.search(f.stem)
        if match:
            vid_id = match.group(1)
            # Extract title (everything before the [ID])
            title = f.stem[:match.start()].strip()
            if not title:
                title = vid_id
        else:
            # No video ID - generate hash-based ID and use filename as title
            vid_id = generate_file_id(f)
            title = f.stem

        if vid_id in seen_ids:
            continue
        seen_ids.add(vid_id)
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
        elif path == "/api/sonic-map":
            self._api_sonic_map()
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
        elif path == "/api/youtube-search":
            self._api_youtube_search(qs)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/smart-shuffle":
            self._api_smart_shuffle_post()
        elif self.path == "/api/download":
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
        elif self.path == "/api/feedback":
            self._api_feedback()
        elif self.path == "/api/vibe-flow":
            self._api_vibe_flow()
        elif self.path == "/api/next-similar":
            self._api_next_similar()
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
        seen_ids = set()

        # Always scan local directory first - this includes all downloaded files
        scanned = scan_downloaded_tracks()
        for i, e in enumerate(scanned):
            vid_id = e["id"]
            seen_ids.add(vid_id)
            tracks.append({
                "index": i + 1, "id": vid_id,
                "title": e["title"], "status": "downloaded",
            })

        # If playlist is loaded, add any pending/failed entries not in local files
        if state["entries"]:
            for e in state["entries"]:
                vid_id = e["id"]
                if vid_id in seen_ids:
                    continue  # Already added from local scan
                if vid_id == state.get("current_id"):
                    status = "downloading"
                elif vid_id in state["failed"]:
                    status = "failed"
                else:
                    status = "pending"
                tracks.append({
                    "index": len(tracks) + 1, "id": vid_id,
                    "title": e["title"], "status": status,
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

    def _api_feedback(self):
        """Log user feedback for YouTube suggestions (plays, thumbs up)."""
        body = self._read_json_body()
        action = body.get("action")  # "play" or "thumbup"
        yt_id = body.get("yt_id")  # YouTube video ID
        yt_title = body.get("yt_title", "")
        query_id = body.get("query_id", "")  # Original track that triggered search
        query_title = body.get("query_title", "")
        optimized_query = body.get("optimized_query", "")

        if not action or not yt_id:
            self._json_response({"error": "Missing action or yt_id"}, 400)
            return

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "yt_id": yt_id,
            "yt_title": yt_title,
            "query_id": query_id,
            "query_title": query_title,
            "optimized_query": optimized_query,
        }
        save_feedback(app_state["output_dir"], entry)
        print(f"Feedback logged: {action} on '{yt_title}' (from query: '{query_title}')")
        self._json_response({"ok": True})

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

    def _api_vibe_flow(self):
        """Generate a Flow shuffle starting from a text vibe/mood description.

        POST body: {"prompt": "dark techno heavy bass", "energy_arc": "none"}
        Returns shuffled track order starting from best semantic match.
        """
        body = self._read_json_body()
        prompt = body.get("prompt", "").strip()
        energy_arc = body.get("energy_arc", "none")

        if not prompt:
            self._json_response({"error": "No prompt provided"}, 400)
            return

        emb_data = get_embeddings()
        if emb_data is None:
            self._json_response({"error": "No embeddings found"}, 404)
            return

        try:
            from recommender import search_by_text, load_clap
            if app_state.get("clap_model") is None:
                app_state["clap_model"], app_state["clap_processor"] = load_clap(device="cuda")

            # Find best matching track for the vibe
            results = search_by_text(
                emb_data, app_state["clap_model"], app_state["clap_processor"],
                prompt, n=1, device="cuda")

            if not results:
                self._json_response({"error": "No matching tracks found"}, 404)
                return

            seed_id = results[0]["id"]

            # Get all downloaded tracks
            track_ids = list(emb_data["ids"])

            # Use the regular shuffle implementation with seed
            mode = "trajectory" if energy_arc != "none" else "flow"
            self._api_smart_shuffle_impl(mode, track_ids, seed=seed_id, energy_arc=energy_arc)

        except ImportError:
            self._json_response({"error": "CLAP model not available"}, 501)
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _api_next_similar(self):
        """Get the most similar track to the current one, excluding already played tracks.

        POST body: {"current": "track_id", "exclude": ["id1", "id2", ...]}
        Returns: {"next": {"id": "...", "title": "..."}} or {"next": null}
        """
        body = self._read_json_body()
        current_id = body.get("current")
        exclude = set(body.get("exclude", []))

        if not current_id:
            self._json_response({"error": "No current track provided"}, 400)
            return

        emb_data = get_embeddings()
        if not emb_data:
            self._json_response({"next": None, "error": "No embeddings"})
            return

        ids = emb_data["ids"]
        titles = emb_data["titles"]
        clap = emb_data["clap"]

        # Find current track index
        id_to_idx = {vid: i for i, vid in enumerate(ids)}
        if current_id not in id_to_idx:
            self._json_response({"next": None, "error": "Current track not found in embeddings"})
            return

        current_idx = id_to_idx[current_id]
        current_emb = clap[current_idx]

        # Normalize for cosine similarity
        current_norm = current_emb / (np.linalg.norm(current_emb) + 1e-8)

        # Find most similar track not in exclude list
        best_idx = None
        best_sim = -2
        for i, vid in enumerate(ids):
            if vid in exclude or vid == current_id:
                continue
            emb_norm = clap[i] / (np.linalg.norm(clap[i]) + 1e-8)
            sim = float(np.dot(current_norm, emb_norm))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx is None:
            self._json_response({"next": None, "message": "No more tracks available"})
            return

        self._json_response({
            "next": {"id": ids[best_idx], "title": titles[best_idx]},
            "similarity": best_sim
        })

    def _api_embeddings_status(self):
        """Return status of the embeddings file."""
        if not _HAS_RECOMMENDER:
            self._json_response({"available": False, "count": 0})
            return
        # Prefer segment embeddings (more complete), fall back to old embeddings
        seg_data = get_segment_embeddings()
        if seg_data:
            self._json_response({"available": True, "count": len(seg_data["ids"]), "type": "segment"})
        else:
            emb_data = get_embeddings()
            if emb_data:
                self._json_response({"available": True, "count": len(emb_data["ids"]), "type": "legacy"})
            else:
                self._json_response({"available": False, "count": 0})

    def _api_sonic_map(self):
        """Return 2D UMAP projection of track embeddings for visualization."""
        if not _HAS_RECOMMENDER or not _HAS_UMAP:
            self._json_response({
                "available": False,
                "error": "UMAP not available" if not _HAS_UMAP else "Embeddings not available"
            })
            return

        emb_data = get_embeddings()
        if not emb_data or len(emb_data["ids"]) < 10:
            self._json_response({"available": False, "error": "Need at least 10 embedded tracks"})
            return

        emb_path = app_state["output_dir"] / "embeddings.npz"
        mtime = emb_path.stat().st_mtime if emb_path.exists() else 0

        # Use cached projection if embeddings haven't changed
        if app_state["sonic_map"] is not None and mtime <= app_state["sonic_map_mtime"]:
            self._json_response(app_state["sonic_map"])
            return

        # Compute UMAP projection
        try:
            ids = emb_data["ids"]
            titles = emb_data["titles"]
            clap = emb_data["clap"]

            # UMAP to 2D
            reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
            coords_2d = reducer.fit_transform(clap)

            # Normalize to 0-1 range
            mins = coords_2d.min(axis=0)
            maxs = coords_2d.max(axis=0)
            coords_norm = (coords_2d - mins) / (maxs - mins + 1e-8)

            # Build response
            tracks = []
            for i, (vid_id, title) in enumerate(zip(ids, titles)):
                tracks.append({
                    "id": vid_id,
                    "title": title[:50],
                    "x": float(coords_norm[i, 0]),
                    "y": float(coords_norm[i, 1]),
                })

            result = {"available": True, "tracks": tracks, "count": len(tracks)}
            app_state["sonic_map"] = result
            app_state["sonic_map_mtime"] = mtime
            self._json_response(result)

        except Exception as e:
            self._json_response({"available": False, "error": str(e)})

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

    def _api_youtube_search(self, qs: str):
        """Search YouTube for similar music via Gemini."""
        params = parse_qs(qs)
        query = params.get("q", [""])[0]
        n = int(params.get("n", ["8"])[0])

        if not query:
            self._json_response({"error": "No query provided"}, 400)
            return

        if not GEMINI_API_KEY:
            self._json_response({"query": query, "results": [], "message": "Gemini API key not configured"})
            return

        results, error_msg, optimized_query = search_youtube_via_gemini(query, num_results=n)
        response = {"query": query, "results": results, "optimized_query": optimized_query}
        if error_msg:
            response["message"] = error_msg
        self._json_response(response)

    def _api_smart_shuffle_post(self):
        """POST version of smart shuffle - handles large track lists."""
        body = self._read_json_body()
        mode = body.get("mode", "random")
        track_ids = body.get("ids", [])
        seed = body.get("seed")  # Optional: starting track ID
        energy_arc = body.get("energy_arc", "none")  # none, rise, fall, peak, chill
        if isinstance(track_ids, str):
            track_ids = [tid for tid in track_ids.split(",") if tid]
        self._api_smart_shuffle_impl(mode, track_ids, seed=seed, energy_arc=energy_arc)

    def _api_smart_shuffle(self, qs: str):
        """GET version of smart shuffle (for small track lists)."""
        params = parse_qs(qs)
        mode = params.get("mode", ["random"])[0]
        track_ids = params.get("ids", [""])[0].split(",")
        track_ids = [tid for tid in track_ids if tid]
        seed = params.get("seed", [None])[0]
        self._api_smart_shuffle_impl(mode, track_ids, seed=seed)

    def _api_smart_shuffle_impl(self, mode: str, track_ids: list, seed: str = None, energy_arc: str = "none"):
        """Core smart shuffle implementation.

        Args:
            mode: shuffle mode (random, flow, anticluster, dj, trajectory)
            track_ids: list of track IDs to shuffle
            seed: optional starting track ID (for flow/dj modes)
            energy_arc: energy trajectory (none, rise, fall, peak, chill)
        """

        if not track_ids:
            self._json_response({"error": "No track IDs provided"}, 400)
            return

        if mode == "random":
            # Fisher-Yates shuffle
            import random
            shuffled = track_ids[:]
            for i in range(len(shuffled) - 1, 0, -1):
                j = random.randint(0, i)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            self._json_response({"order": shuffled, "mode": mode})
            return

        if not _HAS_RECOMMENDER:
            # Fall back to random if no recommender
            import random
            shuffled = track_ids[:]
            random.shuffle(shuffled)
            self._json_response({"order": shuffled, "mode": "random", "message": "Embeddings not available, using random"})
            return

        # Try segment embeddings first (more complete), fall back to old embeddings
        seg_data = get_segment_embeddings()
        if seg_data:
            ids = seg_data["ids"]
            clap = seg_data["mean_emb"]  # Use mean embedding as track-level representation
        else:
            emb_data = get_embeddings()
            if not emb_data:
                import random
                shuffled = track_ids[:]
                random.shuffle(shuffled)
                self._json_response({"order": shuffled, "mode": "random", "message": "Embeddings not loaded, using random"})
                return
            ids = emb_data["ids"]
            clap = emb_data["clap"]

        # Build index mapping
        id_to_idx = {vid: i for i, vid in enumerate(ids)}
        valid_ids = [tid for tid in track_ids if tid in id_to_idx]

        if len(valid_ids) < 2:
            self._json_response({"order": track_ids, "mode": mode, "message": "Not enough embedded tracks"})
            return

        valid_embs = np.stack([clap[id_to_idx[tid]] for tid in valid_ids])

        if mode == "flow":
            # Start from seed (or random), always pick most similar unplayed track
            # Optimized with numpy masking for O(n) per-step instead of O(n) python loop
            import random
            n = len(valid_ids)

            # Use seed if provided and valid, otherwise random
            if seed and seed in valid_ids:
                start = valid_ids.index(seed)
            else:
                start = random.randint(0, n - 1)

            # Precompute similarity matrix
            norms = np.linalg.norm(valid_embs, axis=1, keepdims=True)
            normed = valid_embs / (norms + 1e-8)
            sim_matrix = normed @ normed.T

            # Track availability with numpy array
            available = np.ones(n, dtype=bool)
            available[start] = False
            order = [start]

            for _ in range(n - 1):
                current = order[-1]
                sims = sim_matrix[current].copy()
                sims[~available] = -np.inf
                best_idx = int(np.argmax(sims))
                order.append(best_idx)
                available[best_idx] = False

            shuffled = [valid_ids[i] for i in order]

        elif mode == "anticluster":
            # Start from seed (or random), always pick LEAST similar to recent tracks
            # Optimized with numpy operations
            import random
            n = len(valid_ids)

            # Use seed if provided and valid, otherwise random
            if seed and seed in valid_ids:
                start = valid_ids.index(seed)
            else:
                start = random.randint(0, n - 1)

            # Precompute similarity matrix
            norms = np.linalg.norm(valid_embs, axis=1, keepdims=True)
            normed = valid_embs / (norms + 1e-8)
            sim_matrix = normed @ normed.T

            available = np.ones(n, dtype=bool)
            available[start] = False
            order = [start]

            for _ in range(n - 1):
                # Average similarity to last 3 tracks (vectorized)
                recent = order[-3:] if len(order) >= 3 else order
                avg_sims = sim_matrix[recent].mean(axis=0)
                avg_sims[~available] = np.inf  # We want minimum similarity
                best_idx = int(np.argmin(avg_sims))
                order.append(best_idx)
                available[best_idx] = False

            shuffled = [valid_ids[i] for i in order]

        elif mode == "dj":
            # DJ Mode: Match ending segments of current track to beginning of next
            # Creates smooth transitions like a professional DJ set
            import random
            seg_data = get_segment_embeddings()
            if not seg_data:
                shuffled = valid_ids[:]
                random.shuffle(shuffled)
                self._json_response({"order": shuffled, "mode": "random", "message": "Segment embeddings not available, using random"})
                return

            seg_ids = seg_data["segment_ids"]
            seg_times = seg_data["segment_times"]
            seg_emb = seg_data["segment_emb"]

            # Group segment indices by track ID
            track_segments = {}
            for i, tid in enumerate(seg_ids):
                if tid not in track_segments:
                    track_segments[tid] = []
                track_segments[tid].append(i)

            # Filter to tracks we have segments for
            dj_ids = [tid for tid in valid_ids if tid in track_segments]
            if len(dj_ids) < 2:
                shuffled = valid_ids[:]
                random.shuffle(shuffled)
                self._json_response({"order": shuffled, "mode": "random", "message": "Not enough tracks with segments"})
                return

            # For each track, compute embedding for ending (last 30s) and beginning (first 20s)
            # Using 30s for endings to skip past fade-outs and dead sound at track ends
            END_BOUNDARY_SECONDS = 30
            BEGIN_BOUNDARY_SECONDS = 20
            end_embs = {}  # tid -> embedding of ending region
            begin_embs = {}  # tid -> embedding of beginning region

            for tid in dj_ids:
                indices = track_segments[tid]
                times = seg_times[indices]
                embs = seg_emb[indices]

                # Ending: segments where end_time >= max_end - END_BOUNDARY_SECONDS
                max_end = times[:, 1].max()
                end_mask = times[:, 1] >= (max_end - END_BOUNDARY_SECONDS)
                if end_mask.any():
                    end_embs[tid] = embs[end_mask].mean(axis=0)
                else:
                    end_embs[tid] = embs[-1]

                # Beginning: segments where start_time <= BEGIN_BOUNDARY_SECONDS
                begin_mask = times[:, 0] <= BEGIN_BOUNDARY_SECONDS
                if begin_mask.any():
                    begin_embs[tid] = embs[begin_mask].mean(axis=0)
                else:
                    begin_embs[tid] = embs[0]

            # Normalize for cosine similarity
            def normalize(v):
                n = np.linalg.norm(v)
                return v / n if n > 0 else v

            for tid in dj_ids:
                end_embs[tid] = normalize(end_embs[tid])
                begin_embs[tid] = normalize(begin_embs[tid])

            # Greedy matching: start from seed (or random), pick next track whose beginning matches current ending
            # Optimized: pre-compute similarity matrix for O(n²) -> O(n) per-step lookups
            n_dj = len(dj_ids)
            id_to_dj_idx = {tid: i for i, tid in enumerate(dj_ids)}

            # Stack embeddings into matrices for fast batch computation
            end_matrix = np.stack([end_embs[tid] for tid in dj_ids])  # (n, 512)
            begin_matrix = np.stack([begin_embs[tid] for tid in dj_ids])  # (n, 512)

            # Pre-compute full similarity matrix: end[i] dot begin[j]
            # This is O(n² * d) but done once with fast BLAS
            sim_matrix = end_matrix @ begin_matrix.T  # (n, n)

            # Track which indices are still available
            available = np.ones(n_dj, dtype=bool)

            # Use seed if provided and valid, otherwise random
            if seed and seed in id_to_dj_idx:
                start_idx = id_to_dj_idx[seed]
            else:
                start_idx = random.randint(0, n_dj - 1)

            available[start_idx] = False
            order_indices = [start_idx]

            # Greedy selection using pre-computed similarities
            for _ in range(n_dj - 1):
                current_idx = order_indices[-1]
                # Get similarities from current track's ending to all beginnings
                sims = sim_matrix[current_idx].copy()
                # Mask out unavailable tracks with -inf
                sims[~available] = -np.inf
                # Pick best
                best_idx = int(np.argmax(sims))
                order_indices.append(best_idx)
                available[best_idx] = False

            order = [dj_ids[i] for i in order_indices]

            # Add any tracks that weren't in segment data
            remaining_no_seg = [tid for tid in valid_ids if tid not in track_segments]
            random.shuffle(remaining_no_seg)
            order.extend(remaining_no_seg)

            shuffled = order

        elif mode == "trajectory":
            # Trajectory mode: arrange tracks based on energy arc
            # Uses embedding norm as energy proxy, then sorts by desired pattern
            import random

            # Compute "energy" for each track using embedding norm and variance
            # Higher norm = more intensity, higher std = more complexity
            energies = {}
            for i, tid in enumerate(valid_ids):
                emb = valid_embs[i]
                # Energy score: combination of norm and standard deviation
                norm = np.linalg.norm(emb)
                std = np.std(emb)
                energies[tid] = norm * 0.5 + std * 50  # Weighted combination

            # Normalize energy scores to 0-1
            min_e = min(energies.values())
            max_e = max(energies.values())
            range_e = max_e - min_e if max_e > min_e else 1
            for tid in energies:
                energies[tid] = (energies[tid] - min_e) / range_e

            # Create target energy curve based on arc type
            n = len(valid_ids)
            target_energies = []
            if energy_arc == "rise":
                # Gradually increase energy: 0.2 -> 1.0
                target_energies = [0.2 + 0.8 * (i / (n - 1)) for i in range(n)]
            elif energy_arc == "fall":
                # Gradually decrease energy: 1.0 -> 0.2
                target_energies = [1.0 - 0.8 * (i / (n - 1)) for i in range(n)]
            elif energy_arc == "peak":
                # Build up then drop: 0.2 -> 1.0 -> 0.4
                mid = n // 2
                target_energies = []
                for i in range(n):
                    if i <= mid:
                        target_energies.append(0.2 + 0.8 * (i / mid))
                    else:
                        target_energies.append(1.0 - 0.6 * ((i - mid) / (n - mid)))
            elif energy_arc == "chill":
                # Stay low energy throughout: 0.2 -> 0.4 gentle wave
                target_energies = [0.2 + 0.2 * np.sin(np.pi * i / n) for i in range(n)]
            else:
                # Default: gentle rise
                target_energies = [0.3 + 0.4 * (i / (n - 1)) for i in range(n)]

            # Match tracks to target energies greedily
            # For each position, find the unassigned track closest to target energy
            remaining_tracks = set(valid_ids)
            shuffled = []

            # If seed provided, start with it
            if seed and seed in remaining_tracks:
                shuffled.append(seed)
                remaining_tracks.remove(seed)
                # Adjust target_energies to skip first slot
                target_energies = target_energies[1:]

            for target in target_energies:
                if not remaining_tracks:
                    break
                # Find track with energy closest to target
                best_tid = min(remaining_tracks, key=lambda t: abs(energies[t] - target))
                shuffled.append(best_tid)
                remaining_tracks.remove(best_tid)

        else:
            # Unknown mode, fall back to random
            import random
            shuffled = valid_ids[:]
            random.shuffle(shuffled)

        self._json_response({"order": shuffled, "mode": mode, "energy_arc": energy_arc})

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

.shuffle-dropdown { position: relative; display: inline-block; }
.shuffle-menu {
  display: none; position: absolute; top: 100%; left: 0; z-index: 100;
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4); min-width: 200px; margin-top: 4px;
}
.shuffle-dropdown.open .shuffle-menu { display: block; }
.shuffle-menu button {
  display: block; width: 100%; padding: 0.6rem 1rem; text-align: left;
  background: transparent; border: none; color: var(--text); cursor: pointer;
  font-size: 0.85rem; transition: background 0.15s;
}
.shuffle-menu button:hover:not(:disabled) { background: var(--bg3); }
.shuffle-menu button.active { color: var(--neon-green); font-weight: 600; }
.shuffle-menu button:disabled { color: var(--text2); cursor: not-allowed; opacity: 0.5; }
.shuffle-menu button:first-child { border-radius: 8px 8px 0 0; }
.shuffle-menu button:last-child { border-radius: 0 0 8px 8px; }

/* Track context menu */
.track-context-menu {
  position: fixed; z-index: 200;
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5); min-width: 180px;
  display: none;
}
.track-context-menu.visible { display: block; }
.track-context-menu button {
  display: block; width: 100%; padding: 0.6rem 1rem; text-align: left;
  background: transparent; border: none; color: var(--text); cursor: pointer;
  font-size: 0.85rem;
}
.track-context-menu button:hover { background: var(--bg3); }
.track-context-menu button:first-child { border-radius: 8px 8px 0 0; }
.track-context-menu button:last-child { border-radius: 0 0 8px 8px; }
.track-context-menu hr { margin: 0.2rem 0; border-color: var(--border); }

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
  cursor: pointer;
  transition: all 0.3s ease;
}
.wa-visualizer.fullscreen {
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  z-index: 10000 !important;
  border: none !important;
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
.wa-visualizer.fullscreen::before { display: none; }
.wa-visualizer canvas {
  width: 100%;
  height: 100%;
  display: block;
  position: relative;
  z-index: 0;
}
.wa-viz-fullscreen-btn {
  position: absolute;
  top: 5px;
  right: 5px;
  z-index: 10;
  background: rgba(0,0,0,0.6);
  border: 1px solid var(--accent);
  color: var(--accent);
  padding: 3px 8px;
  font-size: 11px;
  cursor: pointer;
  border-radius: 3px;
  opacity: 0;
  transition: opacity 0.2s;
}
.wa-visualizer:hover .wa-viz-fullscreen-btn { opacity: 1; }
.wa-viz-fullscreen-btn:hover { background: var(--accent); color: #000; }
.wa-viz-mode-label {
  position: absolute;
  bottom: 5px;
  left: 5px;
  z-index: 10;
  color: var(--accent);
  font-size: 10px;
  text-transform: uppercase;
  opacity: 0.6;
  pointer-events: none;
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

/* ===== PANEL TABS ===== */
.wa-panel-tabs {
  background: linear-gradient(180deg, #2a2a4a, #1a1a3a);
  padding: 3px 8px;
  display: flex; align-items: center; gap: 4px;
  border-bottom: 1px solid var(--wa-border);
}
.wa-panel-tab {
  background: transparent; border: none;
  font-size: 9px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1px; color: #666; cursor: pointer;
  padding: 3px 8px; border-radius: 3px;
  transition: all 0.15s;
}
.wa-panel-tab:hover { color: #aaa; }
.wa-panel-tab.active {
  background: rgba(0,255,100,0.15);
  color: var(--wa-green);
}
.wa-panel-tabs .wa-playlist-count {
  margin-left: auto;
  font-size: 9px; color: var(--wa-green2);
  font-family: 'Courier New', monospace;
}

/* ===== SONIC MAP ===== */
.wa-sonic-map {
  flex: 1; position: relative;
  background: #050510;
  overflow: hidden;
}
.wa-sonic-map.hidden { display: none; }
#sonic-map-canvas {
  width: 100%; height: 100%;
  cursor: crosshair;
}
.sonic-map-tooltip {
  position: absolute;
  background: rgba(0,0,0,0.9);
  border: 1px solid var(--wa-green);
  color: var(--wa-green);
  font-size: 10px; padding: 4px 8px;
  border-radius: 4px;
  pointer-events: none;
  display: none; white-space: nowrap;
  z-index: 100;
}
.sonic-map-legend {
  position: absolute; bottom: 6px; left: 6px;
  font-size: 8px; color: #666;
  display: flex; gap: 8px; align-items: center;
}
.legend-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
}
.legend-dot.current { background: #ff3366; box-shadow: 0 0 6px #ff3366; }
.legend-dot.queued { background: var(--wa-green); }
.legend-dot.other { background: #334; }

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
  border-radius: 12px; width: 900px; max-width: 95vw;
  max-height: 85vh; display: flex; flex-direction: column;
  overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.modal.dual-panel { width: 1100px; }
.dual-panel-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  height: 100%;
  overflow: hidden;
}
.dual-panel-container .panel {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid var(--border);
}
.dual-panel-container .panel:last-child { border-right: none; }
.panel-header {
  padding: 0.75rem 1rem;
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
  font-weight: 600;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
}
.panel-header .icon { font-size: 1.1rem; }
.panel-header.local { color: var(--neon-green); }
.panel-header.youtube { color: var(--red); }
.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 0;
}
@media (max-width: 800px) {
  .dual-panel-container { grid-template-columns: 1fr; }
  .modal.dual-panel { width: 95vw; }
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
<!-- Track context menu (right-click) -->
<div class="track-context-menu" id="track-context-menu">
  <button data-action="play">&#9654; Play Now</button>
  <button data-action="queue-next">Add to Queue (Next)</button>
  <hr>
  <button data-action="flow-from">&#127919; Start Flow from here</button>
  <button data-action="dj-from">&#127911; Start DJ from here</button>
  <hr>
  <button data-action="similar">&#8776; Find Similar</button>
</div>

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
    <div class="shuffle-dropdown" id="shuffle-dropdown">
      <button class="btn sm" id="btn-queue-all" title="Add entire library to playlist" style="background:var(--neon-green);color:#000">&#9654; Queue All &#9662;</button>
      <div class="shuffle-menu" id="shuffle-menu">
        <button data-shuffle="random" class="active">Random</button>
        <button data-shuffle="flow">Flow (smooth transitions)</button>
        <button data-shuffle="anticluster">Anti-cluster (variety)</button>
        <button data-shuffle="dj">DJ Mode (end-to-start matching)</button>
        <hr style="margin:0.3rem 0;border-color:var(--border)">
        <button data-shuffle="trajectory" data-arc="rise">&#128200; Energy Rise</button>
        <button data-shuffle="trajectory" data-arc="fall">&#128201; Energy Fall</button>
        <button data-shuffle="trajectory" data-arc="peak">&#127881; Peak (build &amp; drop)</button>
        <button data-shuffle="trajectory" data-arc="chill">&#127769; Chill (low energy)</button>
      </div>
    </div>
    <button class="btn primary" id="btn-download">Download Selected</button>
    <button class="btn danger" id="btn-cancel" style="display:none">Cancel</button>
    <button class="btn sm" id="btn-clear-failed" title="Reset failed tracks to pending so they can be retried">Clear Failed</button>
  </div>

  <div class="semantic-row" id="semantic-row" style="display:none">
    <input type="text" id="semantic-input" placeholder="Describe a vibe... (e.g. &quot;dark techno heavy bass&quot;, &quot;melodic ambient&quot;)">
    <button class="btn sm" id="btn-semantic">Search</button>
    <button class="btn sm" id="btn-vibe-flow" style="background:var(--accent);color:#000" title="Find matching track and start Flow from there">&#127919; Vibe Flow</button>
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
  <div class="wa-visualizer" id="wa-visualizer">
    <canvas id="wa-viz-canvas"></canvas>
    <button class="wa-viz-fullscreen-btn" id="wa-viz-fullscreen">&#x26F6; FULLSCREEN</button>
    <div class="wa-viz-mode-label" id="wa-viz-mode-label">BARS</div>
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
    <div class="wa-mode-btn" id="wa-adaptive" title="Adaptive Flow: next track is always most similar to current">FLOW</div>
  </div>

  <!-- Playlist/Map tabs -->
  <div class="wa-panel-tabs">
    <button class="wa-panel-tab active" data-panel="playlist">Playlist</button>
    <button class="wa-panel-tab" data-panel="map">Sonic Map</button>
    <span class="wa-playlist-count" id="wa-pl-count">0 tracks</span>
  </div>

  <!-- Playlist panel -->
  <div class="wa-playlist" id="wa-playlist"></div>

  <!-- Sonic Map panel -->
  <div class="wa-sonic-map hidden" id="wa-sonic-map">
    <canvas id="sonic-map-canvas"></canvas>
    <div class="sonic-map-tooltip" id="sonic-map-tooltip"></div>
    <div class="sonic-map-legend">
      <span class="legend-dot current"></span> Current
      <span class="legend-dot queued"></span> Queued
      <span class="legend-dot other"></span> Library
    </div>
  </div>
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

  // Right-click context menu
  const ctxMenu = $('track-context-menu');
  let ctxTrack = null;

  tbody.addEventListener('contextmenu', e => {
    const tr = e.target.closest('tr');
    if (!tr) return;
    e.preventDefault();
    const id = tr.dataset.id;
    ctxTrack = allTracks.find(t => t.id === id);
    if (!ctxTrack) return;

    // Position menu at cursor
    ctxMenu.style.left = e.clientX + 'px';
    ctxMenu.style.top = e.clientY + 'px';
    ctxMenu.classList.add('visible');

    // Enable/disable play button based on status
    const playBtn = ctxMenu.querySelector('[data-action="play"]');
    const queueBtn = ctxMenu.querySelector('[data-action="queue-next"]');
    const flowBtn = ctxMenu.querySelector('[data-action="flow-from"]');
    const djBtn = ctxMenu.querySelector('[data-action="dj-from"]');
    const isDownloaded = ctxTrack.status === 'downloaded';
    if (playBtn) playBtn.disabled = !isDownloaded;
    if (queueBtn) queueBtn.disabled = !isDownloaded;
    if (flowBtn) flowBtn.disabled = !isDownloaded;
    if (djBtn) djBtn.disabled = !isDownloaded;
  });

  // Hide context menu on click elsewhere
  document.addEventListener('click', () => {
    ctxMenu.classList.remove('visible');
  });

  // Context menu actions
  ctxMenu.addEventListener('click', async e => {
    const btn = e.target.closest('button');
    if (!btn || btn.disabled || !ctxTrack) return;
    const action = btn.dataset.action;
    ctxMenu.classList.remove('visible');

    if (action === 'play') {
      player.playNow(ctxTrack);
    } else if (action === 'queue-next') {
      // Insert after current index
      const idx = player.currentIndex >= 0 ? player.currentIndex + 1 : 0;
      player.queue.splice(idx, 0, ctxTrack);
      player.renderQueue();
    } else if (action === 'flow-from') {
      // Start Flow shuffle with this track as seed
      await queueAllWithShuffle('flow', ctxTrack.id, 'none');
    } else if (action === 'dj-from') {
      // Start DJ mode with this track as seed
      await queueAllWithShuffle('dj', ctxTrack.id, 'none');
    } else if (action === 'similar') {
      fetchSimilar(ctxTrack.id);
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

  // Smart shuffle dropdown
  let shuffleMode = 'random';
  const shuffleDropdown = $('shuffle-dropdown');
  const shuffleMenu = $('shuffle-menu');
  const btnQueueAll = $('btn-queue-all');

  // Toggle dropdown on button click
  btnQueueAll.addEventListener('click', (e) => {
    e.stopPropagation();
    shuffleDropdown.classList.toggle('open');
  });

  // Close dropdown when clicking outside
  document.addEventListener('click', (e) => {
    if (!shuffleDropdown.contains(e.target)) {
      shuffleDropdown.classList.remove('open');
    }
  });

  // Handle shuffle mode selection
  let energyArc = 'none';
  shuffleMenu.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (btn.disabled) return;

      const mode = btn.dataset.shuffle;
      const arc = btn.dataset.arc || 'none';
      shuffleMode = mode;
      energyArc = arc;

      // Update active state
      shuffleMenu.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Close menu and queue tracks
      shuffleDropdown.classList.remove('open');
      await queueAllWithShuffle(mode, null, arc);
    });
  });

  async function queueAllWithShuffle(mode, seed = null, arc = 'none') {
    const downloaded = allTracks.filter(t => t.status === 'downloaded');
    if (downloaded.length === 0) {
      alert('No downloaded tracks to queue!');
      return;
    }

    const ids = downloaded.map(t => t.id);
    btnQueueAll.disabled = true;
    btnQueueAll.innerHTML = '&#8987; Shuffling...';

    try {
      const body = {mode: mode, ids: ids};
      if (seed) body.seed = seed;
      if (arc && arc !== 'none') body.energy_arc = arc;

      const resp = await fetch('/api/smart-shuffle', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      const data = await resp.json();

      if (data.error) {
        alert(data.error);
        return;
      }

      // Create track map for quick lookup
      const trackMap = Object.fromEntries(downloaded.map(t => [t.id, t]));

      // Queue tracks in returned order (batch for performance)
      player.playlist = [];
      const tracksToAdd = data.order
        .filter(id => trackMap[id])
        .map(id => trackMap[id]);
      player.addBatchToQueue(tracksToAdd);

      // Start playing from the beginning
      if (!player.currentTrack) {
        player.currentIndex = 0;
        player.loadAndPlay();
      }

      const modeNames = {random: 'Random', flow: 'Flow', anticluster: 'Variety', dj: 'DJ', trajectory: 'Energy'};
      let label = modeNames[data.mode] || data.mode;
      if (data.energy_arc && data.energy_arc !== 'none') {
        const arcNames = {rise: '↗', fall: '↘', peak: '⛰', chill: '🌙'};
        label += ' ' + (arcNames[data.energy_arc] || data.energy_arc);
      }
      btnQueueAll.innerHTML = `&#10003; ${data.order.length} (${label})`;
      if (data.message) console.log('Shuffle:', data.message);
    } catch (err) {
      console.error('Shuffle error:', err);
      alert('Shuffle failed: ' + err.message);
    } finally {
      setTimeout(() => {
        btnQueueAll.disabled = false;
        btnQueueAll.innerHTML = '&#9654; Queue All &#9662;';
      }, 2000);
    }
  }

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
    adaptive: false,    // adaptive flow mode
    playedInSession: new Set(),  // tracks already played (for adaptive mode)
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
      $('wa-adaptive').addEventListener('click', () => {
        this.adaptive = !this.adaptive;
        $('wa-adaptive').classList.toggle('on', this.adaptive);
        if (this.adaptive) {
          // In adaptive mode, disable regular shuffle
          this.shuffle = false;
          $('wa-shuffle').classList.remove('on');
        }
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
      this.vizMode = 0;
      this.vizModes = ['bars', 'wave', 'circular', 'particles', 'kaleidoscope', 'nebula', 'matrix'];
      this.particles = [];
      this.vizContainer = $('wa-visualizer');
      this.vizModeLabel = $('wa-viz-mode-label');
      this.isFullscreen = false;

      // Click canvas to change viz mode
      this.vizCanvas.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') return;
        this.vizMode = (this.vizMode + 1) % this.vizModes.length;
        this.vizModeLabel.textContent = this.vizModes[this.vizMode].toUpperCase();
      });

      // Fullscreen button
      $('wa-viz-fullscreen').addEventListener('click', (e) => {
        e.stopPropagation();
        this.toggleVizFullscreen();
      });

      // ESC to exit fullscreen
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.isFullscreen) {
          this.toggleVizFullscreen();
        }
      });
    },

    toggleVizFullscreen() {
      this.isFullscreen = !this.isFullscreen;
      this.vizContainer.classList.toggle('fullscreen', this.isFullscreen);
      $('wa-viz-fullscreen').innerHTML = this.isFullscreen ? '&#10005; EXIT' : '&#x26F6; FULLSCREEN';
    },

    ensureAudioContext() {
      if (this.audioCtx) return;
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 512;
      this.analyser.smoothingTimeConstant = 0.75;
      this.source = this.audioCtx.createMediaElementSource(this.audio);
      this.source.connect(this.analyser);
      this.analyser.connect(this.audioCtx.destination);
      this.startVisualizer();
    },

    startVisualizer() {
      const canvas = this.vizCanvas;
      const ctx = this.vizCtx;
      const analyser = this.analyser;
      const freqBufLen = analyser.frequencyBinCount;
      const freqData = new Uint8Array(freqBufLen);
      const timeData = new Uint8Array(freqBufLen);
      const self = this;

      // State for visualizations
      const peaks = new Float32Array(64).fill(0);
      const peakDecay = 0.97;
      let rotation = 0;
      let hueShift = 0;

      // Particle system
      class Particle {
        constructor(x, y, energy) {
          this.x = x;
          this.y = y;
          this.vx = (Math.random() - 0.5) * energy * 8;
          this.vy = -Math.random() * energy * 12 - 2;
          this.life = 1;
          this.decay = 0.015 + Math.random() * 0.02;
          this.size = 2 + Math.random() * 4 * energy;
          this.hue = Math.random() * 60 + 40; // yellow-ish
        }
        update() {
          this.x += this.vx;
          this.vy += 0.15; // gravity
          this.y += this.vy;
          this.life -= this.decay;
        }
        draw(ctx) {
          if (this.life <= 0) return;
          ctx.beginPath();
          ctx.arc(this.x, this.y, this.size * this.life, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${this.hue}, 100%, 60%, ${this.life * 0.8})`;
          ctx.shadowBlur = 15;
          ctx.shadowColor = `hsla(${this.hue}, 100%, 60%, 0.5)`;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      }

      const drawBars = (w, h) => {
        const barCount = 48;
        const gap = 2;
        const barWidth = Math.floor((w - gap * barCount) / barCount);
        const step = Math.floor(freqBufLen / barCount);

        for (let i = 0; i < barCount; i++) {
          let sum = 0;
          for (let j = 0; j < step; j++) sum += freqData[i * step + j] || 0;
          const val = (sum / step) / 255;
          const barH = val * h * 0.85;

          if (val > peaks[i]) peaks[i] = val;
          else peaks[i] = Math.max(peaks[i] * peakDecay - 0.015, 0);

          const x = i * (barWidth + gap);
          const freq = i / barCount;

          const grad = ctx.createLinearGradient(x, h, x, h - barH);
          if (freq < 0.33) {
            grad.addColorStop(0, 'rgba(255, 0, 128, 0.9)');
            grad.addColorStop(1, 'rgba(255, 100, 255, 0.6)');
          } else if (freq < 0.66) {
            grad.addColorStop(0, 'rgba(255, 180, 0, 0.9)');
            grad.addColorStop(1, 'rgba(255, 255, 100, 0.6)');
          } else {
            grad.addColorStop(0, 'rgba(0, 200, 255, 0.9)');
            grad.addColorStop(1, 'rgba(150, 255, 255, 0.6)');
          }

          ctx.fillStyle = grad;
          ctx.shadowBlur = 12;
          ctx.shadowColor = freq < 0.33 ? '#ff00ff' : freq < 0.66 ? '#f0e130' : '#00ffff';
          ctx.fillRect(x, h - barH, barWidth, barH);
          ctx.shadowBlur = 0;

          if (peaks[i] > 0.05) {
            ctx.fillStyle = '#fff';
            ctx.shadowBlur = 8;
            ctx.shadowColor = '#fff';
            ctx.fillRect(x, h - peaks[i] * h * 0.85 - 3, barWidth, 3);
            ctx.shadowBlur = 0;
          }
        }
      };

      const drawWave = (w, h) => {
        analyser.getByteTimeDomainData(timeData);
        const centerY = h / 2;

        // Draw multiple waves with different colors
        const colors = ['#ff00ff', '#f0e130', '#00ffff'];
        colors.forEach((color, layer) => {
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = 3 - layer;
          ctx.shadowBlur = 20;
          ctx.shadowColor = color;

          const offset = layer * 2;
          for (let i = 0; i < freqBufLen; i++) {
            const x = (i / freqBufLen) * w;
            const v = (timeData[i] / 128.0) - 1;
            const y = centerY + v * (h / 2.5) * (1 + layer * 0.1);

            if (i === 0) ctx.moveTo(x, y + offset);
            else ctx.lineTo(x, y + offset);
          }
          ctx.stroke();
          ctx.shadowBlur = 0;
        });

        // Draw center line
        ctx.strokeStyle = 'rgba(240, 225, 48, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(w, centerY);
        ctx.stroke();
      };

      const drawCircular = (w, h) => {
        const cx = w / 2;
        const cy = h / 2;
        const radius = Math.min(w, h) * 0.35;
        const bars = 64;
        rotation += 0.005;
        hueShift = (hueShift + 0.5) % 360;

        // Get average energy for pulsing
        let avgEnergy = 0;
        for (let i = 0; i < freqBufLen; i++) avgEnergy += freqData[i];
        avgEnergy = avgEnergy / freqBufLen / 255;

        const pulseRadius = radius * (1 + avgEnergy * 0.3);

        for (let i = 0; i < bars; i++) {
          const angle = (i / bars) * Math.PI * 2 + rotation;
          const freqIdx = Math.floor((i / bars) * freqBufLen);
          const val = freqData[freqIdx] / 255;
          const barLen = val * radius * 0.8 + 5;

          const x1 = cx + Math.cos(angle) * pulseRadius * 0.4;
          const y1 = cy + Math.sin(angle) * pulseRadius * 0.4;
          const x2 = cx + Math.cos(angle) * (pulseRadius * 0.4 + barLen);
          const y2 = cy + Math.sin(angle) * (pulseRadius * 0.4 + barLen);

          const hue = (i / bars) * 60 + hueShift; // yellow-cyan range
          ctx.strokeStyle = `hsla(${hue}, 100%, 60%, 0.9)`;
          ctx.lineWidth = 4;
          ctx.shadowBlur = 15;
          ctx.shadowColor = `hsla(${hue}, 100%, 50%, 0.6)`;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }

        // Inner glow circle
        const innerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, pulseRadius * 0.4);
        innerGrad.addColorStop(0, `rgba(240, 225, 48, ${0.1 + avgEnergy * 0.2})`);
        innerGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = innerGrad;
        ctx.beginPath();
        ctx.arc(cx, cy, pulseRadius * 0.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      };

      const drawParticles = (w, h) => {
        // Get bass energy for particle spawning
        let bassEnergy = 0;
        for (let i = 0; i < 8; i++) bassEnergy += freqData[i];
        bassEnergy = bassEnergy / 8 / 255;

        // Spawn particles based on bass
        if (bassEnergy > 0.5 && Math.random() < bassEnergy) {
          const x = Math.random() * w;
          self.particles.push(new Particle(x, h, bassEnergy));
        }

        // Also spawn from center on strong beats
        if (bassEnergy > 0.7 && Math.random() < 0.3) {
          self.particles.push(new Particle(w/2 + (Math.random()-0.5)*100, h*0.6, bassEnergy));
        }

        // Update and draw particles
        self.particles = self.particles.filter(p => p.life > 0 && p.y < h + 50);
        self.particles.forEach(p => {
          p.update();
          p.draw(ctx);
        });

        // Draw subtle frequency bars at bottom
        const barCount = 32;
        const barWidth = w / barCount;
        for (let i = 0; i < barCount; i++) {
          const val = freqData[Math.floor(i * freqBufLen / barCount)] / 255;
          const barH = val * 30;
          const hue = 50 + i * 2;
          ctx.fillStyle = `hsla(${hue}, 100%, 60%, 0.4)`;
          ctx.fillRect(i * barWidth, h - barH, barWidth - 1, barH);
        }
      };

      // Kaleidoscope - mirrored psychedelic patterns
      let kAngle = 0;
      const drawKaleidoscope = (w, h) => {
        const cx = w / 2, cy = h / 2;
        const segments = 8;
        kAngle += 0.01;

        // Get overall energy
        let energy = 0;
        for (let i = 0; i < freqBufLen; i++) energy += freqData[i];
        energy = energy / freqBufLen / 255;

        ctx.save();
        ctx.translate(cx, cy);

        for (let seg = 0; seg < segments; seg++) {
          ctx.save();
          ctx.rotate((seg / segments) * Math.PI * 2);
          if (seg % 2) ctx.scale(1, -1); // Mirror alternate segments

          // Draw frequency-driven shapes
          for (let i = 0; i < 32; i++) {
            const val = freqData[i * 4] / 255;
            const dist = 20 + i * 8 + val * 60;
            const size = 3 + val * 15;
            const angle = kAngle + i * 0.2 + val * 0.5;

            const x = Math.cos(angle) * dist;
            const y = Math.sin(angle) * dist * 0.3;
            const hue = (i * 11 + hueShift) % 360;

            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${hue}, 100%, 60%, ${0.3 + val * 0.5})`;
            ctx.shadowBlur = 20;
            ctx.shadowColor = `hsla(${hue}, 100%, 50%, 0.8)`;
            ctx.fill();
          }
          ctx.restore();
        }
        ctx.restore();
        ctx.shadowBlur = 0;
      };

      // Nebula - cosmic clouds
      const stars = [];
      for (let i = 0; i < 100; i++) {
        stars.push({x: Math.random(), y: Math.random(), s: Math.random() * 2 + 0.5, b: Math.random()});
      }
      let nebulaTime = 0;
      const drawNebula = (w, h) => {
        nebulaTime += 0.02;

        // Get bass and treble
        let bass = 0, treble = 0;
        for (let i = 0; i < 8; i++) bass += freqData[i];
        for (let i = freqBufLen - 16; i < freqBufLen; i++) treble += freqData[i];
        bass = bass / 8 / 255;
        treble = treble / 16 / 255;

        // Draw twinkling stars
        stars.forEach(star => {
          const twinkle = 0.3 + Math.sin(nebulaTime * 3 + star.b * 10) * 0.3 + treble * 0.4;
          ctx.beginPath();
          ctx.arc(star.x * w, star.y * h, star.s * (1 + treble), 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${twinkle})`;
          ctx.fill();
        });

        // Draw nebula clouds
        const cx = w / 2, cy = h / 2;
        for (let layer = 0; layer < 3; layer++) {
          const grad = ctx.createRadialGradient(
            cx + Math.sin(nebulaTime + layer) * 50 * bass,
            cy + Math.cos(nebulaTime * 0.7 + layer) * 30 * bass,
            0,
            cx, cy, Math.min(w, h) * 0.6
          );
          const hue = (layer * 120 + nebulaTime * 20) % 360;
          grad.addColorStop(0, `hsla(${hue}, 100%, 50%, ${0.1 + bass * 0.3})`);
          grad.addColorStop(0.5, `hsla(${hue + 30}, 80%, 40%, ${0.05 + bass * 0.1})`);
          grad.addColorStop(1, 'transparent');
          ctx.fillStyle = grad;
          ctx.fillRect(0, 0, w, h);
        }

        // Draw pulsing central orb
        const orbSize = 30 + bass * 80;
        const orbGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, orbSize);
        orbGrad.addColorStop(0, `rgba(255, 220, 100, ${0.8 + bass * 0.2})`);
        orbGrad.addColorStop(0.3, `rgba(255, 100, 50, ${0.4 + bass * 0.3})`);
        orbGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = orbGrad;
        ctx.beginPath();
        ctx.arc(cx, cy, orbSize, 0, Math.PI * 2);
        ctx.fill();
      };

      // Matrix - falling code rain
      const columns = [];
      const matrixChars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*';
      const drawMatrix = (w, h) => {
        const colWidth = 14;
        const numCols = Math.ceil(w / colWidth);

        // Initialize columns
        while (columns.length < numCols) {
          columns.push({y: Math.random() * -100, speed: 2 + Math.random() * 5, chars: []});
        }

        // Get energy for effects
        let energy = 0;
        for (let i = 0; i < 32; i++) energy += freqData[i];
        energy = energy / 32 / 255;

        ctx.font = '12px monospace';

        columns.forEach((col, i) => {
          // Update position
          col.y += col.speed * (0.5 + energy * 1.5);
          if (col.y > h + 100) {
            col.y = Math.random() * -50;
            col.speed = 2 + Math.random() * 5;
          }

          // Draw characters
          const x = i * colWidth;
          const charCount = 15;
          for (let j = 0; j < charCount; j++) {
            const y = col.y - j * 14;
            if (y < 0 || y > h) continue;

            const char = matrixChars[Math.floor(Math.random() * matrixChars.length)];
            const brightness = 1 - (j / charCount);
            const hue = 80 + energy * 40; // Green to yellow with energy

            if (j === 0) {
              ctx.fillStyle = `rgba(255, 255, 255, ${0.9 + energy * 0.1})`;
              ctx.shadowBlur = 15;
              ctx.shadowColor = `hsl(${hue}, 100%, 70%)`;
            } else {
              ctx.fillStyle = `hsla(${hue}, 100%, ${40 + brightness * 30}%, ${brightness * 0.8})`;
              ctx.shadowBlur = 0;
            }
            ctx.fillText(char, x, y);
          }
        });
        ctx.shadowBlur = 0;
      };

      const draw = () => {
        self.vizAnimId = requestAnimationFrame(draw);
        const w = canvas.width = canvas.clientWidth * 2;
        const h = canvas.height = canvas.clientHeight * 2;
        canvas.style.width = canvas.clientWidth + 'px';
        canvas.style.height = canvas.clientHeight + 'px';

        analyser.getByteFrequencyData(freqData);

        // Clear with fade for trails (different fade rates per mode)
        const fadeRates = [0.4, 0.4, 0.4, 0.15, 0.08, 0.12, 0.7];
        ctx.fillStyle = `rgba(0,0,0,${fadeRates[self.vizMode] || 0.4})`;
        ctx.fillRect(0, 0, w, h);

        // Draw based on mode
        switch (self.vizMode) {
          case 0: drawBars(w, h); break;
          case 1: drawWave(w, h); break;
          case 2: drawCircular(w, h); break;
          case 3: drawParticles(w, h); break;
          case 4: drawKaleidoscope(w, h); break;
          case 5: drawNebula(w, h); break;
          case 6: drawMatrix(w, h); break;
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

    addToQueue(track, skipRender = false) {
      if (!this.playlist.find(t => t.id === track.id)) {
        this.playlist.push({id: track.id, title: track.title});
        if (!skipRender) this.renderPlaylist();
      }
    },

    // Batch add for performance - renders only once at the end
    addBatchToQueue(tracks) {
      const existingIds = new Set(this.playlist.map(t => t.id));
      for (const track of tracks) {
        if (!existingIds.has(track.id)) {
          this.playlist.push({id: track.id, title: track.title});
          existingIds.add(track.id);
        }
      }
      this.renderPlaylist();
    },

    loadAndPlay() {
      const track = this.playlist[this.currentIndex];
      if (!track) return;
      this.currentTrack = track;

      // Track for adaptive mode
      this.playedInSession.add(track.id);

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

    async next() {
      if (!this.playlist.length) return;

      // Adaptive Flow mode: find most similar track via API
      if (this.adaptive && this.currentTrack) {
        try {
          const exclude = Array.from(this.playedInSession);
          const resp = await fetch('/api/next-similar', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({current: this.currentTrack.id, exclude: exclude})
          });
          const data = await resp.json();

          if (data.next) {
            // Find or add track to playlist
            let idx = this.playlist.findIndex(t => t.id === data.next.id);
            if (idx === -1) {
              // Add to playlist if not there
              this.playlist.push(data.next);
              idx = this.playlist.length - 1;
              this.renderPlaylist();
            }
            this.currentIndex = idx;
            this.loadAndPlay();
            return;
          }
          // Fall through to normal behavior if no similar found
        } catch (err) {
          console.error('Adaptive mode error:', err);
          // Fall through to normal behavior
        }
      }

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

      // Update sonic map
      if (typeof sonicMap !== 'undefined') sonicMap.onPlaylistChange();
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

  // Track current search context for feedback
  let currentSearchContext = { queryId: '', queryTitle: '', optimizedQuery: '' };

  async function fetchSimilar(videoId) {
    showModal('Finding similar tracks...', true);
    const modal = document.querySelector('.modal');
    modal.classList.add('dual-panel');

    try {
      // Fetch local similar tracks
      const localPromise = fetch(`/api/similar/${videoId}?n=15`).then(r => r.json());

      // Get the track title for YouTube search
      const track = allTracks.find(t => t.id === videoId);
      const queryTitle = track ? track.title : videoId;

      // Store context for feedback
      currentSearchContext = { queryId: videoId, queryTitle: queryTitle, optimizedQuery: '' };

      // Start YouTube search in parallel (limited to 5 results)
      const ytPromise = fetch(`/api/youtube-search?q=${encodeURIComponent(queryTitle)}&n=5`)
        .then(r => r.json())
        .catch(() => ({ results: [] }));

      const [localData, ytData] = await Promise.all([localPromise, ytPromise]);

      if (localData.error) {
        showModalError(localData.error);
        return;
      }

      // Store the optimized query if available
      if (ytData.optimized_query) {
        currentSearchContext.optimizedQuery = ytData.optimized_query;
      }

      // Calculate if we need YouTube suggestions (avg score < 50%)
      const avgScore = localData.results.length > 0
        ? localData.results.reduce((sum, r) => sum + r.score, 0) / localData.results.length
        : 0;
      const needsYouTube = avgScore < 0.5 || localData.results.length < 5;

      showDualPanelResults(
        `Similar to: ${localData.query_title}`,
        localData.results,
        ytData.results || [],
        needsYouTube,
        ytData.message || ''
      );
    } catch(e) {
      showModalError('Failed to fetch: ' + e.message);
    }
  }

  async function logFeedback(action, ytId, ytTitle) {
    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          action: action,
          yt_id: ytId,
          yt_title: ytTitle,
          query_id: currentSearchContext.queryId,
          query_title: currentSearchContext.queryTitle,
          optimized_query: currentSearchContext.optimizedQuery,
        })
      });
    } catch(e) { console.error('Feedback log failed:', e); }
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
    document.querySelector('.modal').classList.remove('dual-panel');
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

  function showDualPanelResults(title, localResults, ytResults, highlightYouTube, ytErrorMsg) {
    $('modal-title').textContent = title;
    document.querySelector('.modal').classList.add('dual-panel');

    // Build local results panel
    let localHtml = '';
    if (localResults.length === 0) {
      localHtml = '<div class="modal-error" style="padding:2rem">No local matches found.</div>';
    } else {
      for (let i = 0; i < localResults.length; i++) {
        const r = localResults[i];
        const pct = Math.round(r.score * 100);
        const isDownloaded = allTracks.some(t => t.id === r.id && t.status === 'downloaded');
        let actionBtns = '';
        if (isDownloaded) {
          actionBtns += `<button data-mplay="${r.id}" title="Play">&#9654;</button>`;
          actionBtns += `<button data-mqueue="${r.id}" title="Queue">+</button>`;
        }
        actionBtns += `<button data-msim="${r.id}" title="Find similar">&#8776;</button>`;
        localHtml += `<div class="sim-result">` +
          `<span class="sim-rank">${i + 1}.</span>` +
          `<div class="sim-score-bar"><div class="sim-score-fill" style="width:${pct}%"></div></div>` +
          `<span class="sim-score">${pct}%</span>` +
          `<span class="sim-title" title="${escHtml(r.title)}">${escHtml(r.title)}</span>` +
          `<span class="sim-actions">${actionBtns}</span>` +
          `</div>`;
      }
    }

    // Build YouTube results panel
    let ytHtml = '';
    if (ytResults.length === 0) {
      const msg = ytErrorMsg || 'No YouTube suggestions available';
      ytHtml = `<div class="modal-loading" style="padding:2rem;color:var(--text2)">${escHtml(msg)}</div>`;
    } else {
      for (let i = 0; i < ytResults.length; i++) {
        const r = ytResults[i];
        const channel = r.channel ? ` <span style="color:var(--text2);font-size:0.75rem">- ${escHtml(r.channel)}</span>` : '';
        const isAlreadyDownloaded = allTracks.some(t => t.id === r.id && t.status === 'downloaded');
        const titleAttr = escHtml(r.title).replace(/"/g, '&quot;');
        let actionBtns = '';
        if (isAlreadyDownloaded) {
          actionBtns += `<button data-mplay="${r.id}" title="Play (already downloaded)">&#9654;</button>`;
        } else {
          actionBtns += `<button data-ytplay="${r.id}" data-yttitle="${titleAttr}" title="Listen on YouTube">&#127911;</button>`;
        }
        actionBtns += `<button data-ytlike="${r.id}" data-yttitle="${titleAttr}" title="Thumbs up - this is a good match!" style="color:var(--neon-green)">&#128077;</button>`;
        ytHtml += `<div class="sim-result">` +
          `<span class="sim-rank" style="color:var(--red)">${i + 1}.</span>` +
          `<span class="sim-title" title="${escHtml(r.title)}" style="flex:1">${escHtml(r.title)}${channel}</span>` +
          `<span class="sim-actions">${actionBtns}</span>` +
          `</div>`;
      }
    }

    // Indicator for when YouTube is highlighted
    const ytIndicator = highlightYouTube && ytResults.length > 0
      ? '<span style="color:var(--yellow);font-size:0.7rem;margin-left:8px">(low local matches - check these!)</span>'
      : '';

    $('modal-body').innerHTML = `
      <div class="dual-panel-container">
        <div class="panel">
          <div class="panel-header local"><span class="icon">&#128190;</span> Your Library</div>
          <div class="panel-body">${localHtml}</div>
        </div>
        <div class="panel">
          <div class="panel-header youtube"><span class="icon">&#9654;</span> YouTube Suggestions${ytIndicator}</div>
          <div class="panel-body">${ytHtml}</div>
        </div>
      </div>
    `;
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
    // YouTube play button with feedback logging
    const ytPlayBtn = e.target.closest('[data-ytplay]');
    if (ytPlayBtn) {
      const ytId = ytPlayBtn.dataset.ytplay;
      const ytTitle = ytPlayBtn.dataset.yttitle || '';
      logFeedback('play', ytId, ytTitle);
      window.open(`https://www.youtube.com/watch?v=${ytId}`, '_blank');
      return;
    }
    // YouTube thumbs up button
    const ytLikeBtn = e.target.closest('[data-ytlike]');
    if (ytLikeBtn) {
      const ytId = ytLikeBtn.dataset.ytlike;
      const ytTitle = ytLikeBtn.dataset.yttitle || '';
      logFeedback('thumbup', ytId, ytTitle);
      ytLikeBtn.textContent = '\u2705'; // checkmark
      ytLikeBtn.style.color = 'var(--neon-green)';
      ytLikeBtn.disabled = true;
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

  // Vibe Flow: semantic search + Flow shuffle
  async function doVibeFlow(prompt) {
    if (!prompt.trim()) {
      alert('Enter a vibe description first!');
      return;
    }
    const btn = $('btn-vibe-flow');
    btn.disabled = true;
    btn.innerHTML = '&#8987; Finding vibe...';

    try {
      const resp = await fetch('/api/vibe-flow', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt: prompt})
      });
      const data = await resp.json();

      if (data.error) {
        alert('Vibe Flow failed: ' + data.error);
        return;
      }

      // Build track map from all tracks
      const downloaded = allTracks.filter(t => t.status === 'downloaded');
      const trackMap = Object.fromEntries(downloaded.map(t => [t.id, t]));

      // Queue tracks in returned order
      player.queue = [];
      data.order.forEach(id => {
        if (trackMap[id]) player.addToQueue(trackMap[id]);
      });

      // Start playing
      if (!player.currentTrack) {
        player.currentIndex = 0;
        player.loadAndPlay();
      }

      btn.innerHTML = `&#10003; ${data.order.length} from "${prompt.slice(0, 20)}..."`;
    } catch (err) {
      console.error('Vibe Flow error:', err);
      alert('Vibe Flow failed: ' + err.message);
    } finally {
      setTimeout(() => {
        btn.disabled = false;
        btn.innerHTML = '&#127919; Vibe Flow';
      }, 3000);
    }
  }

  $('btn-vibe-flow').addEventListener('click', () => {
    doVibeFlow($('semantic-input').value);
  });

  /* ==================================================================
     SONIC MAP - 2D visualization of latent space
     ================================================================== */
  const sonicMap = {
    canvas: null,
    ctx: null,
    tracks: [],  // {id, title, x, y}
    trackMap: {}, // id -> track
    loaded: false,
    hoverTrack: null,

    init() {
      this.canvas = $('sonic-map-canvas');
      if (!this.canvas) return;
      this.ctx = this.canvas.getContext('2d');

      // Tab switching
      document.querySelectorAll('.wa-panel-tab').forEach(tab => {
        tab.addEventListener('click', () => {
          document.querySelectorAll('.wa-panel-tab').forEach(t => t.classList.remove('active'));
          tab.classList.add('active');
          const panel = tab.dataset.panel;
          $('wa-playlist').classList.toggle('hidden', panel !== 'playlist');
          $('wa-sonic-map').classList.toggle('hidden', panel !== 'map');
          if (panel === 'map' && !this.loaded) this.load();
          if (panel === 'map') this.render();
        });
      });

      // Mouse events
      this.canvas.addEventListener('mousemove', e => this.onMouseMove(e));
      this.canvas.addEventListener('mouseleave', () => this.hideTooltip());
      this.canvas.addEventListener('click', e => this.onClick(e));

      // Resize handler
      new ResizeObserver(() => { if (this.loaded) this.render(); }).observe(this.canvas.parentElement);
    },

    async load() {
      try {
        const resp = await fetch('/api/sonic-map');
        const data = await resp.json();
        if (data.available && data.tracks) {
          this.tracks = data.tracks;
          this.trackMap = {};
          this.tracks.forEach(t => this.trackMap[t.id] = t);
          this.loaded = true;
          this.render();
        }
      } catch (err) { console.error('Sonic map load error:', err); }
    },

    render() {
      if (!this.ctx || !this.tracks.length) return;
      const rect = this.canvas.parentElement.getBoundingClientRect();
      this.canvas.width = rect.width;
      this.canvas.height = rect.height;
      const ctx = this.ctx;
      const w = this.canvas.width, h = this.canvas.height;
      const pad = 20;

      // Clear
      ctx.fillStyle = '#050510';
      ctx.fillRect(0, 0, w, h);

      // Get current and queued track IDs
      const currentId = player.currentTrack ? player.currentTrack.id : null;
      const queuedIds = new Set(player.playlist.map(t => t.id));

      // Draw queue path
      if (player.playlist.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0,255,100,0.3)';
        ctx.lineWidth = 1;
        let first = true;
        for (const pt of player.playlist) {
          const t = this.trackMap[pt.id];
          if (!t) continue;
          const x = pad + t.x * (w - 2*pad);
          const y = pad + t.y * (h - 2*pad);
          if (first) { ctx.moveTo(x, y); first = false; }
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Draw all tracks
      for (const t of this.tracks) {
        const x = pad + t.x * (w - 2*pad);
        const y = pad + t.y * (h - 2*pad);
        const isCurrent = t.id === currentId;
        const isQueued = queuedIds.has(t.id);

        ctx.beginPath();
        if (isCurrent) {
          ctx.fillStyle = '#ff3366';
          ctx.shadowColor = '#ff3366';
          ctx.shadowBlur = 8;
          ctx.arc(x, y, 6, 0, Math.PI * 2);
        } else if (isQueued) {
          ctx.fillStyle = '#00ff66';
          ctx.shadowColor = '#00ff66';
          ctx.shadowBlur = 4;
          ctx.arc(x, y, 4, 0, Math.PI * 2);
        } else {
          ctx.fillStyle = '#334';
          ctx.shadowBlur = 0;
          ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        }
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    },

    getTrackAt(x, y) {
      const rect = this.canvas.getBoundingClientRect();
      const mx = x - rect.left, my = y - rect.top;
      const w = this.canvas.width, h = this.canvas.height;
      const pad = 20;
      for (const t of this.tracks) {
        const tx = pad + t.x * (w - 2*pad);
        const ty = pad + t.y * (h - 2*pad);
        const dist = Math.sqrt((mx - tx)**2 + (my - ty)**2);
        if (dist < 10) return t;
      }
      return null;
    },

    onMouseMove(e) {
      const t = this.getTrackAt(e.clientX, e.clientY);
      if (t) {
        this.hoverTrack = t;
        const tooltip = $('sonic-map-tooltip');
        tooltip.textContent = t.title;
        tooltip.style.display = 'block';
        const rect = this.canvas.getBoundingClientRect();
        tooltip.style.left = (e.clientX - rect.left + 10) + 'px';
        tooltip.style.top = (e.clientY - rect.top - 20) + 'px';
      } else {
        this.hideTooltip();
      }
    },

    hideTooltip() {
      this.hoverTrack = null;
      $('sonic-map-tooltip').style.display = 'none';
    },

    onClick(e) {
      const t = this.getTrackAt(e.clientX, e.clientY);
      if (t) {
        // Add to queue and play
        player.addToQueue({id: t.id, title: t.title});
        const idx = player.playlist.findIndex(pt => pt.id === t.id);
        if (idx >= 0) {
          player.currentIndex = idx;
          player.loadAndPlay();
        }
        this.render();
      }
    },

    // Re-render when playlist changes
    onPlaylistChange() {
      if (this.loaded && !$('wa-sonic-map').classList.contains('hidden')) {
        this.render();
      }
    }
  };

  // ---- Init ----
  player.init();
  sonicMap.init();
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
    parser.add_argument("--auto-embed", action="store_true",
                        help="Automatically embed new audio files on startup")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    app_state["output_dir"] = output_dir
    app_state["playlist_url"] = args.url

    print(f"Output directory: {output_dir}")

    # Load completed and failed progress immediately (fast, local files)
    app_state["completed"] = load_progress(output_dir)
    app_state["failed"] = load_failed(output_dir)

    # Scan local library on startup
    local_tracks = scan_downloaded_tracks()
    print(f"Local library: {len(local_tracks)} tracks ready to play")

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

    # Auto-embed new music files on startup
    def check_and_embed_new_files():
        """Check for new audio files and start embedding them."""
        if not _HAS_RECOMMENDER:
            return

        try:
            from recommender import find_audio_files, load_segment_embeddings, cmd_embed_segments
            import argparse as _argparse

            audio_files = find_audio_files(output_dir)
            if not audio_files:
                return

            # Check what's already embedded
            seg_data = load_segment_embeddings(output_dir)
            embedded_ids = set(seg_data["ids"]) if seg_data else set()

            # Find new files
            new_files = [(vid, p) for vid, p in audio_files if vid not in embedded_ids]
            if not new_files:
                print(f"All {len(audio_files)} audio files already embedded.")
                return

            print(f"\nFound {len(new_files)} new audio files to embed (of {len(audio_files)} total)")
            broadcast_sse("embedding_started", {"new_count": len(new_files), "total_count": len(audio_files)})

            # Create args object for cmd_embed_segments
            embed_args = _argparse.Namespace(
                output=str(output_dir),
                device="cuda"
            )

            # Run embedding (this will take a while)
            cmd_embed_segments(embed_args)

            print("Embedding complete!")
            broadcast_sse("embedding_complete", {"count": len(new_files)})

        except Exception as e:
            print(f"Auto-embedding error: {e}")
            broadcast_sse("embedding_error", {"error": str(e)})

    if args.auto_embed:
        print("Auto-embedding enabled, checking for new files...")
        threading.Thread(target=check_and_embed_new_files, daemon=True).start()

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
