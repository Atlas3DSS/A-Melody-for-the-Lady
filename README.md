# A Melody for the Lady

A YouTube playlist downloader with a built-in Winamp-style audio player. Downloads audio at the best possible quality with full metadata, thumbnails, and lyrics — then lets you play everything right in your browser.

Built for a 2,000+ song playlist. Handles batching, rate-limiting, and resume automatically.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![yt-dlp](https://img.shields.io/badge/yt--dlp-powered-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

### Downloader
- **Best quality audio** — extracts and converts to Opus (best quality-to-size ratio)
- **Full metadata** — artist, title, album art embedded into each file
- **Lyrics & subtitles** — grabs available captions/lyrics
- **Smart batching** — processes in batches of 25 with random pauses to avoid API throttling
- **Resume support** — tracks progress in `.download_progress.json`, pick up where you left off
- **Retry logic** — 3 retries with exponential backoff on failures

### Web UI
- **Track browser** — view all tracks with download status (downloaded/pending/failed)
- **Search & filter** — find tracks by title, filter by status
- **Selective download** — cherry-pick which tracks to grab
- **Live progress** — real-time download progress via Server-Sent Events

### Winamp Player
- **Classic Winamp 2.x aesthetic** — dark metallic UI, green LCD display, beveled buttons
- **Spectrum visualizer** — 32-bar frequency analyzer via Web Audio API
- **Scrolling marquee** — track title scrolls across the LCD
- **Full transport controls** — play, pause, stop, prev, next
- **Seek & volume** — full seeking support with HTTP Range requests
- **Shuffle / Repeat / Repeat One** — all the classics
- **Playlist panel** — queue tracks, reorder, remove, click to jump

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the web UI

```bash
python3 playlist_server.py --no-browser
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

### Or run headless (CLI only)

```bash
python3 playlist_dl.py
```

## Usage

### Web server options

```
python3 playlist_server.py [options]

  -o, --output DIR      Output directory (default: ~/Music/playlist_download)
  -p, --port PORT       Port to serve on (default: 8080)
  --no-browser          Don't auto-open browser
  --url URL             Override playlist URL
```

### CLI downloader options

```
python3 playlist_dl.py [options]

  -o, --output DIR      Output directory (default: ~/Music/playlist_download)
  -b, --batch-size N    Tracks per batch (default: 25)
  --start-from N        Skip the first N tracks
  --url URL             Playlist URL to download
  --dry-run             Just list tracks, don't download
```

## How It Works

1. Fetches the full playlist metadata via yt-dlp
2. Compares against `.download_progress.json` to find what's already done
3. Downloads in batches with pauses between tracks (2-5s) and batches (30-60s)
4. Each track gets: `.opus` audio, `.info.json` metadata, thumbnail, description, subtitles
5. The web UI streams audio directly from the download directory with HTTP Range support

## Tech Stack

- **Python stdlib** — `http.server`, `threading`, `json` (zero web framework dependencies)
- **yt-dlp** — audio extraction and metadata
- **mutagen** — metadata embedding
- **Web Audio API** — spectrum visualizer
- **Server-Sent Events** — real-time download progress
- **Single-page HTML** — everything embedded in one Python file, no build step
