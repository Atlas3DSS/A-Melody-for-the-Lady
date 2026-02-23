#!/usr/bin/env python3
"""
Batch playlist downloader for large YouTube playlists.
Downloads audio-only at best quality with metadata, track info, and lyrics.
Processes in batches to avoid API throttling.
"""

import subprocess
import json
import time
import sys
import os
import random
import argparse
from pathlib import Path

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLuDePDS-QMFF5yrciQc5qwbWOTvPutJW5"
DEFAULT_OUTPUT_DIR = Path.home() / "Music" / "playlist_download"
BATCH_SIZE = 25
PAUSE_BETWEEN_BATCHES = (30, 60)  # random sleep range in seconds
PAUSE_BETWEEN_TRACKS = (2, 5)
MAX_RETRIES = 3
PROGRESS_FILE_NAME = ".download_progress.json"


def get_playlist_entries(url: str) -> list[dict]:
    """Fetch the flat list of video IDs and titles from the playlist."""
    print(f"Fetching playlist metadata from:\n  {url}\n")
    result = subprocess.run(
        [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error fetching playlist:\n{result.stderr}")
        sys.exit(1)

    entries = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        entries.append({
            "id": data.get("id"),
            "title": data.get("title", "Unknown"),
            "url": data.get("url") or data.get("id"),
        })

    print(f"Found {len(entries)} tracks in playlist.\n")
    return entries


def load_progress(output_dir: Path) -> set[str]:
    """Load set of already-downloaded video IDs."""
    progress_file = output_dir / PROGRESS_FILE_NAME
    if progress_file.exists():
        data = json.loads(progress_file.read_text())
        return set(data.get("completed", []))
    return set()


def save_progress(output_dir: Path, completed: set[str]):
    """Persist the set of completed video IDs."""
    progress_file = output_dir / PROGRESS_FILE_NAME
    progress_file.write_text(json.dumps({
        "completed": sorted(completed),
        "count": len(completed),
    }, indent=2))


def download_track(video_id: str, output_dir: Path) -> bool:
    """Download a single track with full metadata and lyrics."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        # Audio only, best quality
        "-x",
        "--audio-format", "opus",
        "--audio-quality", "0",

        # Embed everything available
        "--embed-metadata",
        "--embed-thumbnail",

        # Write external metadata files
        "--write-info-json",
        "--write-thumbnail",
        "--write-description",

        # Subtitles / lyrics — keep it minimal to avoid 429s
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en",
        "--sub-format", "srt/vtt/best",

        # Don't fail the whole download if subs/thumbnail/etc error out
        "--ignore-errors",

        # Output template: Artist - Title or fallback to uploader - title
        "-o", str(output_dir / "%(playlist_index|00)04d - %(artist,uploader|Unknown)s - %(title)s [%(id)s].%(ext)s"),

        # Don't re-download
        "--no-overwrites",

        # Throttle-friendly options
        "--sleep-requests", "1.5",
        "--sleep-subtitles", "3",
        "--no-warnings",
        "--retries", "5",
        "--fragment-retries", "5",

        # Geo bypass just in case
        "--geo-bypass",

        video_url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # With --ignore-errors, yt-dlp may return non-zero even if the audio
    # downloaded fine (e.g. subtitle fetch failed). Check if the audio
    # file actually landed on disk.
    if result.returncode == 0:
        return True

    # Check if an audio file was actually written despite the error
    tag = f"[{video_id}]"
    for f in output_dir.iterdir():
        if tag in f.name and f.suffix in (".opus", ".m4a", ".mp3", ".ogg", ".webm"):
            return True

    err = result.stderr.strip()
    if err:
        print(f"    yt-dlp stderr: {err[:300]}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Batch download a YouTube playlist as audio.")
    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Tracks per batch (default: {BATCH_SIZE})")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip the first N tracks in the playlist (0-indexed)")
    parser.add_argument("--url", type=str, default=PLAYLIST_URL,
                        help="Playlist URL to download")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just list tracks, don't download")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch full playlist
    entries = get_playlist_entries(args.url)

    if args.start_from > 0:
        entries = entries[args.start_from:]
        print(f"Skipping first {args.start_from} tracks, {len(entries)} remaining.\n")

    # Load progress to skip already-downloaded tracks
    completed = load_progress(output_dir)
    remaining = [e for e in entries if e["id"] not in completed]

    print(f"Already downloaded: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Batch size: {args.batch_size}\n")

    if args.dry_run:
        for i, entry in enumerate(remaining[:50], 1):
            print(f"  {i}. {entry['title']} ({entry['id']})")
        if len(remaining) > 50:
            print(f"  ... and {len(remaining) - 50} more")
        return

    # Process in batches
    total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
    failed = []

    for batch_num in range(total_batches):
        batch_start = batch_num * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]

        print(f"{'='*60}")
        print(f"BATCH {batch_num + 1}/{total_batches}  "
              f"(tracks {batch_start + 1}-{batch_end} of {len(remaining)})")
        print(f"{'='*60}\n")

        for i, entry in enumerate(batch, 1):
            vid_id = entry["id"]
            title = entry["title"]
            global_idx = len(completed) + 1
            print(f"  [{i}/{len(batch)}] (#{global_idx} overall) {title}")

            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                if attempt > 1:
                    wait = 10 * attempt
                    print(f"    Retry {attempt}/{MAX_RETRIES} in {wait}s...")
                    time.sleep(wait)

                if download_track(vid_id, output_dir):
                    success = True
                    break

            if success:
                print(f"    Done.")
                completed.add(vid_id)
                save_progress(output_dir, completed)
            else:
                print(f"    FAILED after {MAX_RETRIES} attempts. Skipping.")
                failed.append(entry)

            # Small pause between individual tracks
            if i < len(batch):
                pause = random.uniform(*PAUSE_BETWEEN_TRACKS)
                time.sleep(pause)

        # Pause between batches
        if batch_num < total_batches - 1:
            pause = random.uniform(*PAUSE_BETWEEN_BATCHES)
            print(f"\nBatch complete. Pausing {pause:.0f}s before next batch...\n")
            time.sleep(pause)

    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  Total downloaded: {len(completed)}")
    print(f"  Failed this run:  {len(failed)}")
    print(f"  Output directory: {output_dir}")

    if failed:
        failed_file = output_dir / "failed_tracks.json"
        failed_file.write_text(json.dumps(failed, indent=2))
        print(f"  Failed tracks saved to: {failed_file}")
        print("\n  Failed tracks:")
        for entry in failed:
            print(f"    - {entry['title']} ({entry['id']})")

    print()


if __name__ == "__main__":
    main()
