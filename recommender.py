#!/usr/bin/env python3
"""
Music recommendation engine using dual audio embeddings.

Encoders:
  - ACE-Step VAE: acoustic/timbral similarity (64-dim latent, mean-pooled)
  - CLAP (laion/larger_clap_music): semantic/genre/mood similarity (512-dim)

Usage:
  python recommender.py embed [-o DIR] [--device cuda]
  python recommender.py similar <video_id> [-n 20]
  python recommender.py search "dark techno heavy bass" [-n 20]
  python recommender.py info

Requires the dev_genius venv:
  source /home/orwel/dev_genius/venv/bin/activate
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# Lazy imports for heavy libs so CLI is fast for query commands
_torch = None
_librosa = None


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _import_librosa():
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MUSIC_DIR = Path.home() / "Music" / "playlist_download"
# ACE-Step VAE path - check both Linux and Windows locations
_LINUX_VAE_PATH = Path("/home/orwel/dev_genius/ACE-Step-1.5/checkpoints/vae")
_WIN_VAE_PATH = Path.home() / "dev_genius" / "ACE-Step-1.5" / "checkpoints" / "vae"
ACE_VAE_PATH = _LINUX_VAE_PATH if _LINUX_VAE_PATH.exists() else _WIN_VAE_PATH
VAE_AVAILABLE = ACE_VAE_PATH.exists()

CLAP_MODEL_ID = "laion/larger_clap_music"
EMBEDDINGS_FILE = "embeddings.npz"
SEGMENT_EMBEDDINGS_FILE = "segment_embeddings.npz"  # Full track segmented embeddings
DISCOVERY_EMBEDDINGS_FILE = "discovery_embeddings.npz"
PREVIEW_DIR_NAME = "previews"
AUDIO_EXTENSIONS = (".opus", ".mp3", ".m4a", ".ogg", ".webm")
SAMPLE_RATE_VAE = 48000   # ACE-Step VAE expects 48kHz stereo
SAMPLE_RATE_CLAP = 48000  # CLAP processor handles resampling
MAX_DURATION = 600         # seconds — increased to embed longer tracks fully
SEGMENT_DURATION = 10      # seconds — each segment for CLAP embedding
PREVIEW_DURATION = 30      # seconds — short clips for discovery embedding


def extract_video_id(filename: str) -> str | None:
    """Extract [video_id] from a filename like '0001 - Artist - Title [abc123].opus'."""
    match = re.search(r'\[([a-zA-Z0-9_-]+)\]', filename)
    return match.group(1) if match else None


def find_audio_files(music_dir: Path) -> list[tuple[str, Path]]:
    """Return list of (video_id, path) for all audio files."""
    results = []
    for f in sorted(music_dir.iterdir()):
        if f.suffix in AUDIO_EXTENSIONS:
            vid_id = extract_video_id(f.name)
            if vid_id:
                results.append((vid_id, f))
    return results


# ---------------------------------------------------------------------------
# Discovery: YouTube Music related tracks + preview download
# ---------------------------------------------------------------------------
def get_related_tracks(video_id: str, limit: int = 25) -> list[dict]:
    """Get related tracks from YouTube Music for a given video ID."""
    from ytmusicapi import YTMusic
    yt = YTMusic()
    try:
        results = yt.get_watch_playlist(videoId=video_id, radio=True, limit=limit)
        tracks = []
        for t in results.get("tracks", []):
            vid = t.get("videoId")
            if not vid or vid == video_id:
                continue
            artists = ", ".join(a.get("name", "") for a in t.get("artists", []))
            tracks.append({
                "id": vid,
                "title": t.get("title", "Unknown"),
                "artist": artists,
                "source_id": video_id,
            })
        return tracks
    except Exception as e:
        print(f"    YTMusic error for {video_id}: {e}")
        return []


def download_preview(video_id: str, preview_dir: Path,
                     duration: int = PREVIEW_DURATION) -> Path | None:
    """Download a short audio preview for embedding."""
    import subprocess
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    for ext in AUDIO_EXTENSIONS:
        candidate = preview_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate

    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "opus", "--audio-quality", "5",
        "--download-sections", f"*00:00:00-00:00:{duration:02d}",
        "-o", str(preview_dir / f"{video_id}.%(ext)s"),
        "--no-overwrites", "--no-warnings", "--quiet",
        "--retries", "3",
        "--sleep-requests", "1",
        "--geo-bypass",
        f"https://www.youtube.com/watch?v={video_id}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Find the downloaded file
    for ext in AUDIO_EXTENSIONS:
        candidate = preview_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate

    if result.returncode != 0 and result.stderr:
        print(f"    preview dl error: {result.stderr[:200]}")
    return None


def load_discovery_embeddings(music_dir: Path) -> dict | None:
    """Load discovery embeddings from disk."""
    path = music_dir / DISCOVERY_EMBEDDINGS_FILE
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "ids": list(data["ids"]),
        "titles": list(data["titles"]),
        "artists": list(data["artists"]),
        "source_ids": list(data["source_ids"]),
        "vae": data["vae"],
        "clap": data["clap"],
    }


def save_discovery_embeddings(music_dir: Path, disc_data: dict):
    """Save discovery embeddings to disk."""
    np.savez_compressed(
        music_dir / DISCOVERY_EMBEDDINGS_FILE,
        ids=np.array(disc_data["ids"]),
        titles=np.array(disc_data["titles"]),
        artists=np.array(disc_data["artists"]),
        source_ids=np.array(disc_data["source_ids"]),
        vae=disc_data["vae"],
        clap=disc_data["clap"],
    )


# ---------------------------------------------------------------------------
# Embedding: ACE-Step VAE
# ---------------------------------------------------------------------------
def load_vae(device="cuda"):
    """Load the ACE-Step VAE encoder."""
    torch = _import_torch()
    from diffusers import AutoencoderOobleck

    print(f"Loading ACE-Step VAE from {ACE_VAE_PATH}...")
    vae = AutoencoderOobleck.from_pretrained(str(ACE_VAE_PATH))
    vae = vae.to(device).eval()
    print(f"  VAE loaded on {device}")
    return vae


def encode_vae(vae, audio_path: Path, device="cuda") -> np.ndarray | None:
    """Encode a single audio file to a mean-pooled VAE latent vector."""
    torch = _import_torch()
    librosa = _import_librosa()

    try:
        # Load as stereo 48kHz
        wav, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE_VAE, mono=False,
                               duration=MAX_DURATION)
        if wav.ndim == 1:
            wav = np.stack([wav, wav])  # mono -> stereo
        if wav.shape[0] > 2:
            wav = wav[:2]  # truncate extra channels

        # Shape: [1, channels, samples]
        audio_tensor = torch.from_numpy(wav).unsqueeze(0).float().to(device)

        with torch.no_grad():
            latent = vae.encode(audio_tensor).latent_dist.sample()
            # latent shape: [1, latent_channels, time_frames]
            # Mean-pool over time -> [latent_channels]
            embedding = latent.squeeze(0).mean(dim=-1).cpu().numpy()

        return embedding.astype(np.float32)
    except Exception as e:
        print(f"    VAE encode error: {e}")
        return None


# ---------------------------------------------------------------------------
# Embedding: CLAP
# ---------------------------------------------------------------------------
def load_clap(device="cuda"):
    """Load the CLAP model and processor."""
    torch = _import_torch()
    from transformers import ClapModel, ClapProcessor

    print(f"Loading CLAP model ({CLAP_MODEL_ID})...")
    model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device).eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    print(f"  CLAP loaded on {device}")
    return model, processor


def encode_clap(model, processor, audio_path: Path, device="cuda") -> np.ndarray | None:
    """Encode a single audio file to a CLAP embedding."""
    torch = _import_torch()
    librosa = _import_librosa()

    try:
        wav, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE_CLAP, mono=True,
                               duration=MAX_DURATION)

        inputs = processor(audio=wav, sampling_rate=SAMPLE_RATE_CLAP,
                           return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = model.get_audio_features(**inputs)
            embedding = embedding.squeeze(0).cpu().numpy()

        return embedding.astype(np.float32)
    except Exception as e:
        print(f"    CLAP encode error: {e}")
        return None


def encode_clap_segments(model, processor, audio_path: Path, device="cuda",
                         segment_duration: float = SEGMENT_DURATION) -> list[dict] | None:
    """
    Encode an entire audio file into multiple segment embeddings.

    Returns a list of dicts, each containing:
      - 'embedding': np.ndarray (512-dim CLAP embedding)
      - 'start_time': float (seconds)
      - 'end_time': float (seconds)
      - 'segment_idx': int

    This allows matching against ANY part of the track, not just the intro.
    """
    torch = _import_torch()
    librosa = _import_librosa()

    try:
        # Load full audio (up to MAX_DURATION)
        wav, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE_CLAP, mono=True,
                               duration=MAX_DURATION)

        total_duration = len(wav) / sr
        samples_per_segment = int(segment_duration * sr)

        segments = []
        segment_idx = 0

        # Process in overlapping windows for better coverage
        # Use 50% overlap to catch transitions
        hop_samples = samples_per_segment // 2

        for start_sample in range(0, len(wav) - samples_per_segment // 2, hop_samples):
            end_sample = min(start_sample + samples_per_segment, len(wav))
            segment_wav = wav[start_sample:end_sample]

            # Skip very short segments at the end
            if len(segment_wav) < samples_per_segment // 2:
                continue

            # Pad short final segment if needed
            if len(segment_wav) < samples_per_segment:
                segment_wav = np.pad(segment_wav, (0, samples_per_segment - len(segment_wav)))

            inputs = processor(audio=segment_wav, sampling_rate=SAMPLE_RATE_CLAP,
                               return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                embedding = model.get_audio_features(**inputs)
                embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)

            segments.append({
                'embedding': embedding,
                'start_time': start_sample / sr,
                'end_time': end_sample / sr,
                'segment_idx': segment_idx,
            })
            segment_idx += 1

        return segments if segments else None

    except Exception as e:
        print(f"    CLAP segment encode error: {e}")
        return None


def encode_text_clap(model, processor, text: str, device="cuda") -> np.ndarray:
    """Encode a text query to a CLAP embedding."""
    torch = _import_torch()

    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
        embedding = embedding.squeeze(0).cpu().numpy()

    return embedding.astype(np.float32)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of query vector against each row of matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid div by zero
    normed = matrix / norms
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    query_normed = query / query_norm
    return normed @ query_normed


def load_embeddings(music_dir: Path) -> dict:
    """Load saved embeddings from disk."""
    path = music_dir / EMBEDDINGS_FILE
    if not path.exists():
        print(f"No embeddings found at {path}")
        print("Run 'python recommender.py embed' first.")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    return {
        "ids": list(data["ids"]),
        "titles": list(data["titles"]),
        "vae": data["vae"],      # [N, 64]
        "clap": data["clap"],    # [N, 512]
    }


def load_segment_embeddings(music_dir: Path) -> dict | None:
    """Load segment embeddings from disk.

    Returns dict with:
      Track-level (for fast search):
      - ids: list of unique track IDs
      - titles: list of track titles
      - mean_emb: (N_tracks, 512) mean of all segments per track
      - max_emb: (N_tracks, 512) element-wise max of all segments per track
      - std_emb: (N_tracks, 512) std deviation of segments (captures variation)

      Segment-level (for fine-grained matching):
      - segment_ids: list of track IDs (one per segment)
      - segment_times: (N_segments, 2) array of [start_time, end_time]
      - segment_emb: (N_segments, 512) individual segment embeddings
    """
    path = music_dir / SEGMENT_EMBEDDINGS_FILE
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        # Track-level aggregated
        "ids": list(data["ids"]),
        "titles": list(data["titles"]),
        "mean_emb": data["mean_emb"],
        "max_emb": data["max_emb"],
        "std_emb": data["std_emb"],
        # Segment-level
        "segment_ids": list(data["segment_ids"]),
        "segment_times": data["segment_times"],
        "segment_emb": data["segment_emb"],
    }


def save_segment_embeddings(music_dir: Path, seg_data: dict):
    """Save segment embeddings to disk."""
    path = music_dir / SEGMENT_EMBEDDINGS_FILE
    np.savez_compressed(
        path,
        # Track-level
        ids=np.array(seg_data["ids"], dtype=object),
        titles=np.array(seg_data["titles"], dtype=object),
        mean_emb=np.array(seg_data["mean_emb"], dtype=np.float32),
        max_emb=np.array(seg_data["max_emb"], dtype=np.float32),
        std_emb=np.array(seg_data["std_emb"], dtype=np.float32),
        # Segment-level
        segment_ids=np.array(seg_data["segment_ids"], dtype=object),
        segment_times=np.array(seg_data["segment_times"], dtype=np.float32),
        segment_emb=np.array(seg_data["segment_emb"], dtype=np.float32),
    )


def find_similar_segments(seg_data: dict, video_id: str, n: int = 20,
                          method: str = "hybrid") -> list[dict]:
    """Find similar tracks using segment-level embeddings.

    Methods:
      - "mean": Use mean-pooled track embeddings (fast, general context)
      - "max": Use max-pooled track embeddings (emphasizes prominent features)
      - "meanmax": Concatenate mean+max for richer representation
      - "segment": Match against individual segments, take best per track
      - "hybrid": Combine meanmax track-level with segment-level refinement (best)
    """
    ids = seg_data["ids"]
    titles = seg_data["titles"]
    mean_emb = seg_data["mean_emb"]
    max_emb = seg_data["max_emb"]
    segment_ids = seg_data["segment_ids"]
    segment_emb = seg_data["segment_emb"]
    segment_times = seg_data["segment_times"]

    if video_id not in ids:
        return []

    query_idx = ids.index(video_id)

    if method == "mean":
        query_vec = mean_emb[query_idx]
        scores = cosine_sim_batch(mean_emb, query_vec)
    elif method == "max":
        query_vec = max_emb[query_idx]
        scores = cosine_sim_batch(max_emb, query_vec)
    elif method == "meanmax":
        # Concatenate mean and max for richer representation
        corpus = np.concatenate([mean_emb, max_emb], axis=1)  # (N, 1024)
        query_vec = np.concatenate([mean_emb[query_idx], max_emb[query_idx]])
        scores = cosine_sim_batch(corpus, query_vec)
    elif method == "segment":
        # Pure segment-level matching
        query_seg_indices = [i for i, sid in enumerate(segment_ids) if sid == video_id]
        query_seg_embs = segment_emb[query_seg_indices]
        query_vec = query_seg_embs.mean(axis=0)

        # Score all segments
        all_scores = cosine_sim_batch(segment_emb, query_vec)

        # Take max score per track
        track_scores = {}
        track_best_seg = {}
        for i, (sid, score) in enumerate(zip(segment_ids, all_scores)):
            if sid == video_id:
                continue
            if sid not in track_scores or score > track_scores[sid]:
                track_scores[sid] = score
                track_best_seg[sid] = i

        # Build results
        sorted_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        results = []
        for sid, score in sorted_tracks:
            seg_idx = track_best_seg[sid]
            tid = ids.index(sid) if sid in ids else 0
            results.append({
                "id": sid,
                "title": titles[tid] if tid < len(titles) else sid,
                "score": float(score),
                "match_time": float(segment_times[seg_idx][0]),
                "source": "segment",
            })
        return results
    else:  # hybrid - best of both worlds
        # Stage 1: Fast track-level search with meanmax
        corpus = np.concatenate([mean_emb, max_emb], axis=1)
        query_vec = np.concatenate([mean_emb[query_idx], max_emb[query_idx]])
        track_scores = cosine_sim_batch(corpus, query_vec)

        # Stage 2: Refine top candidates with segment-level matching
        # Get top 2N candidates for refinement
        candidates = np.argsort(track_scores)[::-1][1:n*2+1]  # Skip self

        # Get query's segment embeddings
        query_seg_indices = [i for i, sid in enumerate(segment_ids) if sid == video_id]
        query_seg_embs = segment_emb[query_seg_indices]
        query_seg_mean = query_seg_embs.mean(axis=0)

        refined_scores = {}
        best_match_times = {}
        for cand_idx in candidates:
            cand_id = ids[cand_idx]
            cand_seg_indices = [i for i, sid in enumerate(segment_ids) if sid == cand_id]
            if not cand_seg_indices:
                refined_scores[cand_id] = track_scores[cand_idx]
                best_match_times[cand_id] = 0.0
                continue

            # Find best matching segment
            cand_seg_embs = segment_emb[cand_seg_indices]
            seg_scores = cosine_sim_batch(cand_seg_embs, query_seg_mean)
            best_seg_local = np.argmax(seg_scores)
            best_seg_global = cand_seg_indices[best_seg_local]

            # Combine track-level and segment-level scores (0.6 track + 0.4 segment)
            refined_scores[cand_id] = 0.6 * track_scores[cand_idx] + 0.4 * seg_scores[best_seg_local]
            best_match_times[cand_id] = float(segment_times[best_seg_global][0])

        # Sort by refined score
        sorted_tracks = sorted(refined_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        results = []
        for sid, score in sorted_tracks:
            tid = ids.index(sid)
            results.append({
                "id": sid,
                "title": titles[tid],
                "score": float(score),
                "match_time": best_match_times.get(sid, 0.0),
                "source": "hybrid",
            })
        return results

    # For non-segment methods, build results from track-level scores
    sorted_indices = np.argsort(scores)[::-1]
    results = []
    for idx in sorted_indices:
        if ids[idx] == video_id:
            continue
        results.append({
            "id": ids[idx],
            "title": titles[idx],
            "score": float(scores[idx]),
            "match_time": 0.0,
            "source": method,
        })
        if len(results) >= n:
            break
    return results


def find_similar(emb_data: dict, video_id: str, n: int = 20,
                 alpha: float = 0.5, discovery_data: dict = None) -> list[dict]:
    """Find top-N similar tracks by blended cosine similarity.

    Searches both library embeddings and optionally discovery embeddings.
    """
    ids = emb_data["ids"]
    if video_id not in ids:
        # Also check discovery corpus
        if discovery_data and video_id in discovery_data["ids"]:
            return _find_similar_from_discovery(
                emb_data, discovery_data, video_id, n, alpha)
        print(f"Video ID '{video_id}' not found in embeddings.")
        return []

    idx = ids.index(video_id)
    query_vae = emb_data["vae"][idx]
    query_clap = emb_data["clap"][idx]

    results = []

    # Search library
    results += _score_corpus(
        query_vae, query_clap, emb_data, alpha, source="library", exclude_id=video_id)

    # Search discovery corpus if available
    if discovery_data and len(discovery_data["ids"]) > 0:
        results += _score_corpus(
            query_vae, query_clap, discovery_data, alpha, source="discovery")

    # Sort by score, return top N
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:n]


def _find_similar_from_discovery(emb_data, discovery_data, video_id, n, alpha):
    """Find similar when the query track is from the discovery corpus."""
    idx = discovery_data["ids"].index(video_id)
    query_vae = discovery_data["vae"][idx]
    query_clap = discovery_data["clap"][idx]

    results = []
    results += _score_corpus(
        query_vae, query_clap, emb_data, alpha, source="library")
    results += _score_corpus(
        query_vae, query_clap, discovery_data, alpha, source="discovery",
        exclude_id=video_id)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:n]


def _score_corpus(query_vae, query_clap, corpus, alpha, source="library",
                  exclude_id=None):
    """Score all tracks in a corpus against query embeddings."""
    ids = corpus["ids"]
    vae_all = corpus["vae"]
    clap_all = corpus["clap"]

    if len(ids) == 0:
        return []

    # Normalize
    vae_norms = np.linalg.norm(vae_all, axis=1, keepdims=True)
    vae_norms = np.where(vae_norms == 0, 1, vae_norms)
    vae_normed = vae_all / vae_norms

    clap_norms = np.linalg.norm(clap_all, axis=1, keepdims=True)
    clap_norms = np.where(clap_norms == 0, 1, clap_norms)
    clap_normed = clap_all / clap_norms

    # Normalize query
    qv_norm = np.linalg.norm(query_vae)
    qc_norm = np.linalg.norm(query_clap)
    qv = query_vae / qv_norm if qv_norm > 0 else query_vae
    qc = query_clap / qc_norm if qc_norm > 0 else query_clap

    vae_sims = vae_normed @ qv
    clap_sims = clap_normed @ qc
    blended = alpha * clap_sims + (1 - alpha) * vae_sims

    results = []
    for i in range(len(ids)):
        if ids[i] == exclude_id:
            continue
        entry = {
            "id": ids[i],
            "title": corpus["titles"][i],
            "score": float(blended[i]),
            "clap_score": float(clap_sims[i]),
            "vae_score": float(vae_sims[i]),
            "source": source,
        }
        if source == "discovery" and "artists" in corpus:
            entry["artist"] = corpus["artists"][i]
        results.append(entry)
    return results


def search_by_text(emb_data: dict, model, processor, text: str,
                   n: int = 20, device="cuda") -> list[dict]:
    """Search tracks by text description using CLAP text encoder."""
    text_emb = encode_text_clap(model, processor, text, device=device)
    clap_all = emb_data["clap"]

    sims = cosine_similarity_matrix(text_emb, clap_all)
    ranked = np.argsort(-sims)

    results = []
    for i in ranked[:n]:
        results.append({
            "id": emb_data["ids"][i],
            "title": emb_data["titles"][i],
            "score": float(sims[i]),
        })
    return results


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------
def cmd_embed(args):
    """Batch-encode all audio files."""
    torch = _import_torch()
    music_dir = Path(args.output)
    device = args.device
    use_vae = VAE_AVAILABLE and not getattr(args, 'clap_only', False)

    audio_files = find_audio_files(music_dir)
    if not audio_files:
        print(f"No audio files found in {music_dir}")
        return

    print(f"Found {len(audio_files)} audio files in {music_dir}")
    if not use_vae:
        print("  NOTE: Running in CLAP-only mode (VAE not available or --clap-only specified)")

    # Load existing embeddings to resume
    emb_path = music_dir / EMBEDDINGS_FILE
    existing_ids = set()
    existing_data = None
    if emb_path.exists():
        existing_data = np.load(emb_path, allow_pickle=True)
        existing_ids = set(existing_data["ids"])
        print(f"Resuming: {len(existing_ids)} already embedded")

    new_files = [(vid, p) for vid, p in audio_files if vid not in existing_ids]
    if not new_files:
        print("All files already embedded. Nothing to do.")
        return

    print(f"Encoding {len(new_files)} new files...\n")

    # Load models
    vae = load_vae(device=device) if use_vae else None
    clap_model, clap_processor = load_clap(device=device)
    print()

    # Encode
    new_ids = []
    new_titles = []
    new_vae = []
    new_clap = []
    failed = []

    for i, (vid_id, fpath) in enumerate(new_files):
        title = fpath.stem  # filename without extension
        pct = (i + 1) / len(new_files) * 100
        # Handle Unicode characters that Windows console can't display
        safe_title = title[:60].encode('ascii', 'replace').decode('ascii')
        print(f"[{i+1}/{len(new_files)}] ({pct:.0f}%) {safe_title}")

        v_emb = encode_vae(vae, fpath, device=device) if use_vae else None
        c_emb = encode_clap(clap_model, clap_processor, fpath, device=device)

        # In CLAP-only mode, only c_emb is required
        if use_vae:
            success = v_emb is not None and c_emb is not None
        else:
            success = c_emb is not None

        if success:
            new_ids.append(vid_id)
            new_titles.append(title)
            if use_vae:
                new_vae.append(v_emb)
            new_clap.append(c_emb)
        else:
            failed.append(vid_id)
            print(f"    SKIPPED (encode failed)")

        # Clear CUDA cache periodically
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # Merge with existing
    if existing_data is not None and len(existing_data["ids"]) > 0:
        all_ids = list(existing_data["ids"]) + new_ids
        all_titles = list(existing_data["titles"]) + new_titles
        if use_vae:
            all_vae = np.concatenate([existing_data["vae"]] + ([np.stack(new_vae)] if new_vae else []))
        else:
            # Preserve existing VAE if present, or use zeros
            if "vae" in existing_data and len(existing_data["vae"]) > 0:
                # Pad with zeros for new entries
                vae_dim = existing_data["vae"].shape[1]
                new_zeros = np.zeros((len(new_ids), vae_dim), dtype=np.float32)
                all_vae = np.concatenate([existing_data["vae"], new_zeros])
            else:
                all_vae = np.zeros((len(all_ids), 64), dtype=np.float32)
        all_clap = np.concatenate([existing_data["clap"]] + ([np.stack(new_clap)] if new_clap else []))
    else:
        all_ids = new_ids
        all_titles = new_titles
        if use_vae:
            all_vae = np.stack(new_vae) if new_vae else np.zeros((0, 64), dtype=np.float32)
        else:
            all_vae = np.zeros((len(new_ids), 64), dtype=np.float32)  # placeholder zeros
        all_clap = np.stack(new_clap) if new_clap else np.zeros((0, 512), dtype=np.float32)

    # Save
    np.savez_compressed(
        emb_path,
        ids=np.array(all_ids),
        titles=np.array(all_titles),
        vae=all_vae,
        clap=all_clap,
    )

    print(f"\nDone! Saved {len(all_ids)} embeddings to {emb_path}")
    if failed:
        print(f"Failed: {len(failed)} tracks")
    if use_vae:
        print(f"  VAE dims:  {all_vae.shape if len(all_vae) else 'empty'}")
    else:
        print(f"  VAE dims:  CLAP-only mode (zeros placeholder)")
    print(f"  CLAP dims: {all_clap.shape if len(all_clap) else 'empty'}")


def cmd_embed_segments(args):
    """Create segment-level embeddings for entire tracks.

    This embeds the full audio in overlapping 10-second segments, storing:
    - Track-level: mean, max, std pooled embeddings (for fast search)
    - Segment-level: individual segment embeddings (for fine-grained matching)

    Based on MIR research showing MeanMax pooling outperforms single strategies.
    """
    torch = _import_torch()
    music_dir = Path(args.output)
    device = args.device

    audio_files = find_audio_files(music_dir)
    if not audio_files:
        print(f"No audio files found in {music_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    print(f"Segment duration: {SEGMENT_DURATION}s with 50% overlap")
    print(f"Max track duration: {MAX_DURATION}s")

    # Load existing segment embeddings to resume
    seg_path = music_dir / SEGMENT_EMBEDDINGS_FILE
    existing_ids = set()
    existing_data = None
    if seg_path.exists():
        existing_data = load_segment_embeddings(music_dir)
        if existing_data:
            existing_ids = set(existing_data["ids"])
            print(f"Resuming: {len(existing_ids)} tracks already embedded")

    new_files = [(vid, p) for vid, p in audio_files if vid not in existing_ids]
    if not new_files:
        print("All files already have segment embeddings. Nothing to do.")
        return

    print(f"Processing {len(new_files)} new tracks...\n")

    # Load CLAP model
    clap_model, clap_processor = load_clap(device=device)
    print()

    # Process each track
    new_ids = []
    new_titles = []
    new_mean_emb = []
    new_max_emb = []
    new_std_emb = []
    all_segment_ids = []
    all_segment_times = []
    all_segment_emb = []
    failed = []

    for i, (vid_id, fpath) in enumerate(new_files):
        title = fpath.stem
        pct = (i + 1) / len(new_files) * 100
        safe_title = title[:50].encode('ascii', 'replace').decode('ascii')
        print(f"[{i+1}/{len(new_files)}] ({pct:.0f}%) {safe_title}")

        segments = encode_clap_segments(clap_model, clap_processor, fpath, device=device)

        if segments is None or len(segments) == 0:
            failed.append(vid_id)
            print(f"    SKIPPED (encode failed)")
            continue

        # Extract embeddings
        seg_embs = np.stack([s['embedding'] for s in segments])
        seg_times = np.array([[s['start_time'], s['end_time']] for s in segments])

        # Compute aggregated embeddings
        mean_emb = seg_embs.mean(axis=0)
        max_emb = seg_embs.max(axis=0)
        std_emb = seg_embs.std(axis=0)

        # Store track-level
        new_ids.append(vid_id)
        new_titles.append(title)
        new_mean_emb.append(mean_emb)
        new_max_emb.append(max_emb)
        new_std_emb.append(std_emb)

        # Store segment-level
        for seg_emb, (start, end) in zip(seg_embs, seg_times):
            all_segment_ids.append(vid_id)
            all_segment_times.append([start, end])
            all_segment_emb.append(seg_emb)

        print(f"    {len(segments)} segments ({seg_times[-1][1]:.1f}s total)")

        # Clear CUDA cache periodically
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Merge with existing data
    if existing_data is not None and len(existing_data["ids"]) > 0:
        all_ids = list(existing_data["ids"]) + new_ids
        all_titles = list(existing_data["titles"]) + new_titles
        all_mean = np.concatenate([existing_data["mean_emb"]] +
                                  ([np.stack(new_mean_emb)] if new_mean_emb else []))
        all_max = np.concatenate([existing_data["max_emb"]] +
                                 ([np.stack(new_max_emb)] if new_max_emb else []))
        all_std = np.concatenate([existing_data["std_emb"]] +
                                 ([np.stack(new_std_emb)] if new_std_emb else []))
        # Segment-level
        merged_seg_ids = list(existing_data["segment_ids"]) + all_segment_ids
        merged_seg_times = np.concatenate([existing_data["segment_times"]] +
                                          ([np.stack(all_segment_times)] if all_segment_times else []))
        merged_seg_emb = np.concatenate([existing_data["segment_emb"]] +
                                        ([np.stack(all_segment_emb)] if all_segment_emb else []))
    else:
        all_ids = new_ids
        all_titles = new_titles
        all_mean = np.stack(new_mean_emb) if new_mean_emb else np.zeros((0, 512), dtype=np.float32)
        all_max = np.stack(new_max_emb) if new_max_emb else np.zeros((0, 512), dtype=np.float32)
        all_std = np.stack(new_std_emb) if new_std_emb else np.zeros((0, 512), dtype=np.float32)
        merged_seg_ids = all_segment_ids
        merged_seg_times = np.stack(all_segment_times) if all_segment_times else np.zeros((0, 2), dtype=np.float32)
        merged_seg_emb = np.stack(all_segment_emb) if all_segment_emb else np.zeros((0, 512), dtype=np.float32)

    # Save
    save_segment_embeddings(music_dir, {
        "ids": all_ids,
        "titles": all_titles,
        "mean_emb": all_mean,
        "max_emb": all_max,
        "std_emb": all_std,
        "segment_ids": merged_seg_ids,
        "segment_times": merged_seg_times,
        "segment_emb": merged_seg_emb,
    })

    print(f"\nDone! Saved segment embeddings to {seg_path}")
    print(f"  Tracks: {len(all_ids)}")
    print(f"  Total segments: {len(merged_seg_ids)}")
    print(f"  Track embeddings: mean={all_mean.shape}, max={all_max.shape}, std={all_std.shape}")
    print(f"  Segment embeddings: {merged_seg_emb.shape}")
    if failed:
        print(f"  Failed: {len(failed)} tracks")


def cmd_similar(args):
    """Find similar tracks."""
    music_dir = Path(args.output)
    emb_data = load_embeddings(music_dir)
    disc_data = load_discovery_embeddings(music_dir)

    results = find_similar(emb_data, args.video_id, n=args.n, alpha=args.alpha,
                           discovery_data=disc_data)
    if not results:
        return

    # Find query title from either corpus
    if args.video_id in emb_data["ids"]:
        query_title = emb_data["titles"][emb_data["ids"].index(args.video_id)]
    elif disc_data and args.video_id in disc_data["ids"]:
        query_title = disc_data["titles"][disc_data["ids"].index(args.video_id)]
    else:
        query_title = args.video_id

    # Handle Unicode for Windows console
    safe_title = query_title.encode('ascii', 'replace').decode('ascii')
    print(f"Similar to: {safe_title}")
    print(f"  (alpha={args.alpha}: {args.alpha:.0%} semantic + {1-args.alpha:.0%} acoustic)\n")

    for i, r in enumerate(results, 1):
        src = f" [{r.get('source', 'library')}]" if r.get("source") == "discovery" else ""
        artist = f" by {r['artist']}" if r.get("artist") else ""
        safe_result = r['title'][:60].encode('ascii', 'replace').decode('ascii')
        safe_artist = artist.encode('ascii', 'replace').decode('ascii')
        print(f"  {i:2d}. [{r['score']:.3f}] {safe_result}{safe_artist}{src}")
        print(f"      CLAP={r['clap_score']:.3f}  VAE={r['vae_score']:.3f}  id={r['id']}")


def cmd_search(args):
    """Search by text description."""
    torch = _import_torch()
    music_dir = Path(args.output)
    emb_data = load_embeddings(music_dir)

    clap_model, clap_processor = load_clap(device=args.device)
    results = search_by_text(emb_data, clap_model, clap_processor,
                             args.query, n=args.n, device=args.device)

    print(f'Search: "{args.query}"\n')
    for i, r in enumerate(results, 1):
        safe_title = r['title'][:70].encode('ascii', 'replace').decode('ascii')
        print(f"  {i:2d}. [{r['score']:.3f}] {safe_title}")
        print(f"      id={r['id']}")


def cmd_info(args):
    """Show info about saved embeddings."""
    music_dir = Path(args.output)

    emb_path = music_dir / EMBEDDINGS_FILE
    if emb_path.exists():
        data = np.load(emb_path, allow_pickle=True)
        print(f"Library embeddings: {emb_path}")
        print(f"  Tracks:    {len(data['ids'])}")
        print(f"  VAE shape: {data['vae'].shape}")
        print(f"  CLAP shape:{data['clap'].shape}")
        print(f"  File size: {emb_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"No library embeddings at {emb_path}")

    disc_path = music_dir / DISCOVERY_EMBEDDINGS_FILE
    if disc_path.exists():
        data = np.load(disc_path, allow_pickle=True)
        print(f"\nDiscovery embeddings: {disc_path}")
        print(f"  Tracks:    {len(data['ids'])}")
        print(f"  VAE shape: {data['vae'].shape}")
        print(f"  CLAP shape:{data['clap'].shape}")
        print(f"  File size: {disc_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Show source breakdown
        source_ids = list(data.get("source_ids", []))
        unique_sources = len(set(source_ids))
        print(f"  Discovered from {unique_sources} seed tracks")
    else:
        print(f"\nNo discovery embeddings yet. Run 'python recommender.py crawl'.")

    preview_dir = music_dir / PREVIEW_DIR_NAME
    if preview_dir.exists():
        previews = list(preview_dir.iterdir())
        size_mb = sum(f.stat().st_size for f in previews) / 1024 / 1024
        print(f"\nPreview cache: {len(previews)} files, {size_mb:.1f} MB")


def cmd_crawl(args):
    """Snowball discovery: find related tracks, download previews, embed them."""
    import random as _random

    torch = _import_torch()
    music_dir = Path(args.output)
    device = args.device
    preview_dir = music_dir / PREVIEW_DIR_NAME

    # Load library embeddings (our seed)
    emb_data = load_embeddings(music_dir)
    library_ids = set(emb_data["ids"])
    print(f"Library: {len(library_ids)} embedded tracks")

    # Load existing discovery corpus
    disc_data = load_discovery_embeddings(music_dir)
    if disc_data:
        existing_disc_ids = set(disc_data["ids"])
        print(f"Existing discoveries: {len(existing_disc_ids)}")
    else:
        existing_disc_ids = set()
        disc_data = {
            "ids": [], "titles": [], "artists": [], "source_ids": [],
            "vae": np.zeros((0, 64), dtype=np.float32),
            "clap": np.zeros((0, 512), dtype=np.float32),
        }

    all_known_ids = library_ids | existing_disc_ids

    # Determine seed tracks for this round
    seed_ids = list(library_ids)
    if args.round > 1 and disc_data["ids"]:
        # Later rounds also use discoveries as seeds
        seed_ids += list(existing_disc_ids)

    if args.sample and args.sample < len(seed_ids):
        _random.shuffle(seed_ids)
        seed_ids = seed_ids[:args.sample]

    print(f"Using {len(seed_ids)} seed tracks for discovery")
    print(f"Max discoveries: {args.max_discover}")
    print(f"Fetching related tracks...\n")

    # Phase 1: Collect candidates from YouTube Music
    candidates = {}  # video_id -> {title, artist, source_id}
    for i, seed_id in enumerate(seed_ids):
        if len(candidates) >= args.max_discover * 2:  # fetch extra, some will fail
            break
        pct = (i + 1) / len(seed_ids) * 100
        print(f"\r  Querying seeds: {i+1}/{len(seed_ids)} ({pct:.0f}%) "
              f"— {len(candidates)} candidates found", end="", flush=True)

        related = get_related_tracks(seed_id, limit=args.per_track)
        for track in related:
            tid = track["id"]
            if tid not in all_known_ids and tid not in candidates:
                candidates[tid] = track

        # Rate-limit YouTube Music queries
        time.sleep(0.5)

    print(f"\n\n  Total unique candidates: {len(candidates)}")

    if not candidates:
        print("No new tracks discovered.")
        return

    # Cap to max
    candidate_list = list(candidates.values())[:args.max_discover]
    print(f"  Will process: {len(candidate_list)} tracks\n")

    # Phase 2: Load models
    vae = load_vae(device=device)
    clap_model, clap_processor = load_clap(device=device)
    print()

    # Phase 3: Download previews + embed
    new_ids = []
    new_titles = []
    new_artists = []
    new_source_ids = []
    new_vae = []
    new_clap = []
    dl_failed = 0
    emb_failed = 0

    for i, track in enumerate(candidate_list):
        vid_id = track["id"]
        title = track["title"]
        artist = track.get("artist", "")
        pct = (i + 1) / len(candidate_list) * 100
        print(f"[{i+1}/{len(candidate_list)}] ({pct:.0f}%) {artist} - {title}"[:75])

        # Download preview
        preview_path = download_preview(vid_id, preview_dir)
        if not preview_path:
            print(f"    download failed, skipping")
            dl_failed += 1
            continue

        # Embed
        v_emb = encode_vae(vae, preview_path, device=device)
        c_emb = encode_clap(clap_model, clap_processor, preview_path, device=device)

        if v_emb is not None and c_emb is not None:
            new_ids.append(vid_id)
            new_titles.append(f"{artist} - {title}" if artist else title)
            new_artists.append(artist)
            new_source_ids.append(track["source_id"])
            new_vae.append(v_emb)
            new_clap.append(c_emb)
        else:
            emb_failed += 1
            print(f"    embed failed, skipping")

        # Periodic CUDA cleanup + save
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

        # Rate-limit downloads
        time.sleep(1)

    if not new_ids:
        print("\nNo tracks successfully embedded.")
        return

    # Phase 4: Merge with existing discovery corpus
    merged_ids = list(disc_data["ids"]) + new_ids
    merged_titles = list(disc_data["titles"]) + new_titles
    merged_artists = list(disc_data["artists"]) + new_artists
    merged_source_ids = list(disc_data["source_ids"]) + new_source_ids

    if len(disc_data["ids"]) > 0:
        merged_vae = np.concatenate([disc_data["vae"], np.stack(new_vae)])
        merged_clap = np.concatenate([disc_data["clap"], np.stack(new_clap)])
    else:
        merged_vae = np.stack(new_vae)
        merged_clap = np.stack(new_clap)

    save_discovery_embeddings(music_dir, {
        "ids": merged_ids,
        "titles": merged_titles,
        "artists": merged_artists,
        "source_ids": merged_source_ids,
        "vae": merged_vae,
        "clap": merged_clap,
    })

    # Summary
    print(f"\n{'='*60}")
    print(f"CRAWL COMPLETE")
    print(f"{'='*60}")
    print(f"  New discoveries:     {len(new_ids)}")
    print(f"  Download failures:   {dl_failed}")
    print(f"  Embedding failures:  {emb_failed}")
    print(f"  Total in discovery:  {len(merged_ids)}")
    print(f"  Preview cache:       {preview_dir}")
    print(f"  Discovery file:      {music_dir / DISCOVERY_EMBEDDINGS_FILE}")
    print(f"\nRun 'python recommender.py similar <id>' to search both corpora.")


# ---------------------------------------------------------------------------
# Feature: Smart Playlist Generator
# ---------------------------------------------------------------------------
TAGS_FILE = "track_tags.json"


def generate_playlist(emb_data: dict, seed_id: str, length_minutes: int = 60,
                      drift: float = 0.3, avoid_recent: int = 5) -> list[dict]:
    """Generate a flowing playlist starting from a seed track.

    Args:
        emb_data: Embeddings data
        seed_id: Starting track video ID
        length_minutes: Target playlist length in minutes (assumes ~3.5 min/track)
        drift: How much the playlist can evolve (0=stay similar, 1=explore more)
        avoid_recent: Don't repeat tracks within this many selections

    Returns:
        List of track entries in playlist order
    """
    ids = emb_data["ids"]
    if seed_id not in ids:
        print(f"Seed track '{seed_id}' not found in embeddings.")
        return []

    # Estimate number of tracks (assume avg 3.5 min per track)
    num_tracks = max(1, int(length_minutes / 3.5))

    clap_all = emb_data["clap"]

    # Normalize all embeddings once
    norms = np.linalg.norm(clap_all, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    clap_normed = clap_all / norms

    playlist = []
    used_indices = set()
    current_idx = ids.index(seed_id)

    for i in range(num_tracks):
        playlist.append({
            "id": ids[current_idx],
            "title": emb_data["titles"][current_idx],
            "position": i + 1,
        })
        used_indices.add(current_idx)

        if i >= num_tracks - 1:
            break

        # Get current track's embedding
        current_emb = clap_normed[current_idx]

        # Calculate similarities to all tracks
        sims = clap_normed @ current_emb

        # Apply drift: blend current similarity with exploration
        # Higher drift = more randomness in selection
        if drift > 0:
            noise = np.random.random(len(sims)) * drift * 0.5
            sims = sims * (1 - drift * 0.3) + noise

        # Mask out used tracks (especially recent ones to avoid loops)
        recent_penalty = list(used_indices)[-avoid_recent:] if len(used_indices) > avoid_recent else list(used_indices)
        for idx in used_indices:
            if idx in recent_penalty:
                sims[idx] = -1  # Strong penalty for recent
            else:
                sims[idx] = sims[idx] * 0.5  # Soft penalty for older

        # Select next track from top candidates with some randomness
        top_k = min(10, len(ids) - len(used_indices))
        if top_k <= 0:
            break

        top_indices = np.argsort(-sims)[:top_k * 2]
        valid_indices = [idx for idx in top_indices if idx not in recent_penalty][:top_k]

        if not valid_indices:
            break

        # Weighted random selection from top candidates
        weights = np.array([sims[idx] for idx in valid_indices])
        weights = np.maximum(weights, 0.01)  # Ensure positive
        weights = weights / weights.sum()

        current_idx = np.random.choice(valid_indices, p=weights)

    return playlist


def save_playlist_m3u(playlist: list[dict], music_dir: Path, output_path: Path):
    """Save playlist as M3U file."""
    lines = ["#EXTM3U"]

    for track in playlist:
        vid_id = track["id"]
        title = track["title"]

        # Find the actual audio file
        audio_file = None
        for f in music_dir.iterdir():
            if f"[{vid_id}]" in f.name and f.suffix in AUDIO_EXTENSIONS:
                audio_file = f
                break

        if audio_file:
            lines.append(f"#EXTINF:-1,{title}")
            lines.append(str(audio_file))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved playlist to: {output_path}")


def cmd_playlist(args):
    """Generate a smart playlist."""
    music_dir = Path(args.output)
    emb_data = load_embeddings(music_dir)

    playlist = generate_playlist(
        emb_data,
        seed_id=args.seed,
        length_minutes=args.length,
        drift=args.drift,
    )

    if not playlist:
        return

    seed_title = emb_data["titles"][emb_data["ids"].index(args.seed)]
    safe_title = seed_title.encode('ascii', 'replace').decode('ascii')
    print(f"Generated playlist from: {safe_title[:60]}")
    print(f"  Length: {args.length} min (~{len(playlist)} tracks)")
    print(f"  Drift: {args.drift}\n")

    for track in playlist:
        safe_t = track['title'][:55].encode('ascii', 'replace').decode('ascii')
        print(f"  {track['position']:2d}. {safe_t}")

    if args.save:
        output_path = music_dir / f"playlist_{args.seed}_{args.length}min.m3u"
        save_playlist_m3u(playlist, music_dir, output_path)


# ---------------------------------------------------------------------------
# Feature: Auto-Tagging
# ---------------------------------------------------------------------------
DEFAULT_TAGS = [
    "energetic", "chill", "dark", "uplifting", "heavy", "melodic",
    "electronic", "rock", "hip-hop", "ambient", "aggressive", "dreamy",
    "fast", "slow", "vocals", "instrumental"
]


def tag_tracks(emb_data: dict, model, processor, labels: list[str],
               device="cuda", threshold: float = 0.01) -> dict:
    """Tag all tracks with labels using CLAP text-audio similarity.

    Returns:
        Dict mapping video_id -> {label: score, ...}
    """
    torch = _import_torch()

    # Encode all labels
    print(f"Encoding {len(labels)} labels...")
    label_embeddings = []
    for label in labels:
        emb = encode_text_clap(model, processor, label, device=device)
        label_embeddings.append(emb)
    label_matrix = np.stack(label_embeddings)  # [num_labels, 512]

    # Normalize
    label_norms = np.linalg.norm(label_matrix, axis=1, keepdims=True)
    label_matrix = label_matrix / np.where(label_norms == 0, 1, label_norms)

    clap_all = emb_data["clap"]
    clap_norms = np.linalg.norm(clap_all, axis=1, keepdims=True)
    clap_normed = clap_all / np.where(clap_norms == 0, 1, clap_norms)

    # Compute all similarities: [num_tracks, num_labels]
    print(f"Computing similarities for {len(emb_data['ids'])} tracks...")
    all_sims = clap_normed @ label_matrix.T

    # Build tags dict
    tags = {}
    for i, vid_id in enumerate(emb_data["ids"]):
        track_tags = {}
        for j, label in enumerate(labels):
            score = float(all_sims[i, j])
            if score >= threshold:
                track_tags[label] = round(score, 4)
        tags[vid_id] = track_tags

    return tags


def save_tags(music_dir: Path, tags: dict):
    """Save tags to JSON file."""
    import json
    path = music_dir / TAGS_FILE
    path.write_text(json.dumps(tags, indent=2))
    print(f"Saved tags to: {path}")


def load_tags(music_dir: Path) -> dict:
    """Load tags from JSON file."""
    import json
    path = music_dir / TAGS_FILE
    if path.exists():
        return json.loads(path.read_text())
    return {}


def cmd_tag(args):
    """Auto-tag tracks with mood/genre labels."""
    torch = _import_torch()
    music_dir = Path(args.output)
    emb_data = load_embeddings(music_dir)

    # Parse labels
    if args.labels:
        labels = [l.strip() for l in args.labels.split(",")]
    else:
        labels = DEFAULT_TAGS

    print(f"Tagging {len(emb_data['ids'])} tracks with {len(labels)} labels:")
    print(f"  {', '.join(labels)}\n")

    clap_model, clap_processor = load_clap(device=args.device)

    tags = tag_tracks(emb_data, clap_model, clap_processor, labels,
                      device=args.device, threshold=args.threshold)

    save_tags(music_dir, tags)

    # Show summary
    label_counts = {l: 0 for l in labels}
    for vid_id, track_tags in tags.items():
        for label in track_tags:
            if label in label_counts:
                label_counts[label] += 1

    print(f"\nTag distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(tags) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:12s} {count:4d} ({pct:5.1f}%) {bar}")


# ---------------------------------------------------------------------------
# Feature: Duplicate Detection
# ---------------------------------------------------------------------------
def find_duplicates(emb_data: dict, threshold: float = 0.95) -> list[tuple]:
    """Find duplicate or near-duplicate tracks.

    Returns:
        List of (id1, id2, similarity, title1, title2) tuples
    """
    ids = emb_data["ids"]
    titles = emb_data["titles"]
    clap_all = emb_data["clap"]

    # Normalize
    norms = np.linalg.norm(clap_all, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    clap_normed = clap_all / norms

    # Compute full similarity matrix
    print(f"Computing similarity matrix for {len(ids)} tracks...")
    sim_matrix = clap_normed @ clap_normed.T

    # Find pairs above threshold (only upper triangle to avoid duplicates)
    duplicates = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sim_matrix[i, j] >= threshold:
                duplicates.append((
                    ids[i], ids[j],
                    float(sim_matrix[i, j]),
                    titles[i], titles[j]
                ))

    # Sort by similarity (highest first)
    duplicates.sort(key=lambda x: -x[2])

    return duplicates


def cmd_duplicates(args):
    """Find duplicate or near-duplicate tracks."""
    music_dir = Path(args.output)
    emb_data = load_embeddings(music_dir)

    print(f"Scanning {len(emb_data['ids'])} tracks for duplicates (threshold: {args.threshold})...\n")

    duplicates = find_duplicates(emb_data, threshold=args.threshold)

    if not duplicates:
        print("No duplicates found!")
        return

    print(f"Found {len(duplicates)} potential duplicate pairs:\n")

    for i, (id1, id2, sim, title1, title2) in enumerate(duplicates[:args.limit], 1):
        safe_t1 = title1[:40].encode('ascii', 'replace').decode('ascii')
        safe_t2 = title2[:40].encode('ascii', 'replace').decode('ascii')
        print(f"{i:3d}. [{sim:.3f}] Similarity")
        print(f"     A: {safe_t1}")
        print(f"        id={id1}")
        print(f"     B: {safe_t2}")
        print(f"        id={id2}")
        print()

    if len(duplicates) > args.limit:
        print(f"  ... and {len(duplicates) - args.limit} more pairs")

    # Save to file if requested
    if args.save:
        import json
        output_path = music_dir / "duplicates.json"
        data = [{
            "id1": d[0], "id2": d[1], "similarity": d[2],
            "title1": d[3], "title2": d[4]
        } for d in duplicates]
        output_path.write_text(json.dumps(data, indent=2))
        print(f"\nSaved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Music recommendation engine")
    parser.add_argument("-o", "--output", default=str(DEFAULT_MUSIC_DIR),
                        help="Music directory")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (cuda/cpu)")

    sub = parser.add_subparsers(dest="command")

    # embed
    p_embed = sub.add_parser("embed", help="Batch-encode all audio files")
    p_embed.add_argument("--clap-only", action="store_true",
                         help="Use CLAP embeddings only (skip VAE even if available)")

    # embed-segments
    p_embed_seg = sub.add_parser("embed-segments",
                                  help="Create full-track segment embeddings (mean/max/std + segments)")
    p_embed_seg.add_argument("--segment-duration", type=float, default=SEGMENT_DURATION,
                              help=f"Segment duration in seconds (default: {SEGMENT_DURATION})")

    # similar
    p_sim = sub.add_parser("similar", help="Find similar tracks")
    p_sim.add_argument("video_id", help="Video ID of the query track")
    p_sim.add_argument("-n", type=int, default=20, help="Number of results")
    p_sim.add_argument("--alpha", type=float, default=0.5,
                       help="Blend weight: 0=pure acoustic, 1=pure semantic")

    # search
    p_search = sub.add_parser("search", help="Search by text description")
    p_search.add_argument("query", help="Text description to search for")
    p_search.add_argument("-n", type=int, default=20, help="Number of results")

    # info
    p_info = sub.add_parser("info", help="Show embedding info")

    # crawl
    p_crawl = sub.add_parser("crawl", help="Discover new tracks via YouTube Music")
    p_crawl.add_argument("--sample", type=int, default=None,
                         help="Random sample of seed tracks to query (default: all)")
    p_crawl.add_argument("--per-track", type=int, default=25,
                         help="Related tracks to fetch per seed (default: 25)")
    p_crawl.add_argument("--max-discover", type=int, default=500,
                         help="Max new tracks to discover (default: 500)")
    p_crawl.add_argument("--round", type=int, default=1,
                         help="Crawl round: 1=library seeds only, 2+=include discoveries")

    # playlist - Smart Playlist Generator
    p_playlist = sub.add_parser("playlist", help="Generate a smart playlist from a seed track")
    p_playlist.add_argument("--seed", required=True, help="Video ID of the seed track")
    p_playlist.add_argument("--length", type=int, default=60,
                            help="Target playlist length in minutes (default: 60)")
    p_playlist.add_argument("--drift", type=float, default=0.3,
                            help="How much playlist can evolve: 0=stay similar, 1=explore (default: 0.3)")
    p_playlist.add_argument("--save", action="store_true",
                            help="Save playlist as M3U file")

    # tag - Auto-Tagging
    p_tag = sub.add_parser("tag", help="Auto-tag tracks with mood/genre labels")
    p_tag.add_argument("--labels", type=str, default=None,
                       help="Comma-separated labels (default: energetic,chill,dark,uplifting,...)")
    p_tag.add_argument("--threshold", type=float, default=0.005,
                       help="Minimum score to assign tag (default: 0.005)")

    # duplicates - Duplicate Detection
    p_dup = sub.add_parser("duplicates", help="Find duplicate or near-duplicate tracks")
    p_dup.add_argument("--threshold", type=float, default=0.95,
                       help="Similarity threshold (default: 0.95)")
    p_dup.add_argument("--limit", type=int, default=50,
                       help="Max pairs to display (default: 50)")
    p_dup.add_argument("--save", action="store_true",
                       help="Save results to duplicates.json")

    args = parser.parse_args()

    if args.command == "embed":
        cmd_embed(args)
    elif args.command == "embed-segments":
        cmd_embed_segments(args)
    elif args.command == "similar":
        cmd_similar(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "crawl":
        cmd_crawl(args)
    elif args.command == "playlist":
        cmd_playlist(args)
    elif args.command == "tag":
        cmd_tag(args)
    elif args.command == "duplicates":
        cmd_duplicates(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
