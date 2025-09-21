import typer
from pathlib import Path
import json
import subprocess
from typing import Dict, List, Sequence

import numpy as _np

# soundfile is an optional dependency for non-render commands, but required here
try:
    import soundfile as _sf  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError("python module 'soundfile' not installed – required for the render-stage. Hint: 'poetry add soundfile'.") from e

# Path to project data root (repo_root/data)
DATA_ROOT = Path(__file__).resolve().parents[2] / "projects"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _concat_video(clips: Sequence[Path], tmp_out: Path) -> None:
    """Concatenate *clips* in order into *tmp_out* (video only)."""

    # Build a temporary list file for the concat demuxer
    list_path = tmp_out.with_suffix(".txt")
    with list_path.open("w") as fh:
        for p in clips:
            fh.write(f"file '{p.as_posix()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:v",
        "copy",
        "-an",  # drop existing audio tracks; we'll add narration later
        str(tmp_out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    list_path.unlink(missing_ok=True)


def _build_audio_track(lang: str, ordered_ids: List[str], clip_meta: Dict[str, float], tts_track: Dict[str, Dict], *, offset: float, out_path: Path, allow_overflow: bool = False) -> None:
    """Create a single WAV *out_path* for *lang* by concatenating per-clip WAVs
    with an *offset* of silence before each narration and optional tail padding
    so that every clip's total audio length matches the clip duration.
    """

    # Grab sample rate & channels from the first wav
    first_clip_id = ordered_ids[0]
    first_wav_rel = tts_track[first_clip_id]["wav"]
    first_wav_path = DATA_ROOT / first_wav_rel
    audio0, sr = _sf.read(first_wav_path)
    # Convert to mono to avoid inconsistent channel counts across clips
    if audio0.ndim == 2:
        audio0 = audio0.mean(axis=1)
    channels = 1  # we now enforce mono

    def _silence(seconds: float) -> _np.ndarray:
        if seconds <= 0:
            return _np.zeros(0, dtype=_np.float32)
        n = int(round(seconds * sr))
        return _np.zeros(n, dtype=_np.float32)

    parts: List[_np.ndarray] = []

    for idx, cid in enumerate(ordered_ids):
        wav_rel = tts_track[cid]["wav"]
        wav_path = DATA_ROOT / wav_rel
        audio, sr2 = _sf.read(wav_path)
        if sr2 != sr:
            raise RuntimeError(f"Sample rate mismatch in {wav_path} – expected {sr}, got {sr2}")

        # Convert to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        dur_audio = len(audio) / sr
        dur_clip = clip_meta[cid]
        is_last_clip = (idx == len(ordered_ids) - 1)

        # Pre-pad offset
        parts.append(_silence(offset))
        parts.append(audio)

        # Tail pad so total matches clip duration (best-effort)
        tail = dur_clip - offset - dur_audio
        if tail < -0.05:  # allow slight overshoot tolerance
            # Check if we allow overflow (not for last clip)
            if allow_overflow and not is_last_clip:
                # Allow overflow - don't truncate, just log it
                typer.echo(f"[render] Allowing {-tail:.2f}s overflow from {cid} ({lang}) into next clip")
                # Don't add tail silence since we're overflowing
            else:
                # Standard behavior: truncate to fit
                typer.echo(f"[render] WARNING: narration for {cid} ({lang}) overruns clip by {-tail:.2f}s – will truncate.")
                # trim audio to fit
                new_len = int(round((dur_clip - offset) * sr))
                parts[-1] = audio[:new_len]
        else:
            parts.append(_silence(max(0.0, tail)))

    full_track = _np.concatenate(parts, axis=0)
    _sf.write(out_path, full_track, sr)


def run(project: str, *, offset: float = 1.0, full_quality: bool = False) -> None:
    """Mux video & audio into *final.mp4*.

    Parameters
    ----------
    project
        Project folder under ``projects/``.
    offset
        Seconds of silence inserted before each clip's narration (default 1 s).
    full_quality
        Use original high-resolution files instead of preview files for final render.
    """

    project_path = DATA_ROOT / project
    json_dir = project_path / "json"

    # Load shared metadata
    plan = _load_json(json_dir / "plan.json")
    captions = _load_json(json_dir / "captions.json")

    # Load TTS metadata
    tts_meta_path = json_dir / "tts_meta.json"

    tts_meta = _load_json(tts_meta_path)

    ordered_ids: List[str] = plan.get("ordered_clips")
    if not ordered_ids:
        raise RuntimeError("plan.json missing 'ordered_clips'")

    # Check if overflow compensation is enabled
    allow_overflow = plan.get("allow_overflow", False)

    # Build mapping clip_id -> duration (seconds)
    clip_durations: Dict[str, float] = {c["clip_id"]: c["duration"] for c in captions["clips"]}

    # Map lang -> {clip_id: {...}}
    lang_tracks: Dict[str, Dict[str, Dict]] = {}
    for tr in tts_meta["tracks"]:
        lang = tr["lang"]
        lang_tracks[lang] = {c["id"]: c for c in tr["clips"]}

    # Validate completeness
    for lang, mapping in lang_tracks.items():
        missing = [cid for cid in ordered_ids if cid not in mapping]
        if missing:
            raise RuntimeError(f"tts_meta missing {len(missing)} narration clip(s) for {lang}: {missing}")

    # ------------------------------------------------------------------
    # 1. Concatenate video clips (video-only)
    # ------------------------------------------------------------------

    # Choose source directory based on quality mode
    if full_quality:
        typer.echo("[render] Using original high-resolution files for full quality render")
        input_dir = project_path / "input"
    else:
        # Prefer preview files if available, fallback to original
        preview_dir = project_path / "input_preview"
        if preview_dir.exists() and list(preview_dir.glob("*.mp4")) + list(preview_dir.glob("*.MP4")):
            typer.echo("[render] Using preprocessed files for web-friendly render")
            input_dir = preview_dir
        else:
            typer.echo("[render] Using original files (no preview files available)")
            input_dir = project_path / "input"

    clip_paths = []
    for cid in ordered_ids:
        possible_paths = [
            input_dir / f"{cid}.mp4",
            input_dir / f"{cid}.MP4", 
            input_dir / f"{cid}.mov",
            input_dir / f"{cid}.MOV"
        ]
        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = path
                break
        if found_path:
            clip_paths.append(found_path)
        else:
            raise FileNotFoundError(f"No video file found for clip {cid} (checked: {[p.name for p in possible_paths]})")
    
    if not all(p.exists() for p in clip_paths):
        not_found = [p for p in clip_paths if not p.exists()]
        raise FileNotFoundError(f"Some input clips are missing: {not_found}")

    # Use output directory
    output_dir = project_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_video = output_dir / "_video_noaudio.mp4"
    typer.echo("[render] Concatenating video clips…")
    _concat_video(clip_paths, tmp_video)

    # ------------------------------------------------------------------
    # 2. Build language-specific audio track WAVs
    # ------------------------------------------------------------------

    narration_wavs: List[Path] = []
    langs_sorted = sorted(lang_tracks.keys())  # deterministic order

    for lang in langs_sorted:
        typer.echo(f"[render] Building combined audio track ({lang})…")
        out_wav = output_dir / f"narration_{lang}.wav"
        _build_audio_track(lang, ordered_ids, clip_durations, lang_tracks[lang], offset=offset, out_path=out_wav, allow_overflow=allow_overflow)
        narration_wavs.append(out_wav)

    # ------------------------------------------------------------------
    # 3. Mux video + audio into <project>.mp4
    # ------------------------------------------------------------------

    # Different filename for full quality version
    if full_quality:
        final_mp4 = output_dir / f"{project}_full_quality.mp4"
    else:
        final_mp4 = output_dir / f"{project}.mp4"

    cmd: List[str] = ["ffmpeg", "-y", "-i", str(tmp_video)]

    for wav in narration_wavs:
        cmd.extend(["-i", str(wav)])

    # Mapping: video first, then one audio per language
    cmd.extend(["-map", "0:v:0"])

    for idx, lang in enumerate(langs_sorted, start=1):
        # Input index = idx (since 0 is video)
        cmd.extend(["-map", f"{idx}:a:0", f"-metadata:s:a:{idx-1}", f"language={lang}"])

    cmd.extend([
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(final_mp4),
    ])

    typer.echo("[render] Muxing video & audio…")
    subprocess.run(cmd, check=True, capture_output=True)

    typer.secho(f"[render] Final video written to {final_mp4}", fg=typer.colors.GREEN)

    # Optional cleanup
    tmp_video.unlink(missing_ok=True)

# CLI helper for standalone testing


if __name__ == "__main__":  # pragma: no cover
    typer.run(run)