"""Text-to-Speech stage – Coqui XTTS-v2 (multilingual).

Reads ``plan.json`` produced by :pyfile:`plan.py`, generates one narration
track per requested language and stores WAV files under
``projects/<project>/output/voice_<lang>.wav``.  A side-car ``tts_meta.json`` is
written so :pyfile:`render.py` can mux audio tracks into the final MP4.

This implementation uses the 🐸 **TTS** library with the open-weights
``tts_models/multilingual/multi-dataset/xtts_v2`` model
(https://huggingface.co/coqui/XTTS-v2).  The model is downloaded
automatically on first run and cached locally (≈400 MB).

Example CLI usage::

    python -m ai_doc_composer.cli tts-stage swiss_trip \
        --languages en,fr --speaker ./voices/my_voice.wav
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import wave
from pathlib import Path
from typing import List, Dict

import typer
import torch
import importlib
import numpy as _np

# Optional runtime deps – make them fail-friendly so other CLI commands work without full TTS stack.

try:
    import soundfile as _sf  # type: ignore
    _SF_AVAILABLE = True
except ModuleNotFoundError:
    _SF_AVAILABLE = False

# Optional heavy import (Coqui-TTS). We also guard this so non-TTS commands keep working.

try:
    from TTS.api import TTS  # type: ignore

    _TTS_AVAILABLE = True
except ModuleNotFoundError:
    _TTS_AVAILABLE = False

# Optional Google Gemini TTS dependency – only needed when provider="gemini"
try:
    from google import genai as _genai  # type: ignore
    from google.genai import types as _genai_types  # type: ignore
    _GENAI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    try:
        # Fallback to older package if new one not available
        import google.generativeai as _genai  # type: ignore
        _genai_types = None
        _GENAI_AVAILABLE = True
    except ModuleNotFoundError:
        _GENAI_AVAILABLE = False

# Optional VibeVoice TTS dependency – only needed when provider="vibevoice"
try:
    from ai_doc_composer.experimental.vibevoice.provider import VibeVoiceTTS  # type: ignore
    _VIBEVOICE_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _VIBEVOICE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------


DATA_ROOT = Path(__file__).resolve().parents[2] / "projects"

DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Default Gemini speech model (preview) – can be overridden via --model-id
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-preview-tts"

_tts_obj_cache: Dict[str, "TTS"] = {}

# Allow-list Coqui XTTS config classes – needed since PyTorch 2.6
xtts_globals = [
    "TTS.tts.configs.xtts_config.XttsConfig",
    "TTS.tts.models.xtts.XttsAudioConfig",
    "TTS.tts.models.xtts.Xtts",
    "TTS.tts.models.xtts.XttsArgs",
    "TTS.config.shared_configs.BaseDatasetConfig",
    "TTS.config.shared_configs.BaseAudioConfig",
    "TTS.config.shared_configs.BaseTrainingConfig",
    "TTS.vocoder.configs.hifigan_config.HifiganConfig",
    "TTS.tts.configs.gpt_sovits_config.GPTSovitsConfig",
    "TTS.encoder.configs.speaker_encoder_config.SpeakerEncoderConfig",
]

# Add safe globals for PyTorch serialization (if available)
try:
    import torch.serialization as _ts
    # Safely add globals with error handling
    for path in xtts_globals:
        try:
            module_name, attr = path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            _ts.add_safe_globals({getattr(module, attr)})
        except (ImportError, AttributeError) as e:
            # Some modules might not be available in all TTS versions
            print(f"Warning: Could not import {path} for safe globals: {e}")

    # Allowlist required globals for XTTS checkpoints (PyTorch 2.6 change)
    try:
        _ts.add_safe_globals({
            importlib.import_module("TTS.tts.configs.xtts_config").XttsConfig,
            importlib.import_module("TTS.tts.models.xtts").XttsAudioConfig,
        })
    except Exception:
        # Older torch versions or module path issues; ignore silently
        pass
except ImportError:
    # torch.serialization not available in this PyTorch version, skip
    print("Warning: torch.serialization not available in this PyTorch version - skipping safe globals")

def _ensure_tts(model_id: str, gpu: bool) -> "TTS":  # type: ignore
    if model_id in _tts_obj_cache:
        return _tts_obj_cache[model_id]
    if not _TTS_AVAILABLE:
        raise RuntimeError("python package 'TTS' not installed. Run: pip install TTS")
    typer.echo(f"[tts] Loading TTS model {model_id} … (first run may be slow)")
    obj = TTS(model_id, gpu=gpu)
    _tts_obj_cache[model_id] = obj
    return obj


def _ffprobe_duration(wav_path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return float(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    project: str,
    *,
    provider: str = "xtts",
    languages: str = "en",
    speaker: str | None = None,
    model_id: str = DEFAULT_MODEL,
    gpu: bool | None = None,
    clean: bool = False,
) -> None:
    """Generate narration audio for *project*.

    Parameters
    ----------
    project
        Project folder under ``projects/``.
    languages
        Comma-separated list of BCP-47 codes (e.g. ``en,fr``).
    speaker
        Optional reference WAV for voice cloning. Can be:
        • a single path (applies to all languages) or "default" for model voice
        • a comma-separated mapping like "en:path_en.wav,ru:path_ru.wav" (use per language)
    model_id
        HuggingFace model ID for XTTS (advanced) or model size for VibeVoice ("1.5B" or "7B").
    gpu
        Force GPU (`True` / `False`).  Default = auto-detect.
    provider
        "xtts" (default, Coqui XTTS), "gemini" (Google Gemini TTS API), or "vibevoice" (experimental).
    clean
        If True, remove any existing WAV files under the project's audio/ folder before synthesis.
    """

    plan_path = DATA_ROOT / project / "json" / "plan.json"
    if not plan_path.exists():
        typer.secho(f"[tts] plan.json not found at {plan_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    plan = json.loads(plan_path.read_text())
    if isinstance(plan.get("script"), list):
        base_script = plan["script"]
    else:
        raise RuntimeError("plan.json 'script' must be a list")

    translations = plan.get("translations", {})

    langs = [l.strip() for l in languages.split(",") if l.strip()]
    if not langs:
        typer.secho("[tts] No languages specified", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    output_dir = DATA_ROOT / project / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Dict] = {
        "project": project,
        "tracks": [],
    }

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Clean existing audio files if requested
    if clean:
        deleted = 0
        for p in audio_dir.glob("*.wav"):
            p.unlink(missing_ok=True)
            deleted += 1
        if deleted:
            typer.echo(f"[tts] --clean: removed {deleted} existing WAV file(s)")

    # -----------------------------
    # Parse speaker mapping
    # -----------------------------

    default_speaker_wav: str | None = None
    speaker_map: Dict[str, str] = {}

    if speaker and speaker.lower() != "default":
        # detect mapping pattern lang:path
        if ":" in speaker or "=" in speaker:
            pairs = [p.strip() for p in speaker.split(",") if p.strip()]
            for pair in pairs:
                if ":" in pair:
                    lang_key, path = pair.split(":", 1)
                else:
                    lang_key, path = pair.split("=", 1)
                speaker_map[lang_key.strip()] = path.strip()
        else:
            default_speaker_wav = speaker

    provider = provider.lower()

    # ------------------------------------------------------------------
    # Provider: Coqui XTTS (local)
    # ------------------------------------------------------------------

    if provider == "xtts":
        for lang in langs:
            script = translations.get(lang, base_script)

            typer.echo(f"[tts] Synthesising sentence-per-clip tracks ({lang}) via XTTS…")
            actual_gpu = gpu if gpu is not None else torch.cuda.is_available()
            tts_obj = _ensure_tts(model_id, gpu=actual_gpu)

            speaker_wav = speaker_map.get(lang, default_speaker_wav)

            if speaker_wav in (None, "default", ""):
                speaker_wav = None
                # XTTS requires a speaker for multi-speaker models
                # Use the first available speaker as default
                if hasattr(tts_obj.tts, 'speakers') and tts_obj.tts.speakers:
                    speaker_name = tts_obj.tts.speakers[0]
                else:
                    speaker_name = "Claribel Dervla"  # Common XTTS default speaker
            else:
                speaker_name = None

            track_info: List[Dict] = []

            for clip_id, sentence in zip(plan["ordered_clips"], script):
                audio_path = audio_dir / f"{clip_id}_{lang}.wav"

                if audio_path.exists():
                    typer.echo(f"  • {audio_path.name} already exists, skipping")
                    duration = _ffprobe_duration(audio_path)
                else:
                    out = tts_obj.tts(
                        sentence,
                        speaker_wav=speaker_wav,
                        speaker=speaker_name,
                        language=lang,
                    )  # type: ignore

                    if isinstance(out, tuple):
                        wav_arr, sr = out
                    else:
                        wav_arr = out
                        sr = 24000

                    if not _SF_AVAILABLE:
                        raise RuntimeError("python module 'soundfile' not installed – required for the tts-stage. Hint: 'poetry add soundfile'.")

                    _sf.write(audio_path, wav_arr, sr)
                    duration = len(wav_arr) / sr
                    typer.echo(f"  • {audio_path.name} ({duration:.1f}s)")

                track_info.append({
                    "id": clip_id,
                    "wav": str(audio_path.relative_to(DATA_ROOT)),
                    "duration": duration,
                })

            meta["tracks"].append({"lang": lang, "clips": track_info})

    # ------------------------------------------------------------------
    # Provider: Google Gemini TTS (cloud)
    # ------------------------------------------------------------------
    elif provider == "gemini":
        if not _GENAI_AVAILABLE:
            raise RuntimeError("python package 'google-genai' not installed. Hint: 'pip install google-genai'.")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Environment variable GEMINI_API_KEY not set – required for Gemini TTS provider.")

        # Initialize Gemini client
        if _genai_types:  # New API with actual TTS
            client = _genai.Client(api_key=api_key)
            tts_model = "gemini-2.5-flash-preview-tts"  # Fast model
            # Available voices for different languages
            voices = {
                'en': 'Kore',      # Firm, clear voice
                'es': 'Aoede',     # Breezy voice
                'fr': 'Charon',    # Informative voice
                'de': 'Fenrir',    # Excitable voice
                'it': 'Leda',      # Youthful voice
                'pt': 'Orus',      # Firm voice
                'default': 'Puck'  # Upbeat voice
            }
        else:  # Fallback to old API (placeholder)
            _genai.configure(api_key=api_key)
            client = None
            tts_model = None
            voices = {}
        
        for lang in langs:
            script = translations.get(lang, base_script)

            voice_name = voices.get(lang, voices.get('default', 'Puck')) if voices else 'Puck'

            if client and _genai_types:
                typer.echo(f"[tts] Synthesising sentence-per-clip tracks ({lang}) via Gemini TTS (voice: {voice_name})…")
            else:
                typer.echo(f"[tts] Synthesising via Gemini (placeholder - install 'google-genai' for actual TTS)…")

            track_info: List[Dict] = []

            # Rate limiting for Gemini API (3 requests per minute on free tier)
            last_api_call_time = 0
            api_call_delay = 21  # 21 seconds between calls ensures < 3 per minute
            api_call_count = 0

            for clip_id, sentence in zip(plan["ordered_clips"], script):
                audio_path = audio_dir / f"{clip_id}_{lang}.wav"

                if audio_path.exists():
                    typer.echo(f"  • {audio_path.name} already exists, skipping")
                    duration = _ffprobe_duration(audio_path)
                else:
                    sample_rate = 24000

                    if client and _genai_types:
                        try:
                            # Rate limiting: wait if needed to respect API quota
                            current_time = time.time()
                            if api_call_count > 0:
                                elapsed = current_time - last_api_call_time
                                if elapsed < api_call_delay:
                                    wait_time = api_call_delay - elapsed
                                    typer.echo(f"  ⏳ Rate limiting: waiting {wait_time:.1f}s before next API call...")
                                    time.sleep(wait_time)

                            # Generate actual TTS using Gemini API
                            typer.echo(f"  🎯 Generating audio for {clip_id} (API call #{api_call_count + 1})...")
                            response = client.models.generate_content(
                                model=tts_model,
                                contents=sentence,
                                config=_genai_types.GenerateContentConfig(
                                    response_modalities=["AUDIO"],
                                    speech_config=_genai_types.SpeechConfig(
                                        voice_config=_genai_types.VoiceConfig(
                                            prebuilt_voice_config=_genai_types.PrebuiltVoiceConfig(
                                                voice_name=voice_name,
                                            )
                                        )
                                    ),
                                )
                            )

                            # Extract audio data
                            audio_data = response.candidates[0].content.parts[0].inline_data.data

                            # Save as WAV file
                            with wave.open(str(audio_path), 'wb') as wf:
                                wf.setnchannels(1)  # Mono
                                wf.setsampwidth(2)  # 16-bit
                                wf.setframerate(sample_rate)
                                wf.writeframes(audio_data)

                            duration = _ffprobe_duration(audio_path)
                            typer.echo(f"  {audio_path.name} ({duration:.1f}s)")

                            # Update rate limiting counters
                            last_api_call_time = time.time()
                            api_call_count += 1

                        except Exception as e:
                            error_str = str(e)
                            typer.echo(f"  WARNING: Gemini TTS failed for {clip_id}: {error_str}")

                            # Check if it's a quota exhaustion error
                            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                                typer.secho(f"\nQUOTA EXHAUSTED: Daily limit reached for Gemini TTS API", fg=typer.colors.RED, bold=True)
                                typer.echo("Please wait until quota resets (usually after 24 hours) or upgrade to a paid plan.")
                                typer.echo("\nOptions:")
                                typer.echo("1. Wait for quota to reset tomorrow")
                                typer.echo("2. Use a different TTS provider (--provider xtts or --provider vibevoice)")
                                typer.echo("3. Upgrade to a paid Gemini API plan")
                                raise typer.Exit(1)

                            # For other errors, fallback to silence
                            duration = min(len(sentence.split()) * 0.4, 10.0)
                            silence_samples = int(sample_rate * duration)
                            silence_audio = _np.zeros(silence_samples, dtype=_np.int16)

                            if _SF_AVAILABLE:
                                _sf.write(str(audio_path), silence_audio, sample_rate)
                            else:
                                with wave.open(str(audio_path), 'wb') as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(sample_rate)
                                    wf.writeframes(silence_audio.tobytes())

                            typer.echo(f"  • {audio_path.name} ({duration:.1f}s, fallback silence)")
                    else:
                        # No new API available, generate silence placeholder
                        duration = min(len(sentence.split()) * 0.4, 10.0)
                        silence_samples = int(sample_rate * duration)
                        silence_audio = _np.zeros(silence_samples, dtype=_np.int16)

                        if _SF_AVAILABLE:
                            _sf.write(str(audio_path), silence_audio, sample_rate)
                        else:
                            with wave.open(str(audio_path), 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(sample_rate)
                                wf.writeframes(silence_audio.tobytes())

                        typer.echo(f"  • {audio_path.name} ({duration:.1f}s, placeholder)")

                track_info.append({
                    "id": clip_id,
                    "wav": str(audio_path.relative_to(DATA_ROOT)),
                    "duration": duration,
                })

            meta["tracks"].append({"lang": lang, "clips": track_info})

    # ------------------------------------------------------------------
    # Provider: VibeVoice (experimental local)
    # ------------------------------------------------------------------
    elif provider == "vibevoice":
        if not _VIBEVOICE_AVAILABLE:
            raise RuntimeError(
                "VibeVoice TTS provider not available. Make sure you have the required dependencies installed "
                "(transformers, accelerate) and the models downloaded. See VIBEVOICE.md for setup instructions."
            )

        # For VibeVoice, model_id can be "1.5B" or "7B", default to 1.5B
        vibe_model_size = "1.5B" if model_id == DEFAULT_MODEL else model_id
        if vibe_model_size not in ["1.5B", "7B"]:
            vibe_model_size = "1.5B"

        typer.echo(f"[tts] Initializing VibeVoice {vibe_model_size} model...")
        vibe_tts = VibeVoiceTTS(model_size=vibe_model_size)

        for lang in langs:
            if lang != "en":
                typer.echo(f"[tts] Warning: VibeVoice currently only supports English. Skipping {lang}")
                continue

            script = translations.get(lang, base_script)
            typer.echo(f"[tts] Synthesising long-form narration ({lang}) via VibeVoice...")

            track_info: List[Dict] = []

            for clip_id, sentence in zip(plan["ordered_clips"], script):
                audio_path = audio_dir / f"{clip_id}_{lang}.wav"

                if audio_path.exists():
                    typer.echo(f"  • {audio_path.name} already exists, skipping")
                    duration = _ffprobe_duration(audio_path)
                else:
                    # VibeVoice can handle long narration using multi-speaker format
                    audio_array, sample_rate = vibe_tts.synthesize_long_narration(sentence)

                    if not _SF_AVAILABLE:
                        raise RuntimeError("python module 'soundfile' not installed – required for the tts-stage. Hint: 'poetry add soundfile'.")

                    _sf.write(audio_path, audio_array, sample_rate)
                    duration = len(audio_array) / sample_rate
                    typer.echo(f"  • {audio_path.name} ({duration:.1f}s)")

                track_info.append({
                    "id": clip_id,
                    "wav": str(audio_path.relative_to(DATA_ROOT)),
                    "duration": duration,
                })

            meta["tracks"].append({"lang": lang, "clips": track_info})

    else:
        raise ValueError(f"Unknown TTS provider '{provider}'. Expected 'xtts', 'gemini', or 'vibevoice'.")

    # Write TTS metadata
    meta_path = DATA_ROOT / project / "json" / "tts_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    typer.secho(f"[tts] Meta written to {meta_path}", fg=typer.colors.GREEN)


# Typer CLI glue for standalone testing


if __name__ == "__main__":  # pragma: no cover
    import typer as _ty

    _app = _ty.Typer(add_completion=False)

    @_app.command()
    def main(
        project: str,
        languages: str = _ty.Option("en"),
        speaker: str | None = _ty.Option(None, help="Reference WAV for voice cloning (optional)"),
        model_id: str = _ty.Option(DEFAULT_MODEL),
        gpu: bool | None = _ty.Option(None, help="Force GPU on/off"),
        provider: str = _ty.Option("xtts", help="TTS provider: 'xtts', 'gemini', or 'vibevoice'"),
        clean: bool = _ty.Option(False, help="Remove existing WAV files before synthesis"),
    ) -> None:
        run(project, languages=languages, speaker=speaker, model_id=model_id, gpu=gpu, provider=provider, clean=clean)

    _app()