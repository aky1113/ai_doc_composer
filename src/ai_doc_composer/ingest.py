import json
import datetime as _dt
import subprocess
from pathlib import Path
from typing import List, Dict, Union, Optional
from io import BytesIO
import os
from dotenv import load_dotenv

import typer
from PIL import Image, UnidentifiedImageError

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from .context import ContextManager
from .rate_limiter import rate_limited_call

try:
    import google.generativeai as genai
except ImportError:
    genai = None

load_dotenv()

DATA_ROOT = Path(__file__).resolve().parents[2] / "projects"

# Load model & processor lazily (only once per Python process)
_processor: Union[Blip2Processor, 'BlipProcessor', None] = None  # type: ignore
_model: object | None = None


def _load_model(model_id_override: Optional[str] = None) -> None:
    global _processor, _model
    if _processor is None or _model is None:
        try:
            if model_id_override:
                primary = model_id_override
            else:
                primary = "Salesforce/blip2-flan-t5-base"  # baseline CPU model

            typer.echo(f"[ingest] Loading model {primary}… (first run may be slow)")
            model_id = primary
            _processor = Blip2Processor.from_pretrained(model_id)
            device_arg = "mps" if torch.backends.mps.is_available() else "cpu"
            _model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map=device_arg)
        except Exception as e:
            typer.echo(f"[ingest] Gated or unavailable model: {e}. Falling back to BLIP-1 base.")
            fallback = "Salesforce/blip-image-captioning-base"
            from transformers import BlipProcessor, BlipForConditionalGeneration
            _processor = BlipProcessor.from_pretrained(fallback)
            _model = BlipForConditionalGeneration.from_pretrained(fallback, device_map="cpu")


def _extract_first_frame(video_path: Path) -> Image.Image:
    """Return the first frame of a video as a PIL Image using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    return Image.open(BytesIO(proc.stdout)).convert("RGB")


def _caption_with_retry(video_file: Path, provider: str, prompt_with_project: str, frames: int = 1, max_retries: int = 3) -> List[str]:
    """Generate caption with retry logic for API failures."""
    import time
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            wait_time = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s, 4s, etc.
            typer.echo(f" retry {attempt}/{max_retries} (waiting {wait_time}s)…", nl=False)
            time.sleep(wait_time)
        
        try:
            if provider == "gemini":
                system_prompt = (
                    "Respond ONLY with a single-line JSON object with two keys: "
                    "description (<=100 chars, 2-4 short sentences) and location. "
                    "No markdown, no code fences, no explanation."
                )
                resp_text = _gemini_caption(video_file, f"{system_prompt}\n{prompt_with_project}")
                # try to isolate JSON part
                import re as _re
                json_match = _re.search(r"\{.*\}", resp_text, flags=_re.S)
                if json_match:
                    resp_text = json_match.group(0)
                import json as _json
                try:
                    parsed = _json.loads(resp_text)
                    description = parsed.get("description", "")
                    location_guess = parsed.get("location", "")
                    return [description, location_guess] if location_guess else [description]
                except _json.JSONDecodeError:
                    return [resp_text]  # fallback
            else:  # BLIP
                duration_sec = _ffprobe_duration(video_file) or 0
                captions_list: List[str] = []
                for i in range(frames):
                    if frames == 1:
                        img = _extract_first_frame(video_file)
                    else:
                        ts = (duration_sec * i) / (frames - 1)
                        ts = min(ts, max(duration_sec - 0.05, 0))
                        img = _safe_frame(video_file, ts)
                    c_raw = _caption_image(img, prompt=prompt_with_project)
                    # Remove prompt echo if model repeats it
                    c = c_raw[len(prompt_with_project):].strip() if prompt_with_project and c_raw.startswith(prompt_with_project) else c_raw
                    if c:
                        captions_list.append(c)
                
                # de-duplicate while preserving order
                seen = set()
                cleaned = []
                for c in captions_list:
                    if c not in seen:
                        cleaned.append(c)
                        seen.add(c)
                return cleaned
        
        except subprocess.CalledProcessError as e:
            # Don't retry frame extraction failures
            return ["(frame extraction failed)"]
        except Exception as e:
            if attempt == max_retries:
                return [f"(caption error: {e})"]
            # Continue to next retry
    
    # Should never reach here, but just in case
    return ["(unknown error)"]


def _caption_image(img: Image.Image, prompt: str = "") -> str:
    """Generate a caption for a single image using BLIP-2."""
    inputs = _processor(images=img, text=prompt, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = _model.generate(**inputs, max_new_tokens=30)
    caption = _processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return caption


def _ffprobe_duration(video_path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return float(out)
    except Exception:
        return None


def _extract_exif_timestamp(video_path: Path) -> Optional[str]:
    """Extract creation timestamp from video file metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format_tags=creation_time",
            "-of", "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        timestamp_str = result.stdout.strip()
        
        if timestamp_str and timestamp_str != "N/A":
            # Try to parse and normalize the timestamp
            try:
                # Handle various timestamp formats
                if timestamp_str.endswith('Z'):
                    dt = _dt.datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
                else:
                    dt = _dt.datetime.fromisoformat(timestamp_str)
                
                # Return ISO format with timezone
                return dt.isoformat()
            except ValueError:
                # Fallback: try parsing with strptime for common formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"]:
                    try:
                        dt = _dt.datetime.strptime(timestamp_str, fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue
        
        return None
    except (subprocess.CalledProcessError, ValueError, TypeError):
        return None


def _extract_frame_at(video_path: Path, ts: float) -> Image.Image:
    """Extract frame at specific timestamp seconds."""
    cmd = [
        "ffmpeg",
        "-ss",
        f"{ts}",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    return Image.open(BytesIO(proc.stdout)).convert("RGB")


def _safe_frame(video_path: Path, ts: float) -> Image.Image:
    """Try to grab a frame; back off a little if exact ts fails."""
    for backoff in (0.0, 0.25, 0.5):
        try_ts = max(ts - backoff, 0)
        try:
            return _extract_frame_at(video_path, try_ts)
        except (subprocess.CalledProcessError, UnidentifiedImageError, OSError):
            continue
    # fallback to first frame
    return _extract_first_frame(video_path)


def _gemini_caption(video_path: Path, prompt: str = "Describe this clip.") -> str:
    if genai is None:
        raise RuntimeError("google-generativeai not installed")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment or .env")
    genai.configure(api_key=api_key)

    video_bytes = video_path.read_bytes()
    if len(video_bytes) > 20 * 1024 * 1024:
        raise RuntimeError("Gemini inline upload limited to 20MB; clip too large")
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = rate_limited_call(
        "gemini-flash",
        model.generate_content,
        [{
            "inline_data": {
                "data": video_bytes,
                "mime_type": "video/mp4",
            }
        }, prompt]
    )
    text = response.text.strip()
    return text


def run(project: str, *, frames: int = 1, prompt: str = "", model_id: Optional[str] = None, provider: str = "blip", retry_failed: bool = False, max_retries: int = 3) -> None:
    """Vision captioning stage – BLIP-2.

    Scans projects/<project>/input, generates one caption per clip (first frame),
    stores JSON in projects/<project>/json/captions.json.
    """

    # Prefer preview files for processing, fallback to original input files
    project_preview = DATA_ROOT / project / "input_preview" 
    project_input = DATA_ROOT / project / "input"
    
    # Check which directory to use for processing
    if project_preview.exists() and list(project_preview.glob("*.mp4")) + list(project_preview.glob("*.MP4")):
        typer.echo(f"[ingest] Using preprocessed files from input_preview/")
        source_dir = project_preview
    elif project_input.exists():
        typer.echo(f"[ingest] Using original files from input/")
        source_dir = project_input
    else:
        typer.secho(f"Project '{project}' not found at {project_input}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    video_files = sorted(list(source_dir.glob("*.mp4")) + list(source_dir.glob("*.MP4")) + 
                          list(source_dir.glob("*.mov")) + list(source_dir.glob("*.MOV")))
    if not video_files:
        typer.secho("No .mp4 or .mov files found in input/", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Handle retry_failed mode  
    project_path = DATA_ROOT / project
    existing_records = {}
    output_path = project_path / "json" / "captions.json"
    
    if retry_failed and output_path.exists():
        try:
            existing_data = json.loads(output_path.read_text())
            for record in existing_data.get("clips", []):
                existing_records[record["clip_id"]] = record
            typer.echo(f"[ingest] Loaded {len(existing_records)} existing captions")
        except Exception as e:
            typer.echo(f"[ingest] Warning: Could not load existing captions: {e}")
    
    # Filter video files if retry_failed is enabled
    if retry_failed:
        failed_clips = []
        for vf in video_files:
            clip_id = vf.stem
            if clip_id in existing_records:
                record = existing_records[clip_id]
                captions = record.get("captions", [])
                # Check if any caption contains error markers
                has_failed = any("caption error:" in str(cap) or "frame extraction failed" in str(cap) 
                                for cap in captions)
                if has_failed:
                    failed_clips.append(vf)
            else:
                # No existing record = needs processing
                failed_clips.append(vf)
        
        video_files = failed_clips
        typer.echo(f"[ingest] Retry mode: Processing {len(video_files)} failed/missing clip(s)")
        
        if not video_files:
            typer.echo("No failed captions found. All clips already have successful captions.")
            return

    typer.echo(f"[ingest] Processing {len(video_files)} clip(s)…")

    records: List[Dict] = []
    if provider == "blip":
        _load_model(model_id)

    # --- Build prompt helper ---------------------------------------------------
    project_hint = f"The clip is from the '{project}' project."
    if prompt.strip():
        prompt_with_project = f"{prompt.strip()} {project_hint}"
    else:
        prompt_with_project = f"Describe the clip in details. Try to recognize the location, if possible.{project_hint}"

    for vf in video_files:
        typer.echo(f"  • {vf.name} → caption…", nl=False)
        
        # Use the retry-enabled caption generation
        caption_final = _caption_with_retry(vf, provider, prompt_with_project, frames, max_retries)
        duration_sec = _ffprobe_duration(vf)
        
        # Extract EXIF timestamp from original file if available
        original_file = project_input / vf.name.replace('.mp4', '.MOV').replace('.MP4', '.MOV')
        if not original_file.exists():
            # Try other extensions
            for ext in ['.mov', '.avi', '.mkv']:
                test_file = project_input / (vf.stem + ext)
                if test_file.exists():
                    original_file = test_file
                    break
        
        if original_file.exists() and original_file != vf:
            exif_timestamp = _extract_exif_timestamp(original_file)
        else:
            exif_timestamp = _extract_exif_timestamp(vf)

        records.append({
            "clip_id": vf.stem,
            "filename": vf.name,
            "captions": caption_final,
            "duration": duration_sec,
            "exif_timestamp": exif_timestamp,
        })
        
        # Show success message
        if any("caption error:" in str(cap) for cap in caption_final):
            typer.echo(" failed.")
        elif provider == "gemini":
            typer.echo(" done (Gemini).")
        else:
            typer.echo(" done (BLIP).")

    meta_dir = DATA_ROOT / project / "json"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "captions.json"
    
    # In retry mode, merge with existing records
    if retry_failed and existing_records:
        # Update existing records with new results
        for new_record in records:
            existing_records[new_record["clip_id"]] = new_record
        
        # Convert back to list format
        all_records = list(existing_records.values())
        typer.echo(f"[ingest] Merged {len(records)} updated records with {len(existing_records) - len(records)} existing records")
    else:
        all_records = records
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "project": project,
            "generated": _dt.datetime.utcnow().isoformat() + "Z",
            "clips": all_records,
        }, f, indent=2)
    
    # Update context manager with EXIF timestamps
    project_path = DATA_ROOT / project
    context_manager = ContextManager(project_path)
    context_manager.initialize_clips([record["filename"] for record in all_records])
    
    # Update EXIF timestamps in clip contexts
    for record in all_records:
        if record.get("exif_timestamp"):
            context_manager.update_clip_context(
                record["filename"],
                exif_timestamp=record["exif_timestamp"]
            )

    typer.secho(f"[ingest] Captions written to {out_path}", fg=typer.colors.GREEN)
    
    # Report EXIF timestamp extraction results
    exif_count = sum(1 for record in all_records if record.get("exif_timestamp"))
    if exif_count > 0:
        typer.secho(f"[ingest] Extracted EXIF timestamps from {exif_count}/{len(all_records)} clips", fg=typer.colors.BLUE)
    else:
        typer.secho(f"[ingest] No EXIF timestamps found in video files", fg=typer.colors.YELLOW)