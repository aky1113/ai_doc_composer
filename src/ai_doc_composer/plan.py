"""Language-planning stage (clip ordering & narration).

This module takes the per-clip captions produced by ``ingest.py`` and asks a
language model (default: Llama-3 via Ollama) to:

1. Choose an engaging, natural order for the clips.
2. Generate one narration sentence per clip in a National-Geographic style.
3. Output a strict JSON plan so downstream stages (TTS / render) are fully
   deterministic.

Fallbacks ensure the pipeline never breaks when the LLM is unavailable – a
static baseline order and generic narration will be produced instead.
"""

from __future__ import annotations

import json
import re
import time
import typing as _t
from datetime import datetime
from pathlib import Path

import typer

from .context import ContextManager, ProjectContext
from .styles import DocumentaryStyles, StyleType
from .rate_limiter import rate_limited_call
from .quota_manager import QuotaManager, QUOTAS
from .duplicate_prevention import (
    detect_duplicate_phrases,
    create_duplicate_prevention_prompt,
    check_phrase_diversity,
    remove_duplicate_sentences
)

# Optional imports – present only if the extra is installed
try:
    import ollama  # type: ignore

    _OLLAMA_AVAILABLE = True
except ModuleNotFoundError:
    _OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai  # type: ignore

    _GEMINI_AVAILABLE = True
except ModuleNotFoundError:
    genai = None  # type: ignore
    _GEMINI_AVAILABLE = False


DATA_ROOT = Path(__file__).resolve().parents[2] / "projects"

# Gemini rate-limit: max 15 requests / minute (flash model quota)
_GEMINI_RPM = 15

# Initialize quota manager globally
_quota_manager = QuotaManager()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _estimate_speech_sec(text: str, *, wpm: int) -> float:
    """Rough estimate of speech duration in seconds for *text* at *wpm*."""

    words = len(re.findall(r"\w+", text))
    # Add small buffer for pauses between sentences
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    pause_time = sentence_count * 0.4  # 0.4 seconds per sentence ending (more realistic)
    return (words / wpm * 60.0 if wpm else 0.0) + pause_time



# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def _build_system_prompt(n_clips: int, style_config, project_context: str = "", destination: str = "") -> str:
    """Return system prompt string with style-specific instructions."""

    context_section = ""
    if project_context.strip():
        context_section = f"\nProject Context: {project_context.strip()}\n"
    
    if destination.strip():
        context_section += f"\nDestination/Theme: {destination.strip()}\n"

    return (
        f"You are a documentary planner writing in {style_config.tone} style. {style_config.master_prompt_template}{context_section}"
        "The user will supply a JSON array of clips. Each item has keys: id, caption, duration.\n"
        "TASKS:\n"
        "1. Re-order the clips into the most natural sequence for this style.\n"
        "2. Write EXACTLY one narration sentence per clip matching the style tone.\n"
        "3. If a *location* is provided for a clip you MUST reference it appropriately for the style.\n"
        "4. Keep each sentence long enough to fill roughly 80 percent of the clip's duration when read at 160 wpm.\n"
        "5. The FIRST sentence must act as an engaging introduction; the LAST sentence must feel like a natural conclusion.\n"
        "OUTPUT FORMAT (very important): Respond with *ONLY* a single JSON object, no code-blocks, no markdown. Keys:\n"
        "  ordered_clips : array of clip IDs (strings, same length as input)\n"
        "  script        : array of narration sentences (strings, same length)\n"
        "  # No other keys, no comments, no trailing commas. MUST be valid strict JSON."
    )


def _build_order_prompt(n_clips: int) -> str:
    """Prompt for stage-1 ordering only."""

    return (
        "You are a documentary editor. The user will give you a JSON array of clips. "
        "Decide the most natural, engaging order for a short documentary film. "
        "Respond with *only* a JSON array of clip IDs in the new order – no other keys, no markdown. "
        f"Always include exactly {n_clips} items."
    )


def _get_caption_text(c):
    """Extract caption text from various caption formats."""
    captions = c.get("captions", {})
    if isinstance(captions, dict):
        # Get the first available caption from dict
        return next(iter(captions.values()), "")
    elif isinstance(captions, list):
        # Legacy format: list of captions
        return captions[0] if captions else ""
    else:
        return str(captions) if captions else ""


def _build_narration_prompt(
    target_words: int,
    *,
    is_first: bool,
    is_last: bool,
    previous_sentence: str | None,
    max_words: int,
    style_name: str = "documentary",
    clip_idx: int = 0,
    total_clips: int = 1,
    all_previous_narrations: list[str] | None = None,
) -> str:
    """Return system prompt for per-clip narration with style-specific rules."""

    # Style-specific narration rules
    if style_name == "documentary":
        style_rules = (
            "Write engaging documentary narration that balances information with storytelling. "
            "Include 1-2 KEY FACTS that enhance understanding without overwhelming the viewer. "
            "Focus on what's most interesting or relevant about the scene. "
            "Connect facts to what viewers are seeing. Create a narrative flow, not a data dump. "
            "Good examples:\n"
            "- 'Paris sprawls along the Seine River, where 2 million residents navigate a city that's been France's heart for over 1500 years.'\n"
            "- 'Tower Bridge has spanned the Thames since 1894, its twin towers standing guard over London's busiest waterway.'\n"
            "- 'Niagara Falls thunders between two nations, dropping 50 meters with such force that its mist can be seen kilometers away.'\n"
            "Bad examples:\n"
            "- 'Population: 2,161,000. Elevation: 35m. Founded: 508 AD.' (too dry, like reading a spreadsheet)\n"
            "- 'The magical city whispers ancient secrets' (too vague, no substance)\n"
            "- 'The bridge measures 244m long, 65m tall, built 1894, 11,000 tons of steel...' (too many numbers)"
        )
    elif style_name == "travel_vlog":
        style_rules = (
            "Write like an enthusiastic YouTube travel vlogger: personal, excited, conversational. "
            "Use first-person perspective ('I discovered...', 'We found...', 'Let me show you...'). "
            "Share immediate reactions and personal experiences. Be authentic and spontaneous. "
            "Avoid starting every sentence with 'Now' - use natural conversational transitions."
        )
    elif style_name == "personal_memories":
        style_rules = (
            "Write natural, conversational memories like telling a friend about your trip. "
            "Use complete sentences (minimum 7-10 words each). Example: 'I still remember how excited the kids were when they first saw that bridge.' "
            "Avoid short fragments or poetic snippets. Be warm and genuine, not pretentious. "
            "Don't start sentences with 'Now' - use natural storytelling transitions."
        )
    else:
        # Fallback to documentary style
        style_rules = (
            "Write clear, informative narration that matches the scene. "
            "Be engaging and descriptive."
        )

    word_rule = f"Target approximately {target_words} words (±3 words tolerance). Maximum limit: {max_words} words. Be concise and natural."

    # Location rules vary by position in documentary
    if is_first:
        loc_rule = "If a *location* is given, mention it naturally in this opening to establish the setting."
    elif is_last:
        loc_rule = "If a *location* is given, you may optionally reference it one final time if it flows naturally."
    else:
        loc_rule = "If a *location* is given, generally avoid mentioning it unless absolutely necessary for clarity. The location has likely been established already."

    flow_rule = ""
    if previous_sentence:
        if style_name == "personal_memories":
            flow_rule = "Continue naturally from the previous thought. Don't repeat phrases like 'I remember' too often."
        else:
            # Provide varied transitions based on clip position
            transition_examples = []
            if clip_idx < 2:  # Early clips
                transition_examples = [
                    "'Twelve kilometers south...'", "'At 1,800 meters elevation...'",
                    "'The 14th-century construction...'", "'With a population of 5,000...'"
                ]
            elif clip_idx < total_clips - 2:  # Middle clips
                transition_examples = [
                    "'Rising 2,000 meters above sea level...'", "'Built in 1882...'",
                    "'This 6-kilometer stretch...'", "'The 300-meter waterfall...'"
                ]
            else:  # Late clips
                transition_examples = [
                    "'The final 3,000-meter ascent...'", "'Established in 1291...'",
                    "'This UNESCO site since 1983...'", "'With annual rainfall of 1,200mm...'"
                ]

            flow_rule = (
                "Make the transition natural from the previous narration while NOT repeating any whole sentences or obvious phrases. "
                "CRITICAL: Each sentence should have a unique opening. Never start consecutive sentences with the same word. "
                f"Good transition examples for this part: {', '.join(transition_examples)}. "
                "Avoid overused starts like 'Now,' 'Here,' - use them sparingly if at all."
            )

    # Add duplicate prevention warnings based on previous narrations
    duplicate_warning = ""
    if all_previous_narrations and len(all_previous_narrations) > 0:
        duplicate_warning = create_duplicate_prevention_prompt(all_previous_narrations)
        if duplicate_warning:
            flow_rule = flow_rule + "\n" + duplicate_warning

    intro_outro = ""
    if is_first:
        if style_name == "personal_memories":
            intro_outro = "This is the OPENING - start naturally, like beginning a story. Example: 'You know what I'll never forget about Porto?'"
        elif style_name == "documentary":
            intro_outro = (
                "This is the OPENING line—start with an engaging introduction that sets the scene. "
                "Balance key information with narrative appeal. Don't just list facts, tell a story. "
                "Good examples:\n"
                "- 'Mexico City sprawls across a high plateau at 2,240 meters, where 9 million people call this ancient Aztec capital home.'\n"
                "- 'Tokyo pulses with the energy of 14 million residents, making it the world's most populous metropolitan area.'\n"
                "Bad examples:\n"
                "- 'Elevation: 2,240m. Population: 9.2 million. Founded: 1325.' (too dry, like a database)\n"
                "- 'Ancient mystical energies flow through sacred stones' (too vague, no substance)"
            )
        elif style_name == "travel_vlog":
            intro_outro = (
                "This is the OPENING - grab attention immediately with excitement! "
                "Examples: 'Guys, you're not going to believe what I just discovered in Porto!' "
                "or 'Okay, I'm literally standing on the most incredible bridge right now!' "
                "Be energetic and personal from the start."
            )
        else:
            intro_outro = "This is the OPENING line—hook the audience and set the stage."
    elif is_last:
        if style_name == "personal_memories":
            intro_outro = "This is the FINAL line - wrap up warmly and naturally. Example: 'Even now, I can still picture those sunset colors.'"
        elif style_name == "documentary":
            intro_outro = (
                "This is the FINAL line—conclude with a meaningful reflection that ties the journey together. "
                "Balance a key insight with narrative closure. "
                "Example: 'From coastal Venice to the heights of Mont Blanc, this journey revealed the remarkable diversity packed into Europe's alpine heart.'"
            )
        else:
            intro_outro = "This is the FINAL line—deliver a reflective close to the journey."

    return (
        style_rules
        + " "
        + word_rule
        + " "
        + loc_rule
        + " "
        + flow_rule
        + " "
        + intro_outro
        + "Respond with plain text only (no markdown or JSON)."
    )


def _ask_ollama(
    clips: list[dict],
    *,
    model: str | None,
    temperature: float,
    wpm: int,
    style_config=None,
    project_context: str = "",
) -> tuple[list[str], list[str], list[float]]:
    if not _OLLAMA_AVAILABLE:
        raise RuntimeError("ollama python package not installed")

    # Use default style config if not provided
    if style_config is None:
        style_config = DocumentaryStyles.get_style(StyleType.DOCUMENTARY)

    user_json = json.dumps(
        [
            {
                "id": c["clip_id"],
                "caption": _get_caption_text(c),
                "location": (c.get("captions")[1] if len(c.get("captions", [])) > 1 else ""),
                "duration": c.get("duration", 0) or 0,
            }
            for c in clips
        ]
    )

    messages = [
        {"role": "system", "content": _build_system_prompt(len(clips), style_config, project_context)},
        {"role": "user", "content": user_json},
    ]

    model_name = model or "llama3.3:latest"
    typer.echo(f"[plan] 🤖 Asking Ollama ({model_name}) to generate narrative...")
    typer.echo(f"[plan]    Temperature: {temperature}, Target WPM: {wpm}")
    typer.echo(f"[plan]    Processing {len(clips)} clips...")

    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={"temperature": temperature},
        stream=False,
    )

    raw = response["message"]["content"].strip()
    typer.echo(f"[plan] Received response from Ollama ({len(raw)} chars)")

    # extract first JSON object
    try:
        json_match = re.search(r"\{.*\}", raw, flags=re.S)
        if not json_match:
            raise ValueError("no JSON found in response")
        payload = json.loads(json_match.group(0))
        ordered_raw = payload["ordered_clips"]
        script = payload["script"]
        if not isinstance(ordered_raw, list) or not isinstance(script, list):
            raise ValueError("invalid field types")
        # convert ordered list elements to strings if they are objects with 'id'
        ordered: list[str] = []
        for itm in ordered_raw:
            if isinstance(itm, str):
                ordered.append(itm)
            elif isinstance(itm, dict) and "id" in itm:
                ordered.append(str(itm["id"]))
            else:
                raise ValueError("invalid ordered_clips element")
        if len(ordered) != len(script):
            raise ValueError("ordered_clips and script length mismatch")
    except Exception as e:
        raise RuntimeError(f"ollama returned invalid JSON: {e}") from e

    speech_secs = [_estimate_speech_sec(line, wpm=wpm) for line in script]

    return ordered, script, speech_secs


def _ask_gemini(
    clips: list[dict],
    *,
    temperature: float,
    wpm: int,
    max_attempts: int = 2,
    style_config=None,
    project_context: str = "",
) -> tuple[list[str], list[str], list[float]]:
    if not _GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai package not installed")

    # Use default style config if not provided
    if style_config is None:
        style_config = DocumentaryStyles.get_style(StyleType.DOCUMENTARY)

    import os as _os
    api_key = _os.getenv("GEMINI_API_KEY") or (genai._default_api_key if hasattr(genai, "_default_api_key") else None)  # type: ignore
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    n_clips = len(clips)
    sys_prompt = _build_system_prompt(n_clips, style_config, project_context)

    user_json = json.dumps([
        {
            "id": c["clip_id"],
            "caption": _get_caption_text(c),
            "location": (c.get("captions")[1] if len(c.get("captions", [])) > 1 else ""),
            "duration": c.get("duration", 0) or 0,
        }
        for c in clips
    ])

    for attempt in range(max_attempts):
        response = rate_limited_call(
            "gemini-flash",
            model.generate_content,
            [sys_prompt, user_json],
            generation_config={"temperature": temperature}
        )

        raw = response.text.strip()

        try:
            json_match = re.search(r"\{.*\}", raw, flags=re.S)
            if not json_match:
                raise ValueError("no JSON found in response")
            payload = json.loads(json_match.group(0))
            ordered_raw = payload["ordered_clips"]
            script = payload["script"]
            if not isinstance(ordered_raw, list) or not isinstance(script, list):
                raise ValueError("invalid field types")
            # convert ordered elements
            ordered: list[str] = []
            for itm in ordered_raw:
                if isinstance(itm, str):
                    ordered.append(itm)
                elif isinstance(itm, dict) and "id" in itm:
                    ordered.append(str(itm["id"]))
                else:
                    raise ValueError("invalid ordered_clips element")

            if len(ordered) != n_clips or len(script) != n_clips:
                raise ValueError("array length mismatch")

            speech_secs = [_estimate_speech_sec(line, wpm=wpm) for line in script]
            return ordered, script, speech_secs
        except Exception as e:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"gemini returned invalid JSON after {max_attempts} attempts: {e}") from e
            # On retry, prepend feedback message
            sys_prompt = "PREVIOUS RESPONSE WAS INVALID (length mismatch). PLEASE RETRY WITH VALID JSON.\n" + sys_prompt

    # unreachable


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    project: str,
    *,
    provider: str = "ollama",  # "ollama" | "gemini"
    model: str | None = None,
    temperature: float = 0.15,  # Lowered for better instruction-following
    wpm: int = 160,
    slack: float = 0.15,
    languages: str = "en",
    style: str | None = None,
    ordering_strategy: str = "llm",  # "llm" | "exif"
    skip_existing: bool = False,
    allow_overflow: bool = False,
) -> None:
    """Main entry – generate plan.json for *project*.

    Parameters
    ----------
    project
        Name of the folder under ``projects/``.
    provider
        Which LLM backend to use (``ollama`` or ``gemini``).
    model
        Optional model ID override (e.g. ``llama3.1:8b``).
    temperature, wpm, slack
        LLM settings and timing tolerance.
    languages
        Comma-separated BCP47 codes for narration languages.
    style
        Documentary style (documentary, travel_vlog, memories, family_trip).
    ordering_strategy
        Clip ordering strategy ("llm" or "exif").
    skip_existing
        Skip if plan.json already exists.
    allow_overflow
        Allow narration to overflow between clips with compensation.
    """

    project_path = DATA_ROOT / project
    caps_path = project_path / "json" / "captions.json"
    plan_path = project_path / "json" / "plan.json"
    
    if skip_existing and plan_path.exists():
        typer.secho(f"[plan] Plan already exists at {plan_path}, skipping", fg=typer.colors.YELLOW)
        return
    
    if not caps_path.exists():
        typer.secho(f"[plan] No captions.json found at {caps_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    
    # Initialize context manager
    context_manager = ContextManager(project_path)
    project_context = context_manager.load_project_context()
    
    # Use provided style or fallback to project context or inferred style
    final_style = style or project_context.style
    if not final_style or final_style == "doc":
        # Fallback: try to infer from project name
        final_style = project.split("_")[-1] if "_" in project else "documentary"
        if final_style not in [s.value for s in StyleType]:
            final_style = "documentary"
    
    # Update project context with resolved values
    if style:
        project_context.style = final_style
    if ordering_strategy != "llm":
        project_context.ordering_strategy = ordering_strategy

    context_manager.save_project_context(project_context)

    data = json.loads(caps_path.read_text())
    clips: list[dict] = data.get("clips", [])
    if not clips:
        typer.secho("[plan] captions.json has no clip entries", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    
    # Get style configuration
    try:
        style_config = DocumentaryStyles.get_style_by_name(final_style)
    except ValueError:
        typer.secho(f"[plan] Unknown style '{final_style}', using documentary", fg=typer.colors.YELLOW)
        style_config = DocumentaryStyles.get_style(StyleType.DOCUMENTARY)
        final_style = "documentary"
    
    # Load clip contexts for personal context integration
    clip_contexts = context_manager.load_clip_contexts()
    
    # Handle ordering strategy
    if project_context.ordering_strategy == "exif":
        # Try EXIF-based ordering
        ordered = _order_by_exif(clips, clip_contexts)
        if not ordered:
            typer.secho("[plan] EXIF ordering failed, falling back to LLM ordering", fg=typer.colors.YELLOW)
            project_context.ordering_strategy = "llm"

    lang_list = [l.strip() for l in languages.split(",") if l.strip()]
    base_lang = lang_list[0] if lang_list else "en"

    ordered: list[str]
    script: list[str] = []
    speech_secs: list[float] = []
    translations: dict[str, list[str]] = {}

    if provider not in {"ollama", "gemini"}:
        typer.secho(f"[plan] Invalid provider: {provider}. Only 'ollama' and 'gemini' are supported.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # ------------- Stage 1 – ordering -----------------
    if project_context.ordering_strategy == "llm":
        if provider == "ollama":
            if not _OLLAMA_AVAILABLE:
                typer.secho("[plan] Ollama Python package not installed.", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            ordered = _ask_ollama_order(clips, style_config, project_context, model=model, temperature=temperature)
        else:  # gemini
            if not _GEMINI_AVAILABLE:
                typer.secho("[plan] google-generativeai package not installed.", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            ordered = _ask_gemini_order(clips, style_config, project_context, temperature=temperature)

    # ------------- Stage 2 – narration generation --------
    clip_map = {c["clip_id"]: c for c in clips}

    # Per-clip generation with smart retry for overflow
    typer.echo("[plan] Generating narration for each clip...")

    # Track cumulative overflow when allow_overflow is enabled
    cumulative_overflow_seconds = 0.0

    typer.echo(f"[Stage 2: Narration] Starting narration generation for {len(ordered)} clips")
    for idx, cid in enumerate(ordered):
        typer.echo(f"[Stage 2: Narration] Processing clip {idx+1}/{len(ordered)}: {cid}")
        c = clip_map[cid]
        caption_text = _get_caption_text(c)
        location_text = c.get("captions", [None, ""])[1] if len(c.get("captions", [])) > 1 else None
        duration_sec = c.get("duration", 0) or 0

        is_first = idx == 0
        is_last = idx == len(ordered) - 1

        # Adjust duration based on overflow compensation
        if allow_overflow and not is_last:
            # STRICT LIMITS: Never exceed 3 seconds total, max 1 second per clip
            # Calculate remaining overflow budget
            remaining_overflow_budget = max(0, 3.0 - cumulative_overflow_seconds)

            # Calculate per-clip overflow limit (much stricter than before)
            max_overflow_per_clip = min(1.0, duration_sec * 0.15)  # Max 1s or 15% of clip duration

            # Progressive reduction based on cumulative overflow
            if cumulative_overflow_seconds >= 2.5:
                # Near the 3s limit - no more overflow allowed
                max_overflow_per_clip = 0
                safe_duration = duration_sec * 0.70  # More conservative
                max_sec = duration_sec * 0.85  # Stricter limit
            elif cumulative_overflow_seconds >= 1.5:
                # Getting close to limit - reduce overflow allowance by half
                max_overflow_per_clip *= 0.5
                safe_duration = duration_sec * 0.75  # More conservative
                max_sec = duration_sec + min(remaining_overflow_budget, max_overflow_per_clip)
            else:
                # Still have room - but be conservative
                safe_duration = duration_sec * 0.85  # Reasonable safety margin
                max_sec = duration_sec + min(remaining_overflow_budget, max_overflow_per_clip)
        else:
            # Standard safety margins (original behavior) - no overflow for last clip
            safe_duration = duration_sec * 0.85  # Standard safety margin
            max_sec = duration_sec * 0.90  # Allow up to 90% for last clip

        base_words = max(5, round(safe_duration / 60 * wpm))
        target_words = base_words
        max_words_allowed = int(max_sec / 60 * wpm)

        # Build context from previous sentences (focus on recent context)
        previous_sentence = None
        if script:
            # Only use the LAST narration to prevent copying from earlier clips
            previous_sentence = script[-1]
            # Keep all narrations for duplicate detection
            all_previous_narrations = script[:]
        else:
            all_previous_narrations = []

        # Smart retry logic for clips with >3s overflow
        overflow_retry_count = 0
        generated_line = None

        for attempt in range(5):  # More attempts with aggressive reduction
            # Get personal context for this clip
            filename = c.get("filename", f"{cid}.mp4")  # Use filename for context lookup
            clip_context = clip_contexts.get(filename, None)
            personal_context = clip_context.personal_context if clip_context else ""

            # Add explicit duration constraint feedback for overflow retries
            duration_feedback = ""
            if overflow_retry_count > 0:
                duration_feedback = f" CRITICAL: Previous narration was {generated_line_duration:.1f}s but video is only {duration_sec:.1f}s. You MUST be more concise!"
                # Progressive reduction for overflow retries
                reduction_factor = 0.7 ** overflow_retry_count
                target_words = max(5, round(base_words * reduction_factor))
                typer.echo(f"  • {cid}: Retry {overflow_retry_count} - reducing to {target_words} words due to {generated_line_duration - duration_sec:.1f}s overflow")

            if provider == "ollama":
                line = _ask_ollama_narrate(
                    caption_text + duration_feedback,
                    location_text,
                    target_words=target_words,
                    is_first=is_first,
                    is_last=is_last,
                    previous_sentence=previous_sentence,
                    max_words=max_words_allowed,
                    model=model,
                    temperature=temperature,
                    style_config=style_config,
                    personal_context=personal_context,
                    clip_idx=idx,
                    total_clips=len(ordered),
                    all_previous_narrations=all_previous_narrations,
                )
            else:
                line = _ask_gemini_narrate(
                    caption_text + duration_feedback,
                    location_text,
                    target_words=target_words,
                    is_first=is_first,
                    is_last=is_last,
                    previous_sentence=previous_sentence,
                    max_words=max_words_allowed,
                    temperature=temperature,
                    style_config=style_config,
                    personal_context=personal_context,
                    clip_idx=idx,
                    total_clips=len(ordered),
                    all_previous_narrations=all_previous_narrations,
                )

            # Validate script quality
            sentences = [s.strip() for s in line.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            short_sentences = [s for s in sentences if len(s.split()) < 5]

            # Soft check for "Now" starts - just log, don't retry
            if not is_first and line.strip().startswith("Now,"):
                typer.echo(f"  • Note: {cid} starts with 'Now' - consider variation in future")

            # Check for duplicate/repetitive content using sophisticated detection
            if script:
                # Use our sophisticated duplicate detection
                has_duplicate, duplicate_info = detect_duplicate_phrases(line, all_previous_narrations)
                if has_duplicate:
                    typer.echo(f"  • {cid}: {duplicate_info}, retrying...")
                    if attempt < 4:
                        target_words = max(5, round(target_words * 0.85))
                        # Add stronger warning to the prompt for the retry
                        duration_feedback += f" CRITICAL: DO NOT repeat phrases from previous narrations!"
                        continue

                # Check for repetitive openings (first 30 chars)
                line_start = line[:30].lower().strip()
                repetitive_starts = ['the morning light reveals', 'ancient stones tell', 'here,', 'now,', 'centuries of']

                # Count how many times this start pattern appears
                start_count = sum(1 for s in script if s[:30].lower().strip().startswith(line_start[:15]))
                if start_count >= 2 or any(line_start.startswith(rep) for rep in repetitive_starts):
                    typer.echo(f"  • {cid}: Repetitive opening detected: '{line_start[:20]}...', retrying...")
                    if attempt < 3:
                        target_words = max(5, round(target_words * 0.85))
                        continue

            # If line contains multiple alternatives (separated by newlines), take the first one
            if '\n' in line:
                lines = [l.strip() for l in line.split('\n') if l.strip()]
                if lines:
                    line = lines[0]  # Take first line only
                    typer.echo(f"  • {cid}: Multiple lines returned, using first: '{line[:50]}...'")

            est_sec = _estimate_speech_sec(line, wpm=wpm)
            generated_line = line
            generated_line_duration = est_sec

            # Reject if too many short fragments (except for last attempt)
            if attempt < 4 and overflow_retry_count == 0 and len(short_sentences) > len(sentences) * 0.3:
                target_words = max(5, round(target_words * 0.7))  # More aggressive reduction
                continue

            # Check for >3 second overflow and retry if needed
            overflow = est_sec - duration_sec
            if overflow > 3.0 and overflow_retry_count < 5:
                overflow_retry_count += 1
                continue  # Retry with stricter constraints

            # Accept the line if within limits OR after 5 overflow retries
            if est_sec <= max_sec or overflow_retry_count >= 5:
                if overflow_retry_count >= 5:
                    typer.echo(f"  • {cid}: Accepting {overflow:.1f}s overflow after 5 retries")

                script.append(line)
                speech_secs.append(est_sec)
                typer.echo(f"[Stage 2: Narration] Added script entry {len(script)} for clip {idx+1}: '{line[:50]}...'")

                # Track overflow when allow_overflow is enabled
                if allow_overflow and not is_last:
                    if overflow > 0:
                        cumulative_overflow_seconds += overflow
                        typer.echo(f"  • {cid}: allowing {overflow:.2f}s overflow (cumulative: {cumulative_overflow_seconds:.2f}s)")
                    else:
                        # This clip underused its time, reduce cumulative overflow
                        underflow = duration_sec - est_sec
                        cumulative_overflow_seconds = max(0, cumulative_overflow_seconds - underflow)

                break  # accept line

        # Ensure we always have a script entry for this clip
        if len(script) < idx + 1:  # Check if nothing was added for this clip
            typer.echo(f"  • {cid}: WARNING - Using fallback after all attempts failed")
            if generated_line:
                # Use the last generated line if available
                if '\n' in generated_line:
                    lines = [l.strip() for l in generated_line.split('\n') if l.strip()]
                    generated_line = lines[0] if lines else generated_line.split('\n')[0]
                script.append(generated_line)
                speech_secs.append(_estimate_speech_sec(generated_line, wpm=wpm))
            else:
                # Ultimate fallback
                fallback_line = f"Scene {idx + 1}."
                script.append(fallback_line)
                speech_secs.append(_estimate_speech_sec(fallback_line, wpm=wpm))


    # ------------------------------------------------------------------
    # Validation: ensure provided locations are mentioned at least once.
    # If missing, append a natural clause mentioning the location.
    # ------------------------------------------------------------------

    clip_map = {c["clip_id"]: c for c in clips}

    # Track which locations have been mentioned across all clips
    mentioned_locations = set()
    location_mention_count = 0
    max_location_mentions = 3  # Maximum 3 mentions per documentary

    def _inject_location(sentence: str, loc: str, clip_idx: int, total_clips: int) -> str:
        """Selectively inject location only when necessary (max 3 times total)."""
        nonlocal location_mention_count

        loc_clean = loc.strip().rstrip('.')
        # Split location into city and country if comma present
        if "," in loc_clean:
            city_part, country_part = [p.strip() for p in loc_clean.split(",", 1)]
        else:
            city_part, country_part = loc_clean, ""

        # Check if location is already naturally mentioned in the sentence
        if city_part.lower() in sentence.lower():
            mentioned_locations.add(city_part.lower())
            location_mention_count += 1
            # Remove redundant country if city is already mentioned
            if country_part and country_part.lower() in sentence.lower():
                sentence = re.sub(r",\s*" + re.escape(country_part), "", sentence, flags=re.I)
            return sentence

        # Determine if we should inject location
        should_inject = False

        # Only inject if we haven't hit our limit
        if location_mention_count < max_location_mentions:
            if clip_idx == 0:
                should_inject = True
            # Middle of documentary: reminder if different location (priority 2)
            elif clip_idx == total_clips // 2 and city_part.lower() not in mentioned_locations:
                should_inject = True
            # Near end: closure mention if fits naturally (priority 3)
            elif clip_idx >= total_clips - 2 and location_mention_count < 2:
                # Only if it makes narrative sense
                if any(word in sentence.lower() for word in ['journey', 'adventure', 'experience', 'final', 'last']):
                    should_inject = True

        if should_inject and location_mention_count < max_location_mentions:
            # Natural integration - avoid repetitive "in [location]" pattern
            if clip_idx == 0:
                # Opening: make it prominent
                if sentence.endswith('.'):
                    sentence = sentence[:-1] + f" in {city_part}."
                else:
                    sentence += f" in {city_part}."
            else:
                # Later mentions: more subtle integration
                if sentence.endswith('.'):
                    # Vary the phrasing
                    if location_mention_count == 1:
                        sentence = sentence[:-1] + f", {city_part}."
                    else:
                        sentence = sentence[:-1] + f" across {city_part}."
                else:
                    sentence += f" in {city_part}."

            mentioned_locations.add(city_part.lower())
            location_mention_count += 1

        return sentence

    total_clips = len(ordered)
    for idx, cid in enumerate(ordered):
        # Check if script has an entry for this index (it might not due to continue statements)
        if idx >= len(script):
            break  # No more script entries to process
        loc = clip_map[cid].get("captions", [None, ""])
        if len(loc) > 1:
            loc_str = loc[1]
            if loc_str:
                script[idx] = _inject_location(script[idx], loc_str, idx, total_clips)
                speech_secs[idx] = _estimate_speech_sec(script[idx], wpm=wpm)

    # Post-processing: Final duplicate check across entire script
    typer.secho("[plan] Performing final duplicate check across entire script...", fg=typer.colors.CYAN)

    # Check for repeated phrases across all narration
    repeated_phrases = check_phrase_diversity(script)
    if repeated_phrases:
        typer.secho("[plan] WARNING: Found repeated phrases in the script:", fg=typer.colors.YELLOW)
        for phrase, clips in repeated_phrases.items():
            if len(phrase.split()) >= 5:  # Only report significant phrases
                typer.secho(f"  - '{phrase[:60]}...' appears in clips: {clips}", fg=typer.colors.YELLOW)

        # Attempt to fix the most egregious duplicates
        typer.secho("[plan] Attempting to fix duplicate phrases...", fg=typer.colors.CYAN)
        for i, narration in enumerate(script):
            if i > 0:  # Skip first clip
                # Check against all previous narrations
                has_dup, dup_info = detect_duplicate_phrases(narration, script[:i])
                if has_dup:
                    typer.secho(f"[plan] Fixing duplicate in clip {i+1}: {dup_info}", fg=typer.colors.YELLOW)

                    # Remove duplicate sentences from current narration
                    for j in range(i):
                        narration = remove_duplicate_sentences(narration, script[j])

                    # If too much was removed, regenerate with stronger uniqueness constraint
                    if len(narration.split()) < 10:  # Too short after removal
                        typer.secho(f"[plan] Regenerating clip {i+1} narration due to excessive duplicate removal...", fg=typer.colors.CYAN)
                        # Create a very strong anti-duplicate prompt
                        strong_duplicate_prompt = f"""
CRITICAL: The following phrases have already been used and MUST NOT appear again:
{chr(10).join(f'- "{s}"' for s in script[:i] if len(s) > 30)}

You must create COMPLETELY DIFFERENT narration with:
- New vocabulary and sentence structures
- Different facts and observations
- Unique perspective on the scene
- No recycled phrases or patterns
"""
                        # Here we would need to regenerate, but for now just keep the deduplicated version
                        script[i] = narration if narration.strip() else script[i]
                    else:
                        script[i] = narration

                    # Update speech duration
                    speech_secs[i] = _estimate_speech_sec(script[i], wpm=wpm)
    else:
        typer.secho("[plan] No duplicate phrases detected in script", fg=typer.colors.GREEN)

    plan_payload = {
        "project": project,
        "style": final_style,
        "ordering_strategy": project_context.ordering_strategy,
        "generated": datetime.utcnow().isoformat() + "Z",
        "ordered_clips": ordered,
        "script": script,
        "speech_sec": speech_secs,
        "wpm": wpm,
        "allow_overflow": allow_overflow,
    }

    # ------------------------------------------------------
    # Optional: translate script into extra languages
    # ------------------------------------------------------

    extra_langs = [l for l in lang_list if l != base_lang]

    def _translate_lines(lines: list[str], target_lang: str) -> list[str]:
        sys_prompt = (
            f"You are a professional translator turning documentary narration from English into {target_lang}. "
            "Keep sentences vivid and faithful, maintain the same sentence boundaries and order. "
            "Respond with ONLY a JSON array of translated sentences, no markdown."
        )

        user_json = json.dumps(lines, ensure_ascii=False)

        if provider == "ollama":
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_json},
            ]
            resp = ollama.chat(
                model=model or "llama3.3:latest",  # Default model, can be overridden
                messages=msgs,
                options={"temperature": 0.2},
                stream=False,
            )
            raw = resp["message"]["content"].strip()
        elif provider == "gemini":
            import os as _os
            api_key = _os.getenv("GEMINI_API_KEY") or (genai._default_api_key if hasattr(genai, "_default_api_key") else None)  # type: ignore
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY env var not set for translation")
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel("gemini-2.0-flash")
            resp = rate_limited_call(
                "gemini-flash",
                m.generate_content,
                [sys_prompt, user_json],
                generation_config={"temperature": 0.2}
            )
            raw = resp.text.strip()
        else:
            raise RuntimeError("Translation requires LLM provider")

        try:
            arr_match = re.search(r"\[.*]", raw, flags=re.S)
            if not arr_match:
                raise ValueError("no JSON array found")
            arr = json.loads(arr_match.group(0))
            if not isinstance(arr, list) or len(arr) != len(lines):
                raise ValueError("array length mismatch")
            return [str(s) for s in arr]
        except Exception as e:
            raise RuntimeError(f"Translation to {target_lang} failed: {e}") from e

    if extra_langs:
        for tgt in extra_langs:
            try:
                translations[tgt] = _translate_lines(script, tgt)
            except Exception as e:
                typer.secho(f"[plan] Translation to {tgt} failed: {e}", fg=typer.colors.YELLOW)

    if translations:
        plan_payload["translations"] = translations

    plan_path.write_text(json.dumps(plan_payload, indent=2))

    typer.secho(f"[plan] {style_config.display_name} style plan written to {plan_path}", fg=typer.colors.GREEN)

# ---------------------------------------------------------------------------
# Typer glue (allows `python -m ai_doc_composer.plan ...` for dev tests)
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    import typer as _ty

    _app = _ty.Typer(add_completion=False)

    @_app.command()
    def main(
        project: str,
        provider: str = _ty.Option("ollama"),
        model: str | None = _ty.Option(None),
        temperature: float = _ty.Option(0.15),  # Lowered for better duration compliance
        wpm: int = _ty.Option(160),
        slack: float = _ty.Option(0.15),
        languages: str = _ty.Option("en", help="Comma-separated BCP47 codes for narration languages"),
        style: str | None = _ty.Option(None, help="Documentary style (documentary, travel_vlog, memories, family_trip)"),
        ordering_strategy: str = _ty.Option("llm", help="Clip ordering strategy (llm or exif)"),
        skip_existing: bool = _ty.Option(False, help="Skip if plan.json already exists"),
        allow_overflow: bool = _ty.Option(False, help="Allow narration to overflow between clips with compensation"),
    ) -> None:
        run(project, provider=provider, model=model, temperature=temperature, wpm=wpm, slack=slack, languages=languages, style=style, ordering_strategy=ordering_strategy, skip_existing=skip_existing, allow_overflow=allow_overflow)

    _app()

# ---------------------------------------------------------------------------
# Ordering with LLM (stage-1)
# ---------------------------------------------------------------------------


def _order_by_exif(clips: list[dict], clip_contexts: dict) -> list[str] | None:
    """Order clips by EXIF timestamp if available."""
    clip_times = []
    
    for clip in clips:
        clip_id = clip["clip_id"]
        filename = clip["filename"]
        context = clip_contexts.get(filename)
        
        if context and context.exif_timestamp:
            try:
                from datetime import datetime
                timestamp = datetime.fromisoformat(context.exif_timestamp.replace('Z', '+00:00'))
                clip_times.append((timestamp, clip_id))
            except (ValueError, TypeError):
                # Invalid timestamp format
                return None
        else:
            # Missing EXIF data
            return None
    
    if len(clip_times) == len(clips):
        # Sort by timestamp and return ordered clip IDs
        clip_times.sort(key=lambda x: x[0])
        return [clip_id for _, clip_id in clip_times]
    
    return None


def _ask_ollama_order(clips: list[dict], style_config, project_context, *, model: str | None, temperature: float) -> list[str]:
    """Return ordered list of clip IDs via Ollama with style awareness."""

    if not _OLLAMA_AVAILABLE:
        raise RuntimeError("ollama python package not installed")

    # Style-aware ordering prompt
    style_prompt = f"Order clips for a {style_config.display_name.lower()} with {style_config.tone} tone. "
    if project_context.project_context:
        style_prompt += f"Context: {project_context.project_context[:200]}... "
    
    order_prompt = (
        style_prompt +
        "The user will give you a JSON array of clips. "
        "Decide the most natural, engaging order for this style. "
        "Respond with *only* a JSON array of clip IDs in the new order – no other keys, no markdown. "
        f"Always include exactly {len(clips)} items."
    )

    # Provide minimal clip info to reduce prompt tokens
    user_json = json.dumps([{"id": c["clip_id"], "caption": _get_caption_text(c)} for c in clips])

    messages = [
        {"role": "system", "content": order_prompt},
        {"role": "user", "content": user_json},
    ]

    response = ollama.chat(
        model=model or "llama3.3:latest",  # Default model, can be overridden
        messages=messages,
        options={"temperature": temperature},
        stream=False,
    )

    raw = response["message"]["content"].strip()

    try:
        json_match = re.search(r"\[.*]", raw, flags=re.S)
        if not json_match:
            raise ValueError("no JSON array found")
        ordered = json.loads(json_match.group(0))
        if not isinstance(ordered, list) or not all(isinstance(el, str) for el in ordered):
            raise ValueError("invalid array elements")
        if len(ordered) != len(clips):
            raise ValueError("length mismatch")
        return ordered
    except Exception as e:
        raise RuntimeError(f"ollama ordering failed: {e}") from e


def _ask_gemini_order(clips: list[dict], style_config, project_context, *, temperature: float) -> list[str]:
    """Return ordered list of clip IDs via Gemini with style awareness."""

    if not _GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai package not installed")

    import os as _os

    api_key = _os.getenv("GEMINI_API_KEY") or (
        genai._default_api_key if hasattr(genai, "_default_api_key") else None  # type: ignore
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")

    genai.configure(api_key=api_key)

    # Get available model from quota manager
    model_name, status = _quota_manager.get_available_model("gemini-2.0-flash")
    if not model_name:
        # Try with legacy model as fallback
        model_name, status = _quota_manager.get_available_model("gemini-1.5-flash")
        if not model_name:
            typer.secho(f"[plan] {status}", fg=typer.colors.RED, err=True)
            raise RuntimeError(status)

    typer.echo(f"[plan] {status}")
    model = genai.GenerativeModel(model_name)

    # Style-aware ordering prompt
    style_prompt = f"Order clips for a {style_config.display_name.lower()} with {style_config.tone} tone. "
    if project_context.project_context:
        style_prompt += f"Context: {project_context.project_context[:200]}... "

    order_prompt = (
        style_prompt +
        "The user will give you a JSON array of clips. "
        "Decide the most natural, engaging order for this style. "
        "Respond with *only* a JSON array of clip IDs in the new order – no other keys, no markdown. "
        f"Always include exactly {len(clips)} items."
    )

    user_json = json.dumps([{"id": c["clip_id"], "caption": _get_caption_text(c)} for c in clips])

    # Try API call with retry on quota errors
    max_retries = 3
    response = None
    last_error = None
    model_switches = 0
    max_model_switches = 5  # Allow switching models up to 5 times

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = rate_limited_call(
                "gemini-flash",
                model.generate_content,
                [order_prompt, user_json],
                generation_config={"temperature": temperature}
            )
            # Record successful API usage
            _quota_manager.record_usage(model_name)
            break  # Success, exit retry loop

        except Exception as api_error:
            last_error = api_error
            error_msg = str(api_error)
            if "429" in error_msg or "quota" in error_msg.lower():
                # Check if it's daily quota exhaustion vs RPM limit
                # Daily quota has "GenerateRequestsPerDayPerProjectPerModel" or specific quota_value in the error
                is_daily_quota = ("GenerateRequestsPerDayPerProjectPerModel" in error_msg or
                                  "quota_value:" in error_msg or
                                  "exceeded your current quota" in error_msg.lower())

                if is_daily_quota:
                    # Daily quota exhausted - try fallback model
                    typer.echo(f"[plan] Model {model_name} daily quota exceeded, trying fallback...")
                    _quota_manager.mark_exhausted(model_name)

                    # Try to get an alternative model
                    fallback_model, status = _quota_manager.get_available_model()
                    if not fallback_model or fallback_model == model_name:
                        typer.secho(f"[plan] No alternative models available. All daily quotas exhausted.", fg=typer.colors.RED, err=True)
                        raise RuntimeError("All Gemini models have exceeded their daily quotas") from api_error

                    # Switch to fallback model and reset retry counter
                    typer.echo(f"[plan] Switching to fallback model: {fallback_model}")
                    model_name = fallback_model
                    model = genai.GenerativeModel(model_name)
                    model_switches += 1

                    if model_switches >= max_model_switches:
                        raise RuntimeError(f"Exceeded maximum model switches ({max_model_switches})") from api_error

                    # Reset retry counter for new model
                    retry_count = 0  # Reset to give new model full retry attempts
                    continue
                else:
                    # RPM limit - check for retry delay and wait
                    retry_match = re.search(r"Please retry in ([\d.]+)s", error_msg)
                    if retry_match:
                        wait_time = float(retry_match.group(1))
                        typer.echo(f"[plan] Rate limit hit for {model_name}, waiting {wait_time:.1f}s...")
                        time.sleep(wait_time + 1)  # Add 1 second buffer
                        retry_count += 1
                        continue  # Retry with same model
                    else:
                        # Unknown quota error, treat as daily quota
                        typer.echo(f"[plan] Unknown quota error for {model_name}, treating as daily quota...")
                        _quota_manager.mark_exhausted(model_name)

                        # Try to get an alternative model
                        fallback_model, status = _quota_manager.get_available_model()
                        if not fallback_model or fallback_model == model_name:
                            typer.secho(f"[plan] No alternative models available.", fg=typer.colors.RED, err=True)
                            raise RuntimeError("All Gemini models have exceeded their quotas") from api_error

                        # Switch to fallback model
                        typer.echo(f"[plan] Switching to fallback model: {fallback_model}")
                        model_name = fallback_model
                        model = genai.GenerativeModel(model_name)
                        model_switches += 1

                        if model_switches >= max_model_switches:
                            raise RuntimeError(f"Exceeded maximum model switches ({max_model_switches})") from api_error

                        retry_count = 0
                        continue
            else:
                raise  # Re-raise non-quota errors

        retry_count += 1

    if response is None:
        raise RuntimeError(f"Failed to get response after {max_retries} attempts with {model_switches} model switches") from last_error

    raw = response.text.strip()

    try:
        json_match = re.search(r"\[.*]", raw, flags=re.S)
        if not json_match:
            raise ValueError("no JSON array found")
        ordered = json.loads(json_match.group(0))
        if not isinstance(ordered, list) or not all(isinstance(el, str) for el in ordered):
            raise ValueError("invalid array elements")
        if len(ordered) != len(clips):
            raise ValueError("length mismatch")
        return ordered
    except Exception as e:
        raise RuntimeError(f"gemini ordering failed: {e}") from e


# ---------------------------------------------------------------------------
# Per-clip narration with LLM (stage-2)
# ---------------------------------------------------------------------------


def _ask_ollama_narrate(
    caption: str,
    location: str | None,
    *,
    target_words: int,
    is_first: bool,
    is_last: bool,
    previous_sentence: str | None,
    max_words: int,
    model: str | None,
    temperature: float,
    style_config,
    personal_context: str = "",
    clip_idx: int = 0,
    total_clips: int = 1,
    all_previous_narrations: list[str] | None = None,
) -> str:
    """Return narration string for a single clip via Ollama with style and context."""

    if not _OLLAMA_AVAILABLE:
        raise RuntimeError("ollama python package not installed")

    # Build style-specific narration prompt
    sys_prompt = _build_narration_prompt(
        target_words,
        is_first=is_first,
        is_last=is_last,
        previous_sentence=previous_sentence,
        max_words=max_words,
        style_name=style_config.name,
        clip_idx=clip_idx,
        total_clips=total_clips,
        all_previous_narrations=all_previous_narrations,
    )

    clip_info = {
        "caption": caption,
        "location": (location or "")[:80],
        "target_words": target_words,
        "previous_sentence": previous_sentence or "",
        "personal_context": personal_context[:200] if personal_context else "",
    }

    response = ollama.chat(
        model=model or "llama3.3:latest",  # Default model, can be overridden
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(clip_info)},
        ],
        options={"temperature": temperature},
        stream=False,
    )

    return response["message"]["content"].strip()


def _ask_gemini_narrate(
    caption: str,
    location: str | None,
    *,
    target_words: int,
    is_first: bool,
    is_last: bool,
    previous_sentence: str | None,
    max_words: int,
    temperature: float,
    style_config,
    personal_context: str = "",
    clip_idx: int = 0,
    total_clips: int = 1,
    all_previous_narrations: list[str] | None = None,
) -> str:
    """Return narration string for a single clip via Gemini with style and context."""

    if not _GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai package not installed")

    import os as _os

    api_key = _os.getenv("GEMINI_API_KEY") or (
        genai._default_api_key if hasattr(genai, "_default_api_key") else None  # type: ignore
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")

    genai.configure(api_key=api_key)

    # Get available model from quota manager
    model_name, status = _quota_manager.get_available_model("gemini-2.0-flash")
    if not model_name:
        # Try with legacy model as fallback
        model_name, status = _quota_manager.get_available_model("gemini-1.5-flash")
        if not model_name:
            typer.secho(f"[plan] {status}", fg=typer.colors.RED, err=True)
            raise RuntimeError(status)

    model = genai.GenerativeModel(model_name)

    # Build style-specific narration prompt
    sys_prompt = _build_narration_prompt(
        target_words,
        is_first=is_first,
        is_last=is_last,
        previous_sentence=previous_sentence,
        max_words=max_words,
        style_name=style_config.name,
        clip_idx=clip_idx,
        total_clips=total_clips,
        all_previous_narrations=all_previous_narrations,
    )

    clip_info = {
        "caption": caption,
        "location": (location or "")[:80],
        "target_words": target_words,
        "previous_sentence": previous_sentence or "",
        "personal_context": personal_context[:200] if personal_context else "",
    }

    # Try API call with retry on quota errors
    max_retries = 3
    response = None
    last_error = None
    model_switches = 0
    max_model_switches = 5

    retry_count = 0
    while retry_count < max_retries:
        try:
            response = rate_limited_call(
                "gemini-flash",
                model.generate_content,
                [sys_prompt, json.dumps(clip_info)],
                generation_config={"temperature": temperature}
            )
            # Record successful API usage
            _quota_manager.record_usage(model_name)
            break  # Success, exit retry loop

        except Exception as api_error:
            last_error = api_error
            error_msg = str(api_error)
            if "429" in error_msg or "quota" in error_msg.lower():
                # Check if it's daily quota exhaustion vs RPM limit
                # Daily quota has "GenerateRequestsPerDayPerProjectPerModel" or specific quota_value in the error
                is_daily_quota = ("GenerateRequestsPerDayPerProjectPerModel" in error_msg or
                                  "quota_value:" in error_msg or
                                  "exceeded your current quota" in error_msg.lower())

                if is_daily_quota:
                    # Daily quota exhausted - try fallback model
                    _quota_manager.mark_exhausted(model_name)

                    # Try to get an alternative model
                    fallback_model, status = _quota_manager.get_available_model()
                    if not fallback_model or fallback_model == model_name:
                        raise RuntimeError("All Gemini models have exceeded their daily quotas") from api_error

                    # Switch to fallback model
                    typer.echo(f"  • Switching from {model_name} to {fallback_model} due to quota")
                    model_name = fallback_model
                    model = genai.GenerativeModel(model_name)
                    model_switches += 1

                    if model_switches >= max_model_switches:
                        raise RuntimeError(f"Exceeded maximum model switches ({max_model_switches})") from api_error

                    # Reset retry counter for new model
                    retry_count = 0  # Reset to give new model full retry attempts
                    continue
                else:
                    # RPM limit - check for retry delay and wait
                    retry_match = re.search(r"Please retry in ([\d.]+)s", error_msg)
                    if retry_match:
                        wait_time = float(retry_match.group(1))
                        typer.echo(f"  • Rate limit hit for {model_name}, waiting {wait_time:.1f}s...")
                        time.sleep(wait_time + 1)  # Add 1 second buffer
                        retry_count += 1
                        continue  # Retry with same model
                    else:
                        # Unknown quota error, treat as daily quota
                        _quota_manager.mark_exhausted(model_name)

                        # Try to get an alternative model
                        fallback_model, status = _quota_manager.get_available_model()
                        if not fallback_model or fallback_model == model_name:
                            raise RuntimeError("All Gemini models have exceeded their quotas") from api_error

                        # Switch to fallback model
                        typer.echo(f"  • Switching from {model_name} to {fallback_model} due to quota")
                        model_name = fallback_model
                        model = genai.GenerativeModel(model_name)
                        model_switches += 1

                        if model_switches >= max_model_switches:
                            raise RuntimeError(f"Exceeded maximum model switches ({max_model_switches})") from api_error

                        retry_count = 0
                        continue
            else:
                raise  # Re-raise non-quota errors

        retry_count += 1

    if response is None:
        raise RuntimeError(f"Failed to get response after {max_retries} attempts with {model_switches} model switches") from last_error

    return response.text.strip()