"""
Modern Streamlit-based AI Documentary Composer UI
"""

import streamlit as st
import os
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tempfile

# Import modules with graceful fallbacks for different execution contexts
try:
    # Try relative imports first (when run as module)
    from . import ingest, plan, render, quality, preprocess
    from .styles import DocumentaryStyles, StyleType
    from .context import ContextManager
    from .stt import STTTranscriber, create_temp_audio_file, save_voice_sample
except ImportError:
    try:
        # Try absolute imports (when run directly with streamlit)
        import sys
        from pathlib import Path
        # Add the parent directory to Python path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ai_doc_composer import ingest, plan, render, quality, preprocess
        from ai_doc_composer.styles import DocumentaryStyles, StyleType
        from ai_doc_composer.context import ContextManager
        from ai_doc_composer.stt import STTTranscriber, create_temp_audio_file, save_voice_sample
    except ImportError as e:
        st.warning(f"Some modules not available: {e}")
        ingest = plan = render = quality = preprocess = None
        DocumentaryStyles = StyleType = ContextManager = None
        STTTranscriber = save_voice_sample = create_temp_audio_file = None

# TTS module will be imported lazily when needed
tts = None

def _get_tts_module():
    """Lazy import TTS module with error handling"""
    global tts
    if tts is None:
        try:
            # Try relative import first
            from . import tts as tts_module
            tts = tts_module
        except ImportError:
            try:
                # Try absolute import
                from ai_doc_composer import tts as tts_module
                tts = tts_module
            except ImportError as e:
                st.warning(f"TTS module not available: {e}")
                tts = False
    return tts if tts is not False else None

def get_stt_transcriber():
    """Get STT transcriber with error handling."""
    try:
        return STTTranscriber(whisper_model="base", use_faster_whisper=True)
    except ImportError:
        return None

def get_user_voice_samples_safe(project_path: Path) -> dict:
    """Get user voice samples with proper import handling."""
    try:
        # Try to import get_user_voice_samples with fallbacks
        try:
            from .stt import get_user_voice_samples
        except ImportError:
            try:
                from ai_doc_composer.stt import get_user_voice_samples
            except ImportError:
                return {}
        
        return get_user_voice_samples(project_path)
    except Exception:
        return {}

def save_user_voice_sample(audio_data: bytes, project_name: str, language: str = "en") -> bool:
    """Save user voice sample with proper import handling."""
    try:
        # Try to import save_voice_sample with fallbacks
        try:
            from .stt import save_voice_sample
        except ImportError:
            try:
                from ai_doc_composer.stt import save_voice_sample
            except ImportError:
                st.error("STT module not available")
                return False
        
        project_path = Path("projects") / project_name
        if project_path.exists():
            save_voice_sample(audio_data, project_path, language)
            st.success(f"✅ Voice sample saved for {language.upper()}")
            return True
        else:
            st.warning("⚠️ Project directory does not exist")
            return False
    except Exception as e:
        st.warning(f"⚠️ Could not save voice sample: {e}")
        return False

def transcribe_audio_data(audio_data: bytes, language: str = "en", 
                          project_name: Optional[str] = None, save_voice: bool = False) -> Optional[str]:
    """Transcribe audio data to text and optionally save as voice sample."""
    if not audio_data:
        return None
    
    transcriber = get_stt_transcriber()
    if not transcriber:
        st.error("Speech-to-text not available. Please install: pip install openai-whisper")
        return None
    
    try:
        # Create temporary audio file
        try:
            temp_audio = create_temp_audio_file(audio_data, suffix=".wav")
        except NameError:
            # Fallback if create_temp_audio_file not available
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(audio_data)
            temp_file.close()
            temp_audio = Path(temp_file.name)
        
        # Transcribe
        with st.spinner("🎙️ Transcribing audio..."):
            text = transcriber.transcribe_audio_simple(temp_audio, language)
        
        # Save voice sample if requested and project exists
        if save_voice and project_name:
            save_user_voice_sample(audio_data, project_name, language)
        
        # Clean up
        temp_audio.unlink(missing_ok=True)
        
        return text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

def reset_plan_pipeline(project_name: str) -> bool:
    """Delete plan.json and all downstream outputs (TTS and final video)."""
    try:
        project_path = Path("projects") / project_name
        if not project_path.exists():
            return False

        # Delete plan.json
        plan_file = project_path / "json" / "plan.json"
        if plan_file.exists():
            plan_file.unlink()

        # Delete tts_meta.json
        tts_meta_file = project_path / "json" / "tts_meta.json"
        if tts_meta_file.exists():
            tts_meta_file.unlink()

        # Clean output folder
        output_dir = project_path / "output"
        if output_dir.exists():
            # Delete all files in output directory
            for file in output_dir.rglob("*"):
                if file.is_file():
                    file.unlink(missing_ok=True)
            # Clean up empty directories
            for dir_path in sorted(output_dir.rglob("*"), reverse=True):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()

        return True
    except Exception as e:
        st.error(f"Failed to reset pipeline: {e}")
        return False

def clean_tts_only(project_name: str) -> bool:
    """Delete TTS outputs but keep plan intact."""
    try:
        project_path = Path("projects") / project_name
        if not project_path.exists():
            return False

        # Delete tts_meta.json
        tts_meta_file = project_path / "json" / "tts_meta.json"
        if tts_meta_file.exists():
            tts_meta_file.unlink()

        # Clean audio folder
        audio_dir = project_path / "output" / "audio"
        if audio_dir.exists():
            shutil.rmtree(audio_dir)

        # Delete narration WAV files
        output_dir = project_path / "output"
        if output_dir.exists():
            for narration_file in output_dir.glob("narration_*.wav"):
                narration_file.unlink(missing_ok=True)

        # Delete final video and intermediate video files
        for video_file in ["final.mp4", f"{project_name}.mp4", "_video_noaudio.mp4"]:
            video_path = output_dir / video_file
            if video_path.exists():
                video_path.unlink()

        return True
    except Exception as e:
        st.error(f"Failed to clean TTS: {e}")
        return False

def find_video_file(project_name: str, filename: str) -> Optional[Path]:
    """Find video file, checking both preview and original directories."""
    project_path = Path("projects") / project_name
    
    # Extract base name without extension
    base_name = filename.split('.')[0] if '.' in filename else filename
    
    # Priority 1: Check preview directory (preprocessed files)
    preview_dir = project_path / "input_preview"
    if preview_dir.exists():
        # Look for .mp4 version first (most common after preprocessing)
        preview_mp4 = preview_dir / f"{base_name}.mp4"
        if preview_mp4.exists():
            return preview_mp4
        
        # Look for any matching file in preview directory
        for ext in ['.mp4', '.mov', '.avi', '.mkv']:
            preview_file = preview_dir / f"{base_name}{ext}"
            if preview_file.exists():
                return preview_file
    
    # Priority 2: Check original input directory
    input_dir = project_path / "input"
    if input_dir.exists():
        # Look for exact filename match first
        exact_match = input_dir / filename
        if exact_match.exists():
            return exact_match
        
        # Look for any matching file with different extension
        for ext in ['.MOV', '.mov', '.mp4', '.avi', '.mkv']:
            original_file = input_dir / f"{base_name}{ext}"
            if original_file.exists():
                return original_file
    
    return None

def extract_video_thumbnail(video_path: Path) -> Optional[any]:
    """Extract first frame from video as thumbnail."""
    try:
        import subprocess
        from PIL import Image
        import io
        
        # Ensure video file exists
        if not video_path.exists():
            return None
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vframes", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-loglevel", "quiet",  # Suppress ffmpeg output
            "-"
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            return None
        
        if not result.stdout:
            return None
            
        img = Image.open(io.BytesIO(result.stdout))
        return img
        
    except Exception as e:
        return None

# Page configuration
st.set_page_config(
    page_title="🎬 AI Documentary Composer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stage-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .stage-card.pending {
        border-left-color: #FFC107;
        background: #fff8e1;
    }
    .stage-card.in-progress {
        border-left-color: #2196F3;
        background: #e3f2fd;
    }
    .stage-card.completed {
        border-left-color: #4CAF50;
        background: #e8f5e8;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .project-timeline {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .upload-zone {
        border: 2px dashed #4CAF50;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f8fffe;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_existing_projects() -> List[str]:
    """Get list of existing project names sorted by most recent activity"""
    projects_dir = Path("projects")
    if not projects_dir.exists():
        return []

    # Get all valid project directories
    project_dirs = [p for p in projects_dir.iterdir() if p.is_dir() and p.name != "__pycache__"]

    # Sort by most recent modification time
    def get_project_last_modified(project_path: Path) -> float:
        """Get the most recent modification time from project's JSON files"""
        max_mtime = project_path.stat().st_mtime  # Start with directory's own mtime

        # Check JSON directory for recent activity
        json_dir = project_path / "json"
        if json_dir.exists():
            for json_file in json_dir.glob("*.json"):
                mtime = json_file.stat().st_mtime
                max_mtime = max(max_mtime, mtime)

        # Check output directory for recent renders
        output_dir = project_path / "output"
        if output_dir.exists():
            for output_file in output_dir.iterdir():
                if output_file.is_file():
                    mtime = output_file.stat().st_mtime
                    max_mtime = max(max_mtime, mtime)

        return max_mtime

    # Sort projects by most recent activity (newest first)
    sorted_projects = sorted(project_dirs, key=get_project_last_modified, reverse=True)
    return [p.name for p in sorted_projects]

def get_project_stage_status(project_name: str) -> Dict[str, Any]:
    """Analyze project completion status for each stage"""
    project_path = Path("projects") / project_name
    json_path = project_path / "json"
    input_path = project_path / "input"
    
    stages = {
        "ingest": {"file": "captions.json", "completed": False, "data": None},
        "plan": {"file": "plan.json", "completed": False, "data": None},
        "tts": {"file": "tts_meta.json", "completed": False, "data": None},
        "render": {"file": None, "completed": False, "data": None}
    }
    
    # Get input files info
    input_files = []
    if input_path.exists():
        input_files = [f.name for f in input_path.iterdir() if f.suffix.lower() in ['.mp4', '.mov', '.avi']]
    
    stages["input_files"] = input_files
    
    # Check JSON files
    for stage, info in stages.items():
        if stage == "input_files":
            continue
        if info["file"]:
            file_path = json_path / info["file"]
            if file_path.exists():
                stages[stage]["completed"] = True
                try:
                    with open(file_path) as f:
                        stages[stage]["data"] = json.load(f)
                except Exception:
                    pass
    
    # Check for final video
    final_video = project_path / "output" / f"{project_name}.mp4"
    if not final_video.exists():
        final_video = project_path / "output" / "final.mp4"
    stages["render"]["completed"] = final_video.exists()
    if stages["render"]["completed"]:
        stages["render"]["data"] = {"video_path": str(final_video)}
    
    return stages

def display_pipeline_status(project_name: str):
    """Display beautiful pipeline status with cards"""
    stages = get_project_stage_status(project_name)
    
    stage_info = [
        ("ingest", "1️⃣ Vision Captioning", "Extract frames and generate captions"),
        ("plan", "2️⃣ Language Planning", "Create narrative script and clip ordering"),  
        ("tts", "3️⃣ Text-to-Speech", "Synthesize voice narration"),
        ("render", "4️⃣ Video Rendering", "Combine video clips with audio")
    ]
    
    st.subheader("🔄 Processing Pipeline")
    
    cols = st.columns(4)
    for i, (stage_key, stage_name, description) in enumerate(stage_info):
        with cols[i]:
            stage_data = stages.get(stage_key, {})
            completed = stage_data.get("completed", False)
            
            if completed:
                status_class = "completed"
                icon = "✅"
                status_text = "COMPLETED"
            else:
                status_class = "pending"
                icon = "⏳"
                status_text = "PENDING"
            
            st.markdown(f"""
            <div class="stage-card {status_class}">
                <h4>{icon} {stage_name}</h4>
                <p><strong>{status_text}</strong></p>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Overall progress
    completed_count = sum(1 for stage_key, _, _ in stage_info 
                         if stages.get(stage_key, {}).get("completed", False))
    
    progress = completed_count / 4
    st.progress(progress, text=f"Overall Progress: {completed_count}/4 stages completed")
    
    return stages

def display_project_metrics(stages: Dict[str, Any]):
    """Display key project metrics in cards"""
    st.subheader("📊 Project Metrics")
    
    cols = st.columns(4)
    
    # Input files count
    input_files = stages.get("input_files", [])
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(input_files)}</h3>
            <p>Video Clips</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Total captions (if available)
    captions_data = stages.get("ingest", {}).get("data", {})
    total_captions = 0
    if captions_data and "clips" in captions_data:
        total_captions = sum(len(clip.get("captions", [])) for clip in captions_data["clips"])
    
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_captions}</h3>
            <p>AI Captions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Script segments (if available)
    plan_data = stages.get("plan", {}).get("data")
    script_segments = len(plan_data.get("script", [])) if plan_data else 0
    
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{script_segments}</h3>
            <p>Script Segments</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Total narration time (if available)
    total_speech_time = 0.0
    if plan_data:
        total_speech_time = sum(plan_data.get("speech_sec", []))

    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_speech_time:.1f}s</h3>
            <p>Narration Time</p>
        </div>
        """, unsafe_allow_html=True)

def display_video_timeline(project_name: str, stages: Dict[str, Any]):
    """Display interactive video timeline"""
    st.subheader("🎬 Documentary Timeline")
    
    plan_data = stages.get("plan", {}).get("data")
    
    if not plan_data or "ordered_clips" not in plan_data:
        st.info("Timeline will appear here once documentary plan is generated")
        
        # Show input videos in original order
        input_files = stages.get("input_files", [])
        if input_files:
            st.write(f"**{len(input_files)} clips ready for processing:**")
            for i, filename in enumerate(input_files, 1):
                st.write(f"{i}. `{filename}`")
        return
    
    # Display ordered timeline
    ordered_clips = plan_data["ordered_clips"]
    script_lines = plan_data.get("script", [])
    speech_durations = plan_data.get("speech_sec", [])
    
    st.markdown("""
    <div class="project-timeline">
    """, unsafe_allow_html=True)
    
    for i, clip_id in enumerate(ordered_clips):
        narration = script_lines[i] if i < len(script_lines) else "No narration"
        duration = speech_durations[i] if i < len(speech_durations) else 0
        
        with st.expander(f"**{i+1}. {clip_id}** ({duration:.1f}s)", expanded=False):
            st.write(f"**🎙️ Narration:** *\"{narration}\"*")
            
            # Try to show video thumbnail if available
            project_path = Path("projects") / project_name / "input"
            matching_files = [f for f in project_path.glob("*") if clip_id in f.name]
            if matching_files:
                st.video(str(matching_files[0]))
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.write(f"**Total clips:** {len(ordered_clips)} | **Total narration:** {sum(speech_durations):.1f} seconds")

def display_stage_details(stages: Dict[str, Any]):
    """Display detailed information for each stage"""
    
    # Vision Captions Analysis
    if stages.get("ingest", {}).get("completed"):
        with st.expander("📸 Vision Captions Analysis", expanded=False):
            captions_data = stages["ingest"]["data"]
            if captions_data and "clips" in captions_data:
                clips = captions_data["clips"]
                
                st.write("### 📊 Processing Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Clips Processed", len(clips))
                    st.metric("Generation Time", captions_data.get('generated', 'Unknown'))
                with col2:
                    total_duration = sum(clip.get("duration", 0) for clip in clips)
                    avg_captions = sum(len(clip.get("captions", [])) for clip in clips) / len(clips) if clips else 0
                    st.metric("Total Duration", f"{total_duration:.1f}s")
                    st.metric("Avg Captions/Clip", f"{avg_captions:.1f}")
                
                st.write("### 🎥 Clip Details")
                for i, clip in enumerate(clips, 1):
                    with st.expander(f"**{i}. {clip.get('filename', 'Unknown')}** ({clip.get('duration', 0):.1f}s)"):
                        for j, caption in enumerate(clip.get('captions', []), 1):
                            st.write(f"{j}. *{caption}*")
    
    # Documentary Plan Details
    if stages.get("plan", {}).get("completed"):
        with st.expander("🎬 Documentary Plan Details", expanded=False):
            plan_data = stages["plan"]["data"]
            if plan_data:
                st.write("### 📊 Plan Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Script Segments", len(plan_data.get("script", [])))
                    st.metric("Total Narration", f"{sum(plan_data.get('speech_sec', [])):.1f}s")
                with col2:
                    total_words = sum(len(line.split()) for line in plan_data.get("script", []))
                    st.metric("Estimated Words", total_words)
                    st.metric("Speaking Pace", f"{plan_data.get('wpm', 0)} WPM")
                
                # Languages
                translations = plan_data.get('translations', {})
                if translations:
                    st.write(f"**🌍 Languages:** {', '.join(translations.keys())}")
                
                st.write("### 🎭 Complete Documentary Script")
                script_lines = plan_data.get("script", [])
                speech_sec = plan_data.get("speech_sec", [])
                ordered_clips = plan_data.get("ordered_clips", [])
                
                for i, script_line in enumerate(script_lines):
                    duration = speech_sec[i] if i < len(speech_sec) else 0
                    clip_name = ordered_clips[i] if i < len(ordered_clips) else "Unknown"
                    
                    st.write(f"**{i+1}. {clip_name}** ({duration:.1f}s)")
                    st.write(f"*\"{script_line}\"*")
                    st.write("---")
    
    # TTS Analysis
    if stages.get("tts", {}).get("completed"):
        with st.expander("🎙️ Text-to-Speech Analysis", expanded=False):
            tts_data = stages["tts"]["data"]
            if tts_data:
                if "tracks" in tts_data:
                    tracks = tts_data["tracks"]
                    
                    st.write("### 📊 Audio Statistics")
                    total_clips = sum(len(track.get("clips", [])) for track in tracks)
                    total_duration = sum(
                        sum(clip.get("duration", 0) for clip in track.get("clips", []))
                        for track in tracks
                    )
                    languages = [track.get("lang", "unknown").upper() for track in tracks]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Audio Files", total_clips)
                        st.metric("Total Duration", f"{total_duration:.1f}s")
                    with col2:
                        st.metric("Language Tracks", len(tracks))
                        st.metric("Languages", ", ".join(languages))
                    
                    st.write("### 🎵 Audio Tracks by Language")
                    for track in tracks:
                        lang = track.get("lang", "unknown").upper()
                        clips = track.get("clips", [])
                        track_duration = sum(clip.get("duration", 0) for clip in clips)
                        
                        with st.expander(f"**🌍 {lang} Track** - {len(clips)} clips, {track_duration:.1f}s total"):
                            for i, clip in enumerate(clips, 1):
                                clip_id = clip.get('id', 'Unknown')
                                duration = clip.get('duration', 0)
                                wav_path = clip.get('wav', '').split('/')[-1] if clip.get('wav') else 'Unknown'
                                
                                st.write(f"{i}. **{clip_id}** ({duration:.1f}s)")
                                st.write(f"   📄 `{wav_path}`")

def process_video_pipeline(
    video_files: List[str],
    project_name: str,
    caption_provider: str,
    plan_provider: str,
    tts_provider: str,
    languages: str,
    temperature: float,
    speaker_voice: Optional[str],
    skip_existing: bool = False,
    auto_preprocess: bool = True,
    full_quality_render: bool = False,
    style: str = "documentary",
    project_context: str = "",
    clip_contexts: Dict[str, str] = None,
    ordering_strategy: str = "llm",
    use_user_voice: bool = False,
    ollama_model: Optional[str] = None,
    allow_overflow: bool = False
) -> Tuple[str, Optional[str], str]:
    """
    Process video pipeline with progress tracking
    Returns: (status_message, final_video_path, quality_report)
    """
    try:
        if not project_name.strip():
            return "❌ Project name is required", None, "No processing performed"
        
        project_path = Path("projects") / project_name
        input_path = project_path / "input"
        
        # Check if this is an existing project or new one
        is_existing_project = project_path.exists()
        
        if not is_existing_project:
            # New project - check for video files
            if not video_files:
                return "❌ No video files uploaded", None, "No processing performed"
            
            # Create directory and copy files
            input_path.mkdir(parents=True)
            
            # Copy uploaded files with original filenames
            for video_file in video_files:
                if video_file:
                    original_filename = Path(video_file.name).name
                    with open(input_path / original_filename, "wb") as f:
                        f.write(video_file.getvalue())
        else:
            # Existing project - check if input files exist
            if not input_path.exists() or not any(f for f in input_path.iterdir() if f.suffix.lower() in ['.mp4', '.mov', '.avi']):
                return "❌ No input video files found in existing project", None, "Project input directory is empty"
        
        # Stage 0: Preprocessing (if enabled and needed)
        if auto_preprocess and preprocess:
            try:
                # Check if preprocessing is needed
                if preprocess.needs_preprocessing(project_name, max_size_mb=20):
                    st.info("🔄 Stage 0: Preprocessing large video files for API compatibility...")
                    preprocess.run(project_name, target_size_mb=8, force=False)
                    st.success("✅ Video preprocessing completed!")
            except Exception as e:
                st.warning(f"⚠️ Preprocessing failed: {e}. Continuing with original files...")
        
        # Run the actual pipeline stages
        try:
            # Stage 1: Ingest (Vision Captioning)
            if ingest and (not skip_existing or not (project_path / "json" / "captions.json").exists()):
                st.info("🔄 Stage 1: Generating captions from video frames...")
                ingest.run(project_name, frames=1, provider=caption_provider)
            
            # Update project context and clip contexts
            if ContextManager:
                context_manager = ContextManager(project_path)
                
                # Update project context
                proj_context = context_manager.load_project_context()
                proj_context.style = style
                proj_context.project_context = project_context
                proj_context.ordering_strategy = ordering_strategy
                context_manager.save_project_context(proj_context)
                
                # Update clip contexts if provided
                if clip_contexts:
                    for filename, personal_context in clip_contexts.items():
                        if personal_context.strip():
                            context_manager.update_clip_context(filename, personal_context)
            
            # Stage 2: Plan (Language Planning)
            if plan and (not skip_existing or not (project_path / "json" / "plan.json").exists()):
                st.info(f"🔄 Stage 2: Creating {style} narrative plan and script...")
                plan_kwargs = {
                    "provider": plan_provider,
                    "temperature": temperature,
                    "languages": languages,
                    "style": style,
                    "ordering_strategy": ordering_strategy,
                    "skip_existing": skip_existing,
                    "allow_overflow": allow_overflow
                }
                # Add model parameter if using ollama
                if plan_provider == "ollama" and ollama_model:
                    plan_kwargs["model"] = ollama_model
                plan.run(project_name, **plan_kwargs)
            
            # Stage 3: TTS (Text-to-Speech)
            tts_module = _get_tts_module()
            if tts_module and (not skip_existing or not (project_path / "json" / "tts_meta.json").exists()):
                st.info("🔄 Stage 3: Synthesizing voice narration...")
                
                # Handle voice configuration
                speaker_config = None
                
                if use_user_voice:
                    # Use user's recorded voice samples
                    user_voices = get_user_voice_samples_safe(project_path)
                    
                    if user_voices:
                        # Build speaker config with user voices
                        voice_mappings = []
                        for lang in languages.split(','):
                            lang = lang.strip()
                            if lang in user_voices:
                                voice_mappings.append(f"{lang}:{user_voices[lang]}")
                                st.success(f"✅ Using your voice for {lang.upper()}")
                            else:
                                st.warning(f"⚠️ No voice sample found for {lang.upper()}, using default")
                        
                        if voice_mappings:
                            speaker_config = ",".join(voice_mappings)
                    else:
                        st.warning("⚠️ No user voice samples found, using default voices")
                        
                elif speaker_voice:
                    # Use uploaded speaker voice file
                    temp_voice_path = project_path / "temp_speaker_voice.wav"
                    with open(temp_voice_path, "wb") as f:
                        f.write(speaker_voice.getvalue())
                    speaker_config = f"en:{temp_voice_path}"
                
                tts_module.run(project_name, provider=tts_provider, 
                              languages=languages, speaker=speaker_config)
            
            # Stage 4: Render (Video Rendering)
            if render and (not skip_existing or not (project_path / "output" / f"{project_name}.mp4").exists()):
                quality_msg = "full quality" if full_quality_render else "web-friendly"
                st.info(f"🔄 Stage 4: Rendering final documentary ({quality_msg})...")
                render.run(project_name, full_quality=full_quality_render)
            
            # Check for final video
            final_video_paths = [
                project_path / "output" / f"{project_name}.mp4",
                project_path / "output" / "final.mp4"
            ]
            
            final_video_path = None
            for video_path in final_video_paths:
                if video_path.exists():
                    final_video_path = str(video_path)
                    break
            
            if final_video_path:
                # Run quality validation if available
                quality_report = "Processing complete!"
                if quality:
                    try:
                        st.info("🔄 Running quality validation...")
                        quality_data = quality.validate_project(project_name, languages=languages)
                        score_out_of_5 = quality_data.overall_score * 5
                        quality_report = f"Quality Score: {score_out_of_5:.1f}/5.0"
                    except Exception as e:
                        quality_report = f"Quality validation failed: {str(e)}"
                
                return "✅ Documentary processing complete!", final_video_path, quality_report
            else:
                return "⚠️ Pipeline completed but final video not found", None, "Check output directory"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            # Log to console for debugging
            print(f"[ERROR] Pipeline error:\n{error_details}")
            return f"❌ Pipeline failed during processing: {str(e)}", None, f"Error: {str(e)}\n\nFull traceback:\n{error_details}"
            
    except Exception as e:
        error_msg = f"❌ Pipeline setup failed: {str(e)}"
        return error_msg, None, f"Error: {str(e)}"

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 AI Documentary Composer</h1>
        <p>Transform your travel video clips into a fully-narrated mini-documentary using AI!</p>
        <p><strong>Pipeline:</strong> Vision Captioning → Language Planning → Text-to-Speech → Video Rendering</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("🎯 Navigation")
        page = st.radio(
            "Choose a page:",
            ["📝 Configuration & Processing", "📊 Results & Analysis"],
            index=0
        )
        
        st.header("🔧 Quick Actions")
        if st.button("🔄 Refresh Projects"):
            st.rerun()
        
        # Show existing projects
        existing_projects = get_existing_projects()
        if existing_projects:
            st.header("📁 Existing Projects")
            for project in existing_projects:
                if st.button(f"📂 {project}", key=f"sidebar_project_{project}"):
                    st.session_state['selected_project'] = project
                    st.rerun()
            
            # Show count of projects
            st.caption(f"{len(existing_projects)} projects found")
    
    # Main content area
    if page == "📝 Configuration & Processing":
        st.header("📝 Project Configuration & Processing")
        
        # Project selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📁 Project Setup")

            # Get existing projects
            existing_projects = get_existing_projects()

            # Auto-detect project mode based on selection
            if existing_projects:
                # Project selector with "New Project" option
                project_options = ["[New Project]"] + existing_projects
                default_index = 0
                if 'selected_project' in st.session_state and st.session_state['selected_project'] in project_options:
                    default_index = project_options.index(st.session_state['selected_project'])

                selected_option = st.selectbox(
                    "Select Project",
                    project_options,
                    index=default_index,
                    key="project_select"
                )

                use_existing = (selected_option != "[New Project]")

                if use_existing:
                    project_name = selected_option
                    st.session_state['selected_project'] = selected_option
                else:
                    project_name = ""
            else:
                use_existing = False
                project_name = ""

            # Pipeline Management Buttons (moved to top)
            if use_existing and project_name:
                col_reset, col_clean = st.columns(2)
                with col_reset:
                    if st.button("🔄 Reset Plan", help="Delete plan and all downstream outputs"):
                        if reset_plan_pipeline(project_name):
                            st.success("✅ Plan and outputs reset!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to reset pipeline")
                with col_clean:
                    if st.button("🧹 Clean TTS", help="Keep plan but regenerate audio"):
                        if clean_tts_only(project_name):
                            st.success("✅ TTS outputs cleaned!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to clean TTS")

            if not use_existing:
                # New project setup
                st.markdown("""
                <div class="upload-zone">
                    <h4>📤 Upload Video Clips</h4>
                </div>
                """, unsafe_allow_html=True)
                
                video_files = st.file_uploader(
                    "Choose video files",
                    type=['mp4', 'mov', 'avi'],
                    accept_multiple_files=True,
                    key="video_upload"
                )
                
                project_name = st.text_input(
                    "New Project Name",
                    value="demo_project",
                    placeholder="e.g., switzerland_trip"
                )
                
                # Save uploaded files immediately to create project directory
                if project_name and video_files:
                    project_path = Path("projects") / project_name
                    input_path = project_path / "input"
                    input_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save video files if they don't exist
                    for video_file in video_files:
                        file_path = input_path / video_file.name
                        if not file_path.exists():
                            with open(file_path, "wb") as f:
                                f.write(video_file.getvalue())
                    
                    # Mark project as updated to trigger UI refresh
                    st.success(f"📁 Project '{project_name}' created with {len(video_files)} video files")
                
                # Style selection
                if DocumentaryStyles:
                    style_options = DocumentaryStyles.list_styles()
                    selected_style = st.selectbox(
                        "📝 Documentary Style",
                        options=list(style_options.keys()),
                        format_func=lambda x: style_options[x],
                        index=0,
                        help="Choose the narrative style for your documentary"
                    )
                else:
                    selected_style = "documentary"
                    st.warning("Style system not available")
                
                # Project context input with STT
                st.markdown("📖 **Project Context (Optional)**")

                # Use session state to get transcribed text
                initial_context = st.session_state.get('project_context_value', '')
                project_context = st.text_area(
                    "Project Context",
                    value=initial_context,
                    placeholder="Describe your trip, the occasion, or any background information that should influence the narrative...",
                    help="This context will help the AI create a more personalized narrative",
                    height=100,
                    label_visibility="collapsed"
                )

                # Audio recording section below text area
                st.markdown("**🎙️ Voice Recording**")
                project_audio = st.audio_input("Record context", key="project_audio", label_visibility="collapsed")

                if project_audio is not None:
                    col_transcribe, col_voice = st.columns(2)
                    with col_transcribe:
                        if st.button("🎯 Transcribe to Text", key="transcribe_project",
                                   help="Convert speech to text and add to context above", use_container_width=True):
                            with st.spinner("🎙️ Transcribing..."):
                                try:
                                    audio_bytes = project_audio.getvalue()
                                    transcribed = transcribe_audio_data(audio_bytes)
                                    if transcribed:
                                        st.session_state['project_context_value'] = transcribed
                                        st.success("✅ Added to text box above!")
                                        st.rerun()
                                    else:
                                        st.error("❌ No text transcribed")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")

                    with col_voice:
                        if st.button("🎤 Save as Voice Sample", key="save_project_voice",
                                   help="Save this recording as your voice sample for TTS", use_container_width=True):
                            with st.spinner("💾 Saving voice sample..."):
                                try:
                                    audio_bytes = project_audio.getvalue()
                                    # Just save voice, don't transcribe
                                    transcribe_audio_data(audio_bytes, "en", project_name, save_voice=True)
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                
                # Clip-level context input
                clip_contexts = {}
                if video_files:
                    with st.expander("📝 Per-Clip Context (Optional)", expanded=False):
                        st.info("Add personal context for individual clips to make the narrative more meaningful")
                        for i, video_file in enumerate(video_files):
                            if video_file:
                                filename = video_file.name
                                st.markdown(f"**📹 {filename}**")
                                
                                # Create layout with thumbnail (bigger thumbnail column)
                                thumb_col, content_col = st.columns([2, 3])
                                
                                with thumb_col:
                                    # Show video thumbnail
                                    try:
                                        # Check if video is already saved to project
                                        if project_name:
                                            saved_video_path = Path("projects") / project_name / "input" / filename
                                            if saved_video_path.exists():
                                                thumbnail = extract_video_thumbnail(saved_video_path)
                                                if thumbnail:
                                                    st.image(thumbnail, caption="Preview", width=280)
                                                else:
                                                    st.info("📹")
                                            else:
                                                # Create temporary file to extract thumbnail
                                                with tempfile.NamedTemporaryFile(suffix=f".{filename.split('.')[-1]}", delete=False) as tmp_file:
                                                    tmp_file.write(video_file.getvalue())
                                                    tmp_path = Path(tmp_file.name)
                                                
                                                thumbnail = extract_video_thumbnail(tmp_path)
                                                if thumbnail:
                                                    st.image(thumbnail, caption="Preview", width=280)
                                                else:
                                                    st.info("📹")
                                                
                                                # Clean up
                                                tmp_path.unlink(missing_ok=True)
                                        else:
                                            st.info("📹")  # No project name yet
                                    except Exception as e:
                                        st.warning(f"Thumbnail error: {e}")
                                        st.info("📹")  # Fallback icon
                                
                                with content_col:
                                    clip_col1, clip_col2 = st.columns([3, 1])
                                    with clip_col1:
                                        # Use session state to maintain transcribed value
                                        initial_value = st.session_state.get(f'clip_context_value_{i}', '')
                                        clip_context = st.text_area(
                                            f"Context for {filename}",
                                            value=initial_value,
                                            placeholder="e.g., 'This is when we arrived at the hotel after a long journey...'",
                                            key=f"clip_context_{i}",
                                            height=60,
                                            label_visibility="collapsed"
                                        )
                                    
                                    with clip_col2:
                                        st.markdown(f"**🎙️**")
                                        clip_audio = st.audio_input(f"Record", key=f"clip_audio_{i}")
                                        
                                        if clip_audio is not None:
                                            clip_trans_col, clip_voice_col = st.columns(2)
                                            with clip_trans_col:
                                                if st.button("🎯 Transcribe", key=f"transcribe_clip_{i}", 
                                                           help="Convert speech to text", use_container_width=True):
                                                    with st.spinner("🎙️ Transcribing..."):
                                                        try:
                                                            audio_bytes = clip_audio.getvalue()
                                                            transcribed = transcribe_audio_data(audio_bytes)
                                                            if transcribed:
                                                                st.session_state[f'clip_context_value_{i}'] = transcribed
                                                                st.success("✅ Added to text box!")
                                                                st.rerun()
                                                            else:
                                                                st.error("❌ No text transcribed")
                                                        except Exception as e:
                                                            st.error(f"❌ Error: {e}")
                                            
                                            with clip_voice_col:
                                                if st.button("🎤", key=f"save_clip_voice_{i}",
                                                           help="Save as voice sample", use_container_width=True):
                                                    with st.spinner("💾 Saving..."):
                                                        try:
                                                            audio_bytes = clip_audio.getvalue()
                                                            transcribe_audio_data(audio_bytes, "en", project_name, save_voice=True)
                                                        except Exception as e:
                                                            st.error(f"❌ Error: {e}")
                                
                                if clip_context.strip():
                                    clip_contexts[filename] = clip_context
        
        with col2:
            if project_name:
                st.subheader("📊 Project Status")
                stages = get_project_stage_status(project_name)
                completed_stages = sum(1 for stage_name, stage_info in stages.items() 
                                     if stage_name != "input_files" and stage_info.get("completed", False))
                
                st.info(f"**{project_name}**\n\n{len(stages.get('input_files', []))} videos | {completed_stages}/4 stages completed")
        
        # Video preprocessing options
        st.subheader("📹 Video Preprocessing")
        col1, col2 = st.columns(2)
        with col1:
            auto_preprocess = st.checkbox("Auto-preprocess large files", value=True, 
                                        help="Automatically compress files >20MB for API compatibility")
        with col2:
            full_quality_render = st.checkbox("Render full quality", value=False,
                                            help="Use original high-resolution files for final render")
        
        # AI Configuration
        st.subheader("⚙️ AI Model Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            caption_provider = st.selectbox("Vision Captioning", ["gemini", "blip"], index=0)
        with col2:
            plan_provider = st.selectbox("Language Planning", ["gemini", "ollama"], index=0)
        with col3:
            tts_provider = st.selectbox("Text-to-Speech", ["xtts", "gemini", "vibevoice"], index=0)

        # Show Gemini quota status if using Gemini
        if plan_provider == "gemini" or caption_provider == "gemini" or tts_provider == "gemini":
            try:
                from ai_doc_composer.quota_manager import QuotaManager
                quota_mgr = QuotaManager()
                with st.expander("📊 Gemini API Quota Status", expanded=False):
                    st.markdown(quota_mgr.get_usage_summary())

                    # Add model selection for Gemini
                    if plan_provider == "gemini":
                        st.markdown("##### Preferred Gemini Model")
                        col1, col2 = st.columns(2)
                        with col1:
                            gemini_models = ["Auto (Best Available)", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-pro"]
                            preferred_model = st.selectbox("Model", gemini_models, index=0,
                                                          help="Select preferred model or let system choose based on quota")
                        with col2:
                            if st.button("🔄 Reset Quota Cache", help="Reset quota tracking (for testing only)"):
                                quota_mgr._reset_daily()
                                st.rerun()
            except ImportError:
                pass
        
        # Ollama model selection (shown only when ollama is selected)
        ollama_model = None
        if plan_provider == "ollama":
            st.markdown("##### Ollama Model")
            col1, col2 = st.columns(2)
            with col1:
                model_options = [
                    "llama3.3:latest",
                    "gpt-oss:20b",
                    "llama3.1:8b",
                    "llama3.2:3b",
                    "qwen2.5:72b",
                    "custom"
                ]
                selected_model = st.selectbox("Select Model", model_options, index=0)
            with col2:
                if selected_model == "custom":
                    ollama_model = st.text_input("Custom Model Name", placeholder="e.g., mixtral:8x7b")
                else:
                    ollama_model = selected_model
                    st.text_input("Selected Model", value=ollama_model, disabled=True)

        col1, col2 = st.columns(2)
        with col1:
            languages = st.text_input("Languages (BCP-47 codes)", value="en", placeholder="e.g., en,es,fr")
        with col2:
            temperature = st.slider("Creativity (Temperature)", 0.05, 0.5, 0.15, 0.05)  # Lowered for duration compliance

        # Documentary configuration (for existing projects)
        if use_existing and project_name and ContextManager:
            project_path = Path("projects") / project_name
            context_manager = ContextManager(project_path)
            proj_context = context_manager.load_project_context()
            
            st.subheader("📝 Style & Context Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                if DocumentaryStyles:
                    style_options = DocumentaryStyles.list_styles()
                    current_style_index = list(style_options.keys()).index(proj_context.style) if proj_context.style in style_options else 0
                    selected_style = st.selectbox(
                        "📝 Documentary Style",
                        options=list(style_options.keys()),
                        format_func=lambda x: style_options[x],
                        index=current_style_index,
                        help="Choose the narrative style for your documentary"
                    )
                else:
                    selected_style = "documentary"
                    st.warning("Style system not available")
            
            with col2:
                ordering_strategy = st.selectbox(
                    "🔄 Clip Ordering Strategy",
                    options=["llm", "exif"],
                    index=0 if proj_context.ordering_strategy == "llm" else 1,
                    format_func=lambda x: "AI-Based Ordering" if x == "llm" else "Chronological (EXIF)",
                    help="Choose how clips should be ordered"
                )
            
            # Project context with STT for existing projects
            st.markdown("📖 **Project Context (Optional)**")

            # Use session state to maintain transcribed value
            initial_context = st.session_state.get('existing_project_context_value', proj_context.project_context)
            project_context = st.text_area(
                "Project Context",
                value=initial_context,
                placeholder="Describe your trip, the occasion, or any background information...",
                help="This context will help the AI create a more personalized narrative",
                height=100,
                label_visibility="collapsed"
            )

            # Audio recording section below text area
            st.markdown("**🎙️ Voice Recording**")
            existing_project_audio = st.audio_input("Record context", key="existing_project_audio", label_visibility="collapsed")

            if existing_project_audio is not None:
                col_ex_transcribe, col_ex_voice = st.columns(2)
                with col_ex_transcribe:
                    if st.button("🎯 Transcribe to Text", key="transcribe_existing_project",
                               help="Convert speech to text and add to context above", use_container_width=True):
                        with st.spinner("🎙️ Transcribing..."):
                            try:
                                audio_bytes = existing_project_audio.getvalue()
                                transcribed = transcribe_audio_data(audio_bytes)
                                if transcribed:
                                    st.session_state['existing_project_context_value'] = transcribed
                                    st.success("✅ Added to text box above!")
                                    st.rerun()
                                else:
                                    st.error("❌ No text transcribed")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")

                with col_ex_voice:
                    if st.button("🎤 Save as Voice Sample", key="save_existing_voice",
                               help="Save this recording as your voice sample for TTS", use_container_width=True):
                        with st.spinner("💾 Saving voice sample..."):
                            try:
                                audio_bytes = existing_project_audio.getvalue()
                                transcribe_audio_data(audio_bytes, "en", project_name, save_voice=True)
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
            
            # Load existing clip contexts
            clip_contexts_data = context_manager.load_clip_contexts()
            clip_contexts = {}
            
            if clip_contexts_data:
                with st.expander("📝 Per-Clip Context (Optional)", expanded=False):
                    for idx, (filename, context_obj) in enumerate(clip_contexts_data.items()):
                        st.markdown(f"**📹 {filename}**")
                        
                        # Create layout with thumbnail for existing clips (bigger thumbnail column)
                        existing_thumb_col, existing_content_col = st.columns([2, 3])
                        
                        with existing_thumb_col:
                            # Show video thumbnail for existing project
                            try:
                                # Use smart file finder to locate video
                                video_path = find_video_file(project_name, filename)
                                
                                if video_path:
                                    thumbnail = extract_video_thumbnail(video_path)
                                    if thumbnail:
                                        st.image(thumbnail, caption="Preview", width=280)
                                    else:
                                        st.info("📹")
                                else:
                                    st.info("📹")  # File not found
                            except Exception as e:
                                st.info("📹")  # Fallback icon
                        
                        with existing_content_col:
                            existing_col1, existing_col2 = st.columns([3, 1])
                            with existing_col1:
                                # Use session state to maintain transcribed value
                                initial_value = st.session_state.get(f'existing_clip_context_value_{idx}', context_obj.personal_context)
                                clip_context = st.text_area(
                                    f"Context for {filename}",
                                    value=initial_value,
                                    placeholder="e.g., 'This is when we arrived at the hotel...'",
                                    key=f"existing_clip_context_{filename}",
                                    height=60,
                                    label_visibility="collapsed"
                                )
                            
                            with existing_col2:
                                st.markdown(f"**🎙️**")
                                existing_audio = st.audio_input(f"Record", key=f"existing_clip_audio_{idx}")
                                
                                if existing_audio is not None:
                                    ex_clip_trans_col, ex_clip_voice_col = st.columns(2)
                                    with ex_clip_trans_col:
                                        if st.button("🎯 Transcribe", key=f"transcribe_existing_clip_{idx}", 
                                                   help="Convert speech to text", use_container_width=True):
                                            with st.spinner("🎙️ Transcribing..."):
                                                try:
                                                    audio_bytes = existing_audio.getvalue()
                                                    transcribed = transcribe_audio_data(audio_bytes)
                                                    if transcribed:
                                                        st.session_state[f'existing_clip_context_value_{idx}'] = transcribed
                                                        st.success("✅ Added to text box!")
                                                        st.rerun()
                                                    else:
                                                        st.error("❌ No text transcribed")
                                                except Exception as e:
                                                    st.error(f"❌ Error: {e}")
                                    
                                    with ex_clip_voice_col:
                                        if st.button("🎤", key=f"save_existing_clip_voice_{idx}",
                                                   help="Save as voice sample", use_container_width=True):
                                            with st.spinner("💾 Saving..."):
                                                try:
                                                    audio_bytes = existing_audio.getvalue()
                                                    transcribe_audio_data(audio_bytes, "en", project_name, save_voice=True)
                                                except Exception as e:
                                                    st.error(f"❌ Error: {e}")
                        
                        clip_contexts[filename] = clip_context
        elif not use_existing:
            # For new projects, use the values set above
            ordering_strategy = "llm"
            pass
        else:
            # Fallback for when context manager is not available
            selected_style = "documentary"
            project_context = ""
            clip_contexts = {}
            ordering_strategy = "llm"
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            skip_existing = st.checkbox("Skip Completed Stages", value=True,
                                       help="Skip stages that already have output files")

            allow_overflow = st.checkbox("Allow Audio Overflow", value=True,
                                        help="Allow narration to flow between clips for smoother transitions")

            # Voice options
            st.markdown("**🎤 Voice Options**")
            use_user_voice = st.checkbox("Use My Voice", value=False, 
                                        help="Use your recorded voice samples for TTS narration")
            
            if not use_user_voice:
                speaker_voice = st.file_uploader("Speaker Voice Sample (Optional WAV)", type=['wav'])
            else:
                speaker_voice = None
                # Show available voice samples
                if project_name:
                    try:
                        project_path = Path("projects") / project_name
                        if project_path.exists():
                            user_voices = get_user_voice_samples_safe(project_path)
                            if user_voices:
                                st.success(f"✅ Found voice samples: {', '.join(user_voices.keys())}")
                                for lang, voice_path in user_voices.items():
                                    with st.expander(f"🎤 {lang.upper()} Voice Sample"):
                                        st.audio(str(voice_path))
                            else:
                                st.warning("⚠️ No voice samples found. Record some context audio and click '🎤 Save Voice' to create voice samples.")
                        else:
                            st.info("ℹ️ Project will be created first, then voice samples can be used.")
                    except ImportError:
                        st.error("STT module not available")
                else:
                    st.info("✅ Will use your recorded voice samples from context recordings")
        
        # Process button
        if st.button("🚀 Create Documentary", type="primary", use_container_width=True):
            if not project_name.strip():
                st.error("❌ Project name is required")
            elif not use_existing and not video_files:
                st.error("❌ Please upload video files for new projects")
            else:
                # Check if modules are available (with helpful message for Docker)
                if not ingest or not plan or not render:
                    st.error("❌ Required modules not available.")
                    st.info("💡 **Docker Mode**: Use cloud providers (Gemini) which don't require local ML libraries. For local models (BLIP-2, XTTS), run with `poetry` instead of Docker.")
                    return
                
                # Show processing progress
                progress_container = st.container()
                with progress_container:
                    st.info("🚀 Starting AI Documentary Pipeline...")
                    
                    # Create progress tracking elements
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process pipeline with real-time updates
                    status, video_path, quality_report = process_video_pipeline(
                        video_files if not use_existing else [],
                        project_name,
                        caption_provider,
                        plan_provider,
                        tts_provider,
                        languages,
                        temperature,
                        speaker_voice,
                        skip_existing,
                        auto_preprocess,
                        full_quality_render,
                        selected_style,
                        project_context,
                        clip_contexts,
                        ordering_strategy,
                        use_user_voice,
                        ollama_model if plan_provider == "ollama" else None,
                        allow_overflow
                    )
                    
                    # Update progress to completion
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    
                    # Show results
                    if "✅" in status:
                        st.success(status)
                        if quality_report:
                            st.info(f"📊 {quality_report}")
                        st.balloons()
                        st.info("🎬 **Switch to 'Results & Analysis' tab** to view your documentary!")
                        
                        # Auto-refresh the project status
                        if 'selected_project' not in st.session_state:
                            st.session_state['selected_project'] = project_name
                        
                    else:
                        st.error(status)
                        if quality_report and quality_report != "No processing performed":
                            # Show error details in expandable section
                            with st.expander("🔍 Error Details", expanded=True):
                                # Split the quality_report to separate short error from full traceback
                                if "Full traceback:" in quality_report:
                                    short_error, full_trace = quality_report.split("Full traceback:", 1)
                                    st.error(f"Error: {short_error.replace('Error: ', '').strip()}")
                                    st.code(full_trace.strip(), language="python")
                                else:
                                    st.error(quality_report)
        
        # Tips section
        st.subheader("💡 Processing Tips")
        st.info("""
        **⚙️ Configuration Tips:**
        - Use Gemini providers for faster cloud processing
        - Higher temperature = more creative narration
        """)
    
    elif page == "📊 Results & Analysis":
        st.header("📊 Project Results & Data Analysis")
        
        # Project selection for results
        existing_projects = get_existing_projects()
        if not existing_projects:
            st.warning("No projects found. Create a project first in the Configuration & Processing page.")
            return
        
        selected_project = st.selectbox(
            "🔍 Select Project to View",
            existing_projects,
            key="results_project_select"
        )
        
        if selected_project:
            # Display pipeline status
            stages = display_pipeline_status(selected_project)
            
            # Display metrics
            display_project_metrics(stages)
            
            # Final video section
            final_video_path = None
            if stages.get("render", {}).get("completed"):
                final_video_path = stages["render"]["data"]["video_path"]
            
            if final_video_path and os.path.exists(final_video_path):
                st.subheader("🎬 Final Documentary")
                st.video(final_video_path)
                
                # Download button
                col1, col2 = st.columns(2)
                with col1:
                    with open(final_video_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Final Video",
                            data=file.read(),
                            file_name=f"{selected_project}_documentary.mp4",
                            mime="video/mp4",
                            type="primary"
                        )

                # Full quality render button (only shows after initial render)
                with col2:
                    # Initialize session state for render status
                    if 'full_quality_rendering' not in st.session_state:
                        st.session_state.full_quality_rendering = False
                    if 'render_output' not in st.session_state:
                        st.session_state.render_output = None
                    if 'render_error' not in st.session_state:
                        st.session_state.render_error = None

                    if st.button("🎬 Render Full Quality", type="secondary", key="full_quality_render", disabled=st.session_state.full_quality_rendering):
                        st.session_state.full_quality_rendering = True
                        st.session_state.render_output = None
                        st.session_state.render_error = None
                        st.rerun()

                    # Handle the rendering process
                    if st.session_state.full_quality_rendering:
                        import subprocess
                        import sys

                        # Debug logging
                        with open('/tmp/render_debug.log', 'a') as f:
                            f.write(f"Starting full quality render at {datetime.now()}\n")

                        st.info("🎬 Starting full quality render...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            # Build command
                            cmd = [
                                sys.executable, "-m", "ai_doc_composer.cli",
                                "render-stage", selected_project,
                                "--full-quality", "--offset", "1.0"
                            ]

                            with open('/tmp/render_debug.log', 'a') as f:
                                f.write(f"Command: {' '.join(cmd)}\n")

                            status_text.text("Running render command...")
                            progress_bar.progress(25)

                            # Run render with full quality flag using CLI
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout

                            with open('/tmp/render_debug.log', 'a') as f:
                                f.write(f"Return code: {result.returncode}\n")
                                f.write(f"Stdout: {result.stdout}\n")
                                f.write(f"Stderr: {result.stderr}\n")

                            progress_bar.progress(75)

                            if result.returncode == 0:
                                st.session_state.render_output = result.stdout
                                st.session_state.full_quality_rendering = False
                                progress_bar.progress(100)
                                status_text.text("✅ Rendering complete!")
                                st.success("✅ Full quality video rendered successfully!")
                                if result.stdout:
                                    with st.expander("Command Output"):
                                        st.code(result.stdout)
                                time.sleep(2)  # Let user see success message
                                st.rerun()
                            else:
                                st.session_state.render_error = result.stderr or "Unknown error"
                                st.session_state.full_quality_rendering = False
                                st.error(f"❌ Render failed: {result.stderr}")
                                progress_bar.empty()
                                status_text.empty()

                        except subprocess.TimeoutExpired:
                            with open('/tmp/render_debug.log', 'a') as f:
                                f.write("Timeout expired\n")
                            st.session_state.render_error = "Rendering timed out after 10 minutes"
                            st.session_state.full_quality_rendering = False
                            st.error("⏱️ Rendering timed out. Please try running from command line.")
                            progress_bar.empty()
                            status_text.empty()
                        except Exception as ex:
                            with open('/tmp/render_debug.log', 'a') as f:
                                f.write(f"Exception: {str(ex)}\n")
                            st.session_state.render_error = str(ex)
                            st.session_state.full_quality_rendering = False
                            st.error(f"❌ Unexpected error: {str(ex)}")
                            progress_bar.empty()
                            status_text.empty()

                    # Show any previous error
                    elif st.session_state.render_error:
                        st.warning(f"Previous render attempt failed: {st.session_state.render_error}")
                
                # Quality report
                quality_file = Path("projects") / selected_project / "json" / "quality_report.json"
                if quality_file.exists():
                    try:
                        with open(quality_file) as f:
                            quality_data = json.load(f)
                        score_out_of_5 = quality_data.get('overall_score', 0) * 5
                        st.metric("Quality Score", f"{score_out_of_5:.1f}/5.0")
                    except Exception:
                        st.info("Quality report available but could not be parsed")
            else:
                st.info("🎬 Final documentary will appear here once processing is complete")
            
            # Video timeline
            display_video_timeline(selected_project, stages)
            
            # Detailed stage information
            st.subheader("🔍 Detailed Stage Analysis")
            display_stage_details(stages)
    
    # Footer
    st.markdown("---")
    st.markdown("*Built for CM3070 Computer Science Final Project | University of London*")

if __name__ == "__main__":
    main()