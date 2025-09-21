"""Speech-to-Text module for AI Documentary Composer.

Provides speech-to-text transcription capabilities for personal context input,
leveraging the existing Whisper infrastructure from the validation module.

Example CLI usage::

    python -m ai_doc_composer.cli record-context myproject clip1.mp4

Example programmatic usage::

    from ai_doc_composer.stt import STTTranscriber
    
    transcriber = STTTranscriber()
    text = transcriber.transcribe_audio("recording.wav", language="en")
"""

from __future__ import annotations

import tempfile
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple

import typer

# Optional dependencies for STT
try:
    import whisper
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False


class STTTranscriber:
    """Speech-to-Text transcriber using Whisper models."""
    
    def __init__(self, whisper_model: str = "base", use_faster_whisper: bool = True):
        """Initialize STT transcriber with Whisper model.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            use_faster_whisper: Whether to use faster-whisper (if available) instead of openai-whisper
        """
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        self.use_faster_whisper = use_faster_whisper and _FASTER_WHISPER_AVAILABLE
        
    def _ensure_whisper(self):
        """Lazy-load Whisper model."""
        if not _WHISPER_AVAILABLE and not _FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "Neither openai-whisper nor faster-whisper is available. "
                "Install with: pip install openai-whisper"
            )
            
        if self.whisper_model is None:
            if self.use_faster_whisper:
                print(f"Loading faster-whisper model: {self.whisper_model_name}")
                self.whisper_model = WhisperModel(
                    self.whisper_model_name, 
                    device="cpu", 
                    compute_type="int8"
                )
            else:
                if not _WHISPER_AVAILABLE:
                    raise ImportError(
                        "openai-whisper not available, but faster-whisper is. "
                        "Set use_faster_whisper=True"
                    )
                print(f"Loading openai-whisper model: {self.whisper_model_name}")
                self.whisper_model = whisper.load_model(self.whisper_model_name)
    
    def transcribe_audio(self, audio_path: Path | str, language: str = "en") -> Tuple[str, float]:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Language code (en, es, fr, etc.)
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
            
        Raises:
            ImportError: If Whisper models are not available
            FileNotFoundError: If audio file doesn't exist
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self._ensure_whisper()
        
        print(f"Transcribing audio: {audio_path.name}")
        start_time = time.time()
        
        if self.use_faster_whisper:
            segments, info = self.whisper_model.transcribe(
                str(audio_path), 
                language=language, 
                beam_size=5
            )
            transcribed_text = " ".join(segment.text for segment in segments).strip()
            # faster-whisper doesn't provide confidence, use detection probability
            confidence = getattr(info, 'language_probability', 0.95)
        else:
            result = self.whisper_model.transcribe(str(audio_path), language=language)
            transcribed_text = result["text"].strip()
            # openai-whisper doesn't provide confidence either, estimate from length
            confidence = min(0.95, len(transcribed_text) / 100)
        
        duration = time.time() - start_time
        print(f"Transcription completed in {duration:.2f}s")
        
        return transcribed_text, confidence
    
    def transcribe_audio_simple(self, audio_path: Path | str, language: str = "en") -> str:
        """Simple transcription that returns only the text.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Transcribed text string
        """
        text, _ = self.transcribe_audio(audio_path, language)
        return text


def create_temp_audio_file(audio_data: bytes, suffix: str = ".wav") -> Path:
    """Create a temporary audio file from audio data.
    
    Args:
        audio_data: Raw audio bytes
        suffix: File extension (e.g., ".wav", ".mp3")
        
    Returns:
        Path to temporary audio file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(audio_data)
    temp_file.close()
    return Path(temp_file.name)


def transcribe_for_context(
    audio_path: Path | str, 
    language: str = "en",
    whisper_model: str = "base"
) -> str:
    """High-level function to transcribe audio for personal context input.
    
    Args:
        audio_path: Path to audio file
        language: Language code
        whisper_model: Whisper model size
        
    Returns:
        Transcribed text suitable for personal context
        
    Example:
        >>> text = transcribe_for_context("my_recording.wav", "en")
        >>> print(f"Personal context: {text}")
    """
    transcriber = STTTranscriber(whisper_model=whisper_model)
    text = transcriber.transcribe_audio_simple(audio_path, language)
    
    # Clean up the text for context use
    text = text.strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text


def save_voice_sample(
    audio_data: bytes, 
    project_path: Path, 
    language: str = "en",
    sample_type: str = "user_voice"
) -> Path:
    """Save user's voice recording as a TTS voice sample.
    
    Args:
        audio_data: Raw audio bytes from recording
        project_path: Path to project directory
        language: Language code for the voice sample
        sample_type: Type of sample ("user_voice", "project_voice")
        
    Returns:
        Path to saved voice sample file
    """
    voices_dir = project_path / "voices"
    voices_dir.mkdir(exist_ok=True)
    
    # Create filename based on type and language
    voice_filename = f"{sample_type}_{language}.wav"
    voice_path = voices_dir / voice_filename
    
    # Save audio data to file
    with open(voice_path, 'wb') as f:
        f.write(audio_data)
    
    print(f"✅ Voice sample saved: {voice_path}")
    return voice_path


def get_user_voice_samples(project_path: Path) -> dict[str, Path]:
    """Get available user voice samples for a project.
    
    Args:
        project_path: Path to project directory
        
    Returns:
        Dictionary mapping language codes to voice sample paths
    """
    voices_dir = project_path / "voices"
    if not voices_dir.exists():
        return {}
    
    voice_samples = {}
    for voice_file in voices_dir.glob("user_voice_*.wav"):
        # Extract language from filename: user_voice_en.wav -> en
        parts = voice_file.stem.split('_')
        if len(parts) >= 3:
            language = parts[2]
            voice_samples[language] = voice_file
    
    return voice_samples


def copy_voice_sample_to_project(source_audio: Path, project_path: Path, language: str = "en") -> Path:
    """Copy an existing audio file as a voice sample for the project.
    
    Args:
        source_audio: Path to existing audio file
        project_path: Path to project directory
        language: Language code for the voice sample
        
    Returns:
        Path to copied voice sample
    """
    if not source_audio.exists():
        raise FileNotFoundError(f"Source audio file not found: {source_audio}")
    
    voices_dir = project_path / "voices"
    voices_dir.mkdir(exist_ok=True)
    
    voice_filename = f"user_voice_{language}.wav"
    voice_path = voices_dir / voice_filename
    
    # Copy the file
    shutil.copy2(source_audio, voice_path)
    
    print(f"✅ Voice sample copied: {voice_path}")
    return voice_path


# CLI interface for STT functionality
def main():
    """CLI entry point for STT transcription."""
    typer.run(transcribe_cli)


def transcribe_cli(
    audio_path: str = typer.Argument(..., help="Path to audio file to transcribe"),
    language: str = typer.Option("en", "--language", "-l", help="Language code (en, es, fr, etc.)"),
    whisper_model: str = typer.Option("base", "--model", "-m", help="Whisper model size"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save transcription to file"),
):
    """Transcribe audio file to text using Whisper."""
    
    try:
        text = transcribe_for_context(audio_path, language, whisper_model)
        
        print(f"\n📝 Transcription Result:")
        print(f"Language: {language}")
        print(f"Model: {whisper_model}")
        print(f"Text: {text}")
        
        if output_file:
            Path(output_file).write_text(text, encoding="utf-8")
            print(f"💾 Saved to: {output_file}")
            
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()