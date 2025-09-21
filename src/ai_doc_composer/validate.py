"""Quality validation module for AI Documentary Composer.

Implements automated quality assurance across the multi-modal pipeline:
- Whisper-based WER (Word Error Rate) validation for TTS quality
- Audio-text synchronization validation 
- Performance benchmarking utilities
- Quality scoring and reporting

Example CLI usage::

    python -m ai_doc_composer.cli validate-stage switzerland --languages en,es
"""

from __future__ import annotations

import json
import time
import psutil
import statistics
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "projects"
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import typer

# Optional dependencies for validation
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

try:
    import jiwer
    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False


@dataclass
class ValidationMetrics:
    """Container for validation results."""
    wer_scores: Dict[str, float]
    duration_accuracy: Dict[str, float] 
    processing_times: Dict[str, float]
    memory_usage: Dict[str, float]
    overall_quality_score: float
    detailed_errors: List[str]


class QualityValidator:
    """Multi-modal quality validation for documentary pipeline."""
    
    def __init__(self, whisper_model: str = "base", use_faster_whisper: bool = True):
        """Initialize validator with Whisper model.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            use_faster_whisper: Whether to use faster-whisper (if available) instead of openai-whisper
        """
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        self.use_faster_whisper = use_faster_whisper and _FASTER_WHISPER_AVAILABLE
        self.validation_cache = {}
        
    def _ensure_whisper(self):
        """Lazy-load Whisper model."""
        if not _WHISPER_AVAILABLE and not _FASTER_WHISPER_AVAILABLE:
            raise ImportError("whisper package required for WER validation. Install with: pip install openai-whisper or pip install faster-whisper")
            
        if self.whisper_model is None:
            if self.use_faster_whisper:
                print(f"Loading faster-whisper model: {self.whisper_model_name}")
                self.whisper_model = WhisperModel(self.whisper_model_name, device="cpu", compute_type="int8")
            else:
                if not _WHISPER_AVAILABLE:
                    raise ImportError("openai-whisper not available, but faster-whisper is. Set use_faster_whisper=True")
                print(f"Loading openai-whisper model: {self.whisper_model_name}")
                self.whisper_model = whisper.load_model(self.whisper_model_name)
            
    def calculate_wer(self, audio_path: Path, reference_text: str, language: str = "en") -> Tuple[float, str]:
        """Calculate Word Error Rate between audio and reference text.
        
        Args:
            audio_path: Path to generated audio file
            reference_text: Original script text
            language: Language code for Whisper
            
        Returns:
            Tuple of (WER score, transcribed text)
        """
        if not _JIWER_AVAILABLE:
            typer.echo("Warning: jiwer not available, using basic word comparison", err=True)
            return self._basic_wer_approximation(audio_path, reference_text)
            
        self._ensure_whisper()
        
        # Cache key for avoiding re-transcription
        whisper_type = "faster" if self.use_faster_whisper else "openai"
        cache_key = f"{audio_path.name}_{language}_{self.whisper_model_name}_{whisper_type}"
        if cache_key in self.validation_cache:
            transcribed_text = self.validation_cache[cache_key]
        else:
            # Transcribe audio with Whisper
            if self.use_faster_whisper:
                segments, info = self.whisper_model.transcribe(str(audio_path), language=language, beam_size=5)
                transcribed_text = " ".join(segment.text for segment in segments).strip()
            else:
                result = self.whisper_model.transcribe(str(audio_path), language=language)
                transcribed_text = result["text"].strip()
            self.validation_cache[cache_key] = transcribed_text
        
        # Calculate WER using jiwer
        try:
            wer = jiwer.wer(reference_text, transcribed_text)
        except Exception as e:
            typer.echo(f"WER calculation failed: {e}", err=True)
            wer = 1.0  # Worst case
            
        return wer, transcribed_text
    
    def _basic_wer_approximation(self, audio_path: Path, reference_text: str) -> Tuple[float, str]:
        """Fallback WER approximation without jiwer."""
        self._ensure_whisper()
        
        if self.use_faster_whisper:
            segments, info = self.whisper_model.transcribe(str(audio_path), beam_size=5)
            transcribed_text = " ".join(segment.text for segment in segments).strip()
        else:
            result = self.whisper_model.transcribe(str(audio_path))
            transcribed_text = result["text"].strip()
        
        # Simple word-level comparison
        ref_words = reference_text.lower().split()
        trans_words = transcribed_text.lower().split()
        
        if not ref_words:
            return 1.0, transcribed_text
            
        # Basic edit distance approximation
        matches = sum(1 for w in trans_words if w in ref_words)
        wer_approx = 1.0 - (matches / len(ref_words))
        
        return wer_approx, transcribed_text
    
    def validate_duration_accuracy(self, audio_path: Path, expected_duration: float, tolerance: float = 0.1) -> float:
        """Validate audio duration matches expected length.
        
        Args:
            audio_path: Path to audio file
            expected_duration: Expected duration in seconds
            tolerance: Acceptable deviation (default 10%)
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        try:
            import soundfile as sf
            data, samplerate = sf.read(str(audio_path))
            actual_duration = len(data) / samplerate
        except ImportError:
            # Fallback using ffprobe if soundfile not available
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', str(audio_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return 0.0
                
            actual_duration = float(result.stdout.strip())
        
        deviation = abs(actual_duration - expected_duration) / expected_duration
        accuracy = max(0.0, 1.0 - (deviation / tolerance))
        
        return accuracy
    
    def benchmark_performance(self, project_path: Path) -> Dict[str, float]:
        """Benchmark pipeline performance metrics.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Dictionary of performance metrics
        """
        json_dir = project_path / "json"
        output_dir = project_path / "output"
        
        metrics = {
            "total_clips": 0,
            "total_audio_files": 0,
            "avg_processing_time_per_clip": 0.0,
            "memory_peak_mb": 0.0,
            "disk_usage_mb": 0.0
        }
        
        # Count processed files
        if json_dir.exists():
            captions_file = json_dir / "captions.json"
            if captions_file.exists():
                with open(captions_file) as f:
                    captions_data = json.load(f)
                    metrics["total_clips"] = len(captions_data.get("clips", []))
        
        if output_dir.exists():
            audio_dir = output_dir / "audio"
            if audio_dir.exists():
                metrics["total_audio_files"] = len(list(audio_dir.glob("*.wav")))
        
        # Estimate disk usage
        if output_dir.exists():
            total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
            metrics["disk_usage_mb"] = total_size / (1024 * 1024)
        
        # Get current memory usage
        process = psutil.Process()
        metrics["memory_peak_mb"] = process.memory_info().rss / (1024 * 1024)
        
        return metrics


def run(project: str, languages: str = "en", whisper_model: str = "base", 
        output_report: bool = True, use_faster_whisper: bool = True) -> ValidationMetrics:
    """Validate quality across the documentary pipeline.
    
    Args:
        project: Project name (directory under projects/)
        languages: Comma-separated language codes
        whisper_model: Whisper model size for transcription
        output_report: Whether to save validation report
        use_faster_whisper: Whether to use faster-whisper instead of openai-whisper
        
    Returns:
        ValidationMetrics object with results
    """
    project_path = DATA_ROOT / project
    if not project_path.exists():
        raise ValueError(f"Project directory not found: {project_path}")
    
    validator = QualityValidator(whisper_model, use_faster_whisper)
    lang_list = [lang.strip() for lang in languages.split(",")]
    
    # Load pipeline artifacts
    json_dir = project_path / "json"
    plan_file = json_dir / "plan.json"
    tts_meta_file = json_dir / "tts_meta.json"
    
    if not plan_file.exists():
        raise ValueError(f"plan.json not found in {json_dir}")
    if not tts_meta_file.exists():
        raise ValueError(f"tts_meta.json not found in {json_dir}")
    
    with open(plan_file) as f:
        plan_data = json.load(f)
    with open(tts_meta_file) as f:
        tts_data = json.load(f)
    
    print(f"🔍 Validating quality for project: {project}")
    print(f"📊 Languages: {languages}")
    print(f"🎤 Whisper model: {whisper_model}")
    
    # Initialize results
    wer_scores = {}
    duration_accuracy = {}
    processing_times = {}
    detailed_errors = []
    
    # Performance benchmarking
    start_time = time.time()
    performance_metrics = validator.benchmark_performance(project_path)
    
    # Validate each language
    for lang in lang_list:
        print(f"\n🔍 Validating {lang.upper()} audio quality...")
        
        lang_wer_scores = []
        lang_duration_scores = []
        
        # Get script lines for this language
        if lang == "en":
            script_lines = plan_data.get("script", [])
        else:
            # Check for translations
            translations = plan_data.get("translations", {})
            script_lines = translations.get(lang, [])
            
        if not script_lines:
            detailed_errors.append(f"No script found for language: {lang}")
            continue
        
        # Validate each audio clip
        audio_dir = project_path / "output" / "audio"
        for i, script_line in enumerate(script_lines):
            # Find corresponding audio file
            clip_id = plan_data["ordered_clips"][i]
            audio_file = audio_dir / f"{clip_id}_{lang}.wav"
            
            if not audio_file.exists():
                detailed_errors.append(f"Audio file missing: {audio_file}")
                continue
            
            try:
                # WER validation
                clip_start_time = time.time()
                wer, transcribed = validator.calculate_wer(audio_file, script_line, lang)
                lang_wer_scores.append(wer)
                
                # Duration validation
                expected_duration = len(script_line.split()) / 150 * 60  # Rough estimate
                duration_acc = validator.validate_duration_accuracy(audio_file, expected_duration)
                lang_duration_scores.append(duration_acc)
                
                clip_time = time.time() - clip_start_time
                processing_times[f"{clip_id}_{lang}"] = clip_time
                
                print(f"  ✓ {clip_id}_{lang}: WER={wer:.3f}, Duration={duration_acc:.3f}")
                
            except Exception as e:
                error_msg = f"Validation failed for {audio_file}: {e}"
                detailed_errors.append(error_msg)
                print(f"  ❌ {error_msg}")
        
        # Aggregate language scores
        if lang_wer_scores:
            wer_scores[lang] = statistics.mean(lang_wer_scores)
        if lang_duration_scores:
            duration_accuracy[lang] = statistics.mean(lang_duration_scores)
    
    # Calculate overall quality score
    all_wer = list(wer_scores.values())
    all_duration = list(duration_accuracy.values())
    
    if all_wer and all_duration:
        # Lower WER is better, higher duration accuracy is better
        avg_wer = statistics.mean(all_wer)
        avg_duration = statistics.mean(all_duration)
        overall_quality = (1.0 - avg_wer) * 0.7 + avg_duration * 0.3
    else:
        overall_quality = 0.0
    
    # Create validation metrics
    validation_time = time.time() - start_time
    memory_usage = {"validation_time_sec": validation_time, **performance_metrics}
    
    metrics = ValidationMetrics(
        wer_scores=wer_scores,
        duration_accuracy=duration_accuracy,
        processing_times=processing_times,
        memory_usage=memory_usage,
        overall_quality_score=overall_quality,
        detailed_errors=detailed_errors
    )
    
    # Output summary
    print(f"\n📊 VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall Quality Score: {overall_quality:.3f}")
    if wer_scores:
        avg_wer = statistics.mean(wer_scores.values())
        print(f"Average WER: {avg_wer:.3f}")
    if duration_accuracy:
        avg_dur = statistics.mean(duration_accuracy.values())
        print(f"Average Duration Accuracy: {avg_dur:.3f}")
    print(f"Validation Time: {validation_time:.2f}s")
    
    if detailed_errors:
        print(f"\n⚠️  {len(detailed_errors)} errors encountered:")
        for error in detailed_errors[:5]:  # Show first 5 errors
            print(f"  • {error}")
    
    # Save validation report
    if output_report:
        report_file = project_path / "json" / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                "wer_scores": wer_scores,
                "duration_accuracy": duration_accuracy,
                "processing_times": processing_times,
                "memory_usage": memory_usage,
                "overall_quality_score": overall_quality,
                "detailed_errors": detailed_errors,
                "validation_timestamp": time.time(),
                "whisper_model": whisper_model
            }, f, indent=2)
        print(f"\n💾 Validation report saved: {report_file}")
    
    return metrics


if __name__ == "__main__":
    typer.run(run)