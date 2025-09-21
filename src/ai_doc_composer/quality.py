"""Practical quality validation for AI Documentary Composer.

Focuses on measurable quality metrics that don't require heavy dependencies:
- Audio-text duration synchronization
- Performance benchmarking  
- File integrity validation
- Pipeline completeness checking

This provides concrete academic metrics without dependency issues.
"""

from __future__ import annotations

import json
import time
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

DATA_ROOT = Path(__file__).resolve().parents[2] / "projects"

import psutil


@dataclass
class QualityReport:
    """Container for quality validation results."""
    audio_sync_scores: Dict[str, float]
    narration_completeness_scores: Dict[str, float]
    file_integrity: Dict[str, bool]
    performance_metrics: Dict[str, float]
    pipeline_completeness: float
    overall_score: float
    detailed_issues: List[str]
    timestamp: float


class PipelineValidator:
    """Practical quality validation for documentary pipeline."""
    
    def __init__(self):
        """Initialize validator."""
        self.start_time = None
        self.memory_baseline = None
        
    def validate_audio_sync(self, audio_path: Path, script_text: str, target_wpm: int = 160) -> float:
        """Validate audio duration matches expected speech duration.
        
        Args:
            audio_path: Path to audio file
            script_text: Text that should be spoken
            target_wpm: Expected words per minute
            
        Returns:
            Sync accuracy score (0.0 to 1.0)
        """
        if not audio_path.exists():
            return 0.0
            
        try:
            # Get actual audio duration using ffprobe
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', str(audio_path)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return 0.0
                
            actual_duration = float(result.stdout.strip())
            
            # Calculate expected duration
            word_count = len(script_text.split())
            expected_duration = (word_count / target_wpm) * 60.0
            
            # Score based on how close actual matches expected (with tolerance)
            if expected_duration == 0:
                return 0.0
                
            ratio = actual_duration / expected_duration
            # Perfect score at 1.0 ratio, decreasing as it deviates
            if 0.8 <= ratio <= 1.2:  # Within 20% is good
                score = 1.0 - abs(1.0 - ratio) * 2  # Linear decrease
            else:
                score = max(0.0, 1.0 - abs(1.0 - ratio))
                
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0

    def validate_narration_completeness(self, audio_path: Path, clip_duration: float) -> Tuple[float, float]:
        """Validate that TTS narration fits within clip duration.
        
        Args:
            audio_path: Path to audio file
            clip_duration: Duration of video clip in seconds
            
        Returns:
            Tuple of (completeness_score, overage_seconds)
            - completeness_score: 1.0 if fits perfectly, decreasing with overage
            - overage_seconds: How many seconds narration exceeds clip (0.0 if fits)
        """
        if not audio_path.exists():
            return 0.0, 0.0
            
        try:
            # Get actual audio duration using ffprobe
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', str(audio_path)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return 0.0, 0.0
                
            tts_duration = float(result.stdout.strip())
            
            # Calculate overage
            overage = max(0.0, tts_duration - clip_duration)
            
            # Score calculation
            if overage == 0.0:
                score = 1.0  # Perfect fit
            else:
                # Penalty based on percentage overage
                overage_ratio = overage / clip_duration
                # More severe penalty for larger overages
                score = max(0.0, 1.0 - (overage_ratio * 2.0))
            
            return score, overage
            
        except Exception:
            return 0.0, 0.0
    
    def validate_file_integrity(self, project_path: Path) -> Dict[str, bool]:
        """Check if all expected pipeline files exist and are valid.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Dictionary of file validation results
        """
        integrity = {}
        
        # Check JSON artifacts
        json_dir = project_path / "json"
        for json_file in ["captions.json", "plan.json", "tts_meta.json"]:
            file_path = json_dir / json_file
            try:
                if file_path.exists():
                    with open(file_path) as f:
                        json.load(f)  # Test if valid JSON
                    integrity[json_file] = True
                else:
                    integrity[json_file] = False
            except Exception:
                integrity[json_file] = False
        
        # Check audio files existence
        audio_dir = project_path / "output" / "audio"
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.wav"))
            integrity["audio_files_exist"] = len(audio_files) > 0
            
            # Check if audio files have reasonable size (not empty/corrupted)
            valid_audio = 0
            for audio_file in audio_files:
                if audio_file.stat().st_size > 1000:  # At least 1KB
                    valid_audio += 1
            integrity["audio_files_valid"] = valid_audio == len(audio_files) if audio_files else False
        else:
            integrity["audio_files_exist"] = False
            integrity["audio_files_valid"] = False
        
        # Check final video - first try project-named file, then fall back to final.mp4
        project_name = project_path.name
        project_video = project_path / "output" / f"{project_name}.mp4"
        final_video = project_path / "output" / "final.mp4"

        if project_video.exists() and project_video.stat().st_size > 10000:
            integrity["final_video"] = True
        elif final_video.exists() and final_video.stat().st_size > 10000:
            integrity["final_video"] = True
        else:
            integrity["final_video"] = False
        
        return integrity
    
    def benchmark_performance(self, project_path: Path) -> Dict[str, float]:
        """Collect performance metrics for the pipeline.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # File system stats
        if project_path.exists():
            total_size = sum(f.stat().st_size for f in project_path.rglob("*") if f.is_file())
            metrics["total_project_size_mb"] = total_size / (1024 * 1024)
        else:
            metrics["total_project_size_mb"] = 0.0
        
        # Count processed items
        json_dir = project_path / "json"
        
        # Count clips from captions
        captions_file = json_dir / "captions.json"
        if captions_file.exists():
            try:
                with open(captions_file) as f:
                    captions_data = json.load(f)
                    metrics["clips_processed"] = len(captions_data.get("clips", []))
            except Exception:
                metrics["clips_processed"] = 0
        else:
            metrics["clips_processed"] = 0
        
        # Count audio files generated
        audio_dir = project_path / "output" / "audio"
        if audio_dir.exists():
            metrics["audio_files_generated"] = len(list(audio_dir.glob("*.wav")))
        else:
            metrics["audio_files_generated"] = 0
        
        # System resource usage
        process = psutil.Process()
        metrics["current_memory_mb"] = process.memory_info().rss / (1024 * 1024)
        metrics["cpu_percent"] = process.cpu_percent()
        
        # Disk usage efficiency (audio size vs expected)
        if metrics["audio_files_generated"] > 0:
            audio_size = sum(f.stat().st_size for f in audio_dir.glob("*.wav"))
            metrics["audio_size_mb"] = audio_size / (1024 * 1024)
            metrics["mb_per_audio_file"] = metrics["audio_size_mb"] / metrics["audio_files_generated"]
        else:
            metrics["audio_size_mb"] = 0.0
            metrics["mb_per_audio_file"] = 0.0
            
        return metrics
    
    def calculate_pipeline_completeness(self, integrity: Dict[str, bool]) -> float:
        """Calculate overall pipeline completeness score.
        
        Args:
            integrity: File integrity validation results
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        # Weight different components
        weights = {
            "captions.json": 0.2,
            "plan.json": 0.2, 
            "tts_meta.json": 0.2,
            "audio_files_exist": 0.2,
            "audio_files_valid": 0.15,
            "final_video": 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in integrity:
                total_score += weight * (1.0 if integrity[component] else 0.0)
            total_weight += weight
            
        return total_score / total_weight if total_weight > 0 else 0.0


def validate_project(project: str, target_wpm: int = 160, languages: str = "en") -> QualityReport:
    """Run comprehensive quality validation on a project.
    
    Args:
        project: Project name (directory under projects/)
        target_wpm: Expected words per minute for audio sync validation
        languages: Comma-separated language codes
        
    Returns:
        QualityReport with all validation results
    """
    project_path = DATA_ROOT / project
    if not project_path.exists():
        raise ValueError(f"Project directory not found: {project_path}")
    
    validator = PipelineValidator()
    lang_list = [lang.strip() for lang in languages.split(",")]
    
    print(f"🔍 Validating project: {project}")
    print(f"📊 Languages: {languages}")
    print(f"🎯 Target WPM: {target_wpm}")
    
    start_time = time.time()
    detailed_issues = []
    
    # Load pipeline artifacts
    json_dir = project_path / "json"
    plan_file = json_dir / "plan.json"
    captions_file = json_dir / "captions.json"
    
    if not plan_file.exists():
        detailed_issues.append("plan.json not found")
        script_lines = []
        ordered_clips = []
    else:
        try:
            with open(plan_file) as f:
                plan_data = json.load(f)
            script_lines = plan_data.get("script", [])
            ordered_clips = plan_data.get("ordered_clips", [])
        except Exception as e:
            detailed_issues.append(f"Failed to load plan.json: {e}")
            script_lines = []
            ordered_clips = []
    
    # Load clip durations from captions
    clip_durations = {}
    if captions_file.exists():
        try:
            with open(captions_file) as f:
                captions_data = json.load(f)
            for clip in captions_data.get("clips", []):
                clip_durations[clip["clip_id"]] = clip["duration"]
        except Exception as e:
            detailed_issues.append(f"Failed to load captions.json: {e}")
    
    # Validate audio sync and narration completeness for each language
    audio_sync_scores = {}
    narration_completeness_scores = {}
    audio_dir = project_path / "output" / "audio"
    
    for lang in lang_list:
        print(f"\n🔍 Validating {lang.upper()} audio sync and completeness...")
        
        sync_scores = []
        completeness_scores = []
        
        for i, script_line in enumerate(script_lines):
            if i >= len(ordered_clips):
                break
                
            clip_id = ordered_clips[i]
            audio_file = audio_dir / f"{clip_id}_{lang}.wav"
            
            if audio_file.exists():
                # Audio sync validation (TTS vs expected from WPM)
                sync_score = validator.validate_audio_sync(audio_file, script_line, target_wpm)
                sync_scores.append(sync_score)
                
                # Narration completeness validation (TTS vs clip duration)
                if clip_id in clip_durations:
                    clip_duration = clip_durations[clip_id]
                    completeness_score, overage = validator.validate_narration_completeness(audio_file, clip_duration)
                    completeness_scores.append(completeness_score)
                    
                    overage_status = f" (OVERAGE: +{overage:.2f}s)" if overage > 0 else ""
                    print(f"  ✓ {clip_id}_{lang}: sync={sync_score:.3f}, completeness={completeness_score:.3f}{overage_status}")
                else:
                    print(f"  ✓ {clip_id}_{lang}: sync={sync_score:.3f}, completeness=N/A (no duration)")
            else:
                detailed_issues.append(f"Audio file missing: {audio_file.name}")
                print(f"  ❌ Missing: {audio_file.name}")
        
        # Calculate language averages
        if sync_scores:
            audio_sync_scores[lang] = statistics.mean(sync_scores)
        else:
            audio_sync_scores[lang] = 0.0
            
        if completeness_scores:
            narration_completeness_scores[lang] = statistics.mean(completeness_scores)
        else:
            narration_completeness_scores[lang] = 0.0
    
    # File integrity validation
    print(f"\n🔍 Validating file integrity...")
    file_integrity = validator.validate_file_integrity(project_path)
    
    for file_name, is_valid in file_integrity.items():
        status = "✓" if is_valid else "❌"
        print(f"  {status} {file_name}")
        if not is_valid:
            detailed_issues.append(f"File integrity issue: {file_name}")
    
    # Performance benchmarking
    print(f"\n📊 Collecting performance metrics...")
    performance_metrics = validator.benchmark_performance(project_path)
    
    for metric, value in performance_metrics.items():
        if isinstance(value, float):
            print(f"  • {metric}: {value:.2f}")
        else:
            print(f"  • {metric}: {value}")
    
    # Calculate scores
    pipeline_completeness = validator.calculate_pipeline_completeness(file_integrity)
    
    # Overall score (weighted average including new completeness metric)
    if audio_sync_scores:
        avg_sync = statistics.mean(audio_sync_scores.values())
    else:
        avg_sync = 0.0
        
    if narration_completeness_scores:
        avg_completeness = statistics.mean(narration_completeness_scores.values())
    else:
        avg_completeness = 0.0
        
    # Revised weighting: sync 30%, completeness 30%, pipeline 40%
    overall_score = (avg_sync * 0.3 + avg_completeness * 0.3 + pipeline_completeness * 0.4)
    
    validation_time = time.time() - start_time
    
    # Create report
    report = QualityReport(
        audio_sync_scores=audio_sync_scores,
        narration_completeness_scores=narration_completeness_scores,
        file_integrity=file_integrity,
        performance_metrics=performance_metrics,
        pipeline_completeness=pipeline_completeness,
        overall_score=overall_score,
        detailed_issues=detailed_issues,
        timestamp=time.time()
    )
    
    # Output summary
    print(f"\n📊 QUALITY VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall Score: {overall_score:.3f}")
    print(f"Pipeline Completeness: {pipeline_completeness:.3f}")
    if audio_sync_scores:
        print(f"Average Audio Sync: {avg_sync:.3f}")
    if narration_completeness_scores:
        print(f"Average Narration Completeness: {avg_completeness:.3f}")
    print(f"Validation Time: {validation_time:.2f}s")
    print(f"Total Issues: {len(detailed_issues)}")
    
    # Save report
    report_file = project_path / "json" / "quality_report.json"
    with open(report_file, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    print(f"💾 Quality report saved: {report_file}")
    
    return report


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "switzerland"
    validate_project(project)