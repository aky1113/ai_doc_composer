"""
Video preprocessing stage for AI Documentary Composer.

Converts large video files to web-friendly formats that comply with API limitations.
Target: <10MB files at 720p resolution for Gemini Vision API compatibility.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple
import typer


def run(project: str, target_size_mb: int = 8, force: bool = False) -> None:
    """
    Preprocess videos to reduce file size for API compatibility.
    
    Args:
        project: Project name (directory under projects/)
        target_size_mb: Target file size in MB (default: 8MB)
        force: Overwrite existing preview files
    """
    project_path = Path("projects") / project
    input_dir = project_path / "input"
    preview_dir = project_path / "input_preview"
    
    if not input_dir.exists():
        typer.echo(f"❌ Input directory not found: {input_dir}", err=True)
        raise typer.Exit(1)
    
    # Create preview directory
    preview_dir.mkdir(exist_ok=True)
    
    # Find video files
    video_files = _find_video_files(input_dir)
    if not video_files:
        typer.echo(f"❌ No video files found in {input_dir}")
        raise typer.Exit(1)
    
    typer.echo(f"🎬 Found {len(video_files)} video files to preprocess")
    
    processed_count = 0
    skipped_count = 0
    
    for video_file in video_files:
        output_file = preview_dir / f"{video_file.stem}.mp4"
        
        # Check if already processed and file size
        if output_file.exists() and not force:
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            if file_size_mb <= target_size_mb * 1.2:  # Allow 20% tolerance
                typer.echo(f"⏭️  Skipping {video_file.name} (already processed)")
                skipped_count += 1
                continue
        
        # Check original file size
        original_size_mb = video_file.stat().st_size / (1024 * 1024)
        
        if original_size_mb <= target_size_mb and not force:
            # File is already small enough, just copy
            typer.echo(f"📋 Copying {video_file.name} ({original_size_mb:.1f}MB)")
            _copy_file(video_file, output_file)
        else:
            # Compress the file
            typer.echo(f"🔄 Compressing {video_file.name} ({original_size_mb:.1f}MB → target: {target_size_mb}MB)")
            success = _compress_video(video_file, output_file, target_size_mb)
            
            if success:
                final_size_mb = output_file.stat().st_size / (1024 * 1024)
                typer.echo(f"✅ Compressed to {final_size_mb:.1f}MB")
            else:
                typer.echo(f"❌ Failed to compress {video_file.name}", err=True)
                continue
        
        processed_count += 1
    
    typer.echo(f"\n🎉 Preprocessing complete!")
    typer.echo(f"📊 Processed: {processed_count}, Skipped: {skipped_count}")
    typer.echo(f"📁 Preview files saved to: {preview_dir}")


def _find_video_files(directory: Path) -> List[Path]:
    """Find all video files in the directory."""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm'}
    video_files = []
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    return sorted(video_files)


def _copy_file(source: Path, destination: Path) -> None:
    """Copy file without modification."""
    try:
        subprocess.run([
            'cp', str(source), str(destination)
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Copy failed: {e.stderr.decode()}", err=True)
        raise


def _compress_video(input_file: Path, output_file: Path, target_size_mb: int) -> bool:
    """
    Compress video using FFmpeg with adaptive quality settings.
    
    Strategy:
    1. Scale to 720p height (maintain aspect ratio)  
    2. Use H.264 codec with optimized settings
    3. Adaptive CRF based on target file size
    """
    # Calculate target bitrate (accounting for audio ~128kbps)
    duration = _get_video_duration(input_file)
    if duration <= 0:
        typer.echo(f"❌ Could not determine video duration for {input_file.name}")
        return False
    
    # Target bitrate calculation (leaving room for audio)
    target_bits = (target_size_mb * 8 * 1024 * 1024)  # Convert MB to bits
    video_bitrate_kbps = max(500, int((target_bits / duration) / 1000) - 128)  # Subtract audio bitrate
    
    # FFmpeg command with adaptive quality
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-i', str(input_file),
        '-vf', 'scale=-2:720',  # Scale to 720p, maintain aspect ratio
        '-c:v', 'libx264',
        '-crf', '28',  # Good quality/size balance
        '-preset', 'fast',
        '-maxrate', f'{video_bitrate_kbps}k',
        '-bufsize', f'{video_bitrate_kbps * 2}k',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',  # Web optimization
        str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            typer.echo(f"FFmpeg error: {result.stderr}", err=True)
            return False
    except FileNotFoundError:
        typer.echo("❌ FFmpeg not found. Please install FFmpeg and ensure it's in PATH.", err=True)
        return False
    except Exception as e:
        typer.echo(f"❌ Compression failed: {str(e)}", err=True)
        return False


def _get_video_duration(video_file: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'csv=p=0',
        '-show_entries', 'format=duration',
        str(video_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def get_file_size_info(project: str) -> Tuple[List[dict], List[dict]]:
    """
    Get file size information for original and preview files.
    
    Returns:
        Tuple of (original_files_info, preview_files_info)
    """
    project_path = Path("projects") / project
    input_dir = project_path / "input"
    preview_dir = project_path / "input_preview"
    
    original_files = []
    preview_files = []
    
    if input_dir.exists():
        for video_file in _find_video_files(input_dir):
            size_mb = video_file.stat().st_size / (1024 * 1024)
            original_files.append({
                'name': video_file.name,
                'size_mb': size_mb,
                'path': str(video_file)
            })
    
    if preview_dir.exists():
        for video_file in _find_video_files(preview_dir):
            size_mb = video_file.stat().st_size / (1024 * 1024)
            preview_files.append({
                'name': video_file.name,
                'size_mb': size_mb,
                'path': str(video_file)
            })
    
    return original_files, preview_files


def needs_preprocessing(project: str, max_size_mb: int = 20) -> bool:
    """
    Check if project needs preprocessing based on file sizes.
    
    Args:
        project: Project name
        max_size_mb: Maximum acceptable file size (default: 20MB for Gemini)
    
    Returns:
        True if any files exceed the size limit
    """
    original_files, _ = get_file_size_info(project)
    return any(file_info['size_mb'] > max_size_mb for file_info in original_files)