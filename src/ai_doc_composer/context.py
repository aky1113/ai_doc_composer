"""
Personal context handling for AI Documentary Composer.

Provides utilities for collecting, storing, and integrating personal context
into documentary narratives.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Project-level context and configuration."""
    style: str = "documentary"
    project_context: str = ""
    ordering_strategy: str = "llm"  # "llm" or "exif"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectContext":
        """Create from dictionary."""
        # Filter out deprecated fields for backward compatibility
        filtered_data = {
            k: v for k, v in data.items()
            if k not in ['generation_mode', 'chapter_duration']
        }
        return cls(**filtered_data)


@dataclass 
class ClipContext:
    """Per-clip context and metadata."""
    filename: str
    personal_context: str = ""
    exif_timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClipContext":
        """Create from dictionary."""
        return cls(**data)


class ContextManager:
    """Manages personal context for projects and clips."""
    
    def __init__(self, project_path: Path):
        """Initialize context manager for a project."""
        self.project_path = project_path
        self.json_dir = project_path / "json"
        self.json_dir.mkdir(exist_ok=True)
        
        self.project_metadata_file = self.json_dir / "project_metadata.json"
        self.clip_contexts_file = self.json_dir / "clip_contexts.json"
    
    def load_project_context(self) -> ProjectContext:
        """Load project-level context and configuration."""
        if self.project_metadata_file.exists():
            try:
                with open(self.project_metadata_file, 'r') as f:
                    data = json.load(f)
                return ProjectContext.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load project metadata: {e}")
        
        # Return default context
        return ProjectContext()
    
    def save_project_context(self, context: ProjectContext) -> None:
        """Save project-level context and configuration."""
        with open(self.project_metadata_file, 'w') as f:
            json.dump(context.to_dict(), f, indent=2)
    
    def load_clip_contexts(self) -> Dict[str, ClipContext]:
        """Load per-clip contexts."""
        if self.clip_contexts_file.exists():
            try:
                with open(self.clip_contexts_file, 'r') as f:
                    data = json.load(f)
                return {
                    filename: ClipContext.from_dict(clip_data)
                    for filename, clip_data in data.items()
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load clip contexts: {e}")
        
        return {}
    
    def save_clip_contexts(self, contexts: Dict[str, ClipContext]) -> None:
        """Save per-clip contexts."""
        data = {
            filename: context.to_dict()
            for filename, context in contexts.items()
        }
        with open(self.clip_contexts_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_clip_context(self, filename: str, personal_context: str = "", 
                           exif_timestamp: Optional[str] = None) -> None:
        """Update context for a specific clip."""
        contexts = self.load_clip_contexts()
        
        if filename in contexts:
            contexts[filename].personal_context = personal_context
            if exif_timestamp:
                contexts[filename].exif_timestamp = exif_timestamp
        else:
            contexts[filename] = ClipContext(
                filename=filename,
                personal_context=personal_context,
                exif_timestamp=exif_timestamp
            )
        
        self.save_clip_contexts(contexts)
    
    def get_clip_context(self, filename: str) -> ClipContext:
        """Get context for a specific clip."""
        contexts = self.load_clip_contexts()
        return contexts.get(filename, ClipContext(filename=filename))
    
    def initialize_clips(self, clip_filenames: List[str]) -> None:
        """Initialize context entries for new clips."""
        contexts = self.load_clip_contexts()
        
        for filename in clip_filenames:
            if filename not in contexts:
                contexts[filename] = ClipContext(filename=filename)
        
        self.save_clip_contexts(contexts)


def collect_context_via_stt(audio_file: Path) -> str:
    """
    Convert speech to text for context input.
    
    TODO: Implement speech-to-text functionality.
    For now, this is a placeholder that will be implemented
    in Phase 3 with proper STT integration.
    """
    logger.warning("Speech-to-text not yet implemented")
    return ""


def integrate_context_into_prompt(base_prompt: str, personal_context: str, 
                                 project_context: str = "") -> str:
    """
    Merge personal context with style prompt.
    
    Integrates both project-level and clip-level personal context
    into the base planning prompt.
    """
    context_sections = []
    
    if project_context.strip():
        context_sections.append(f"Project Context: {project_context.strip()}")
    
    if personal_context.strip():
        context_sections.append(f"Personal Context: {personal_context.strip()}")
    
    if context_sections:
        context_block = "\n\n" + "\n".join(context_sections) + "\n"
        return base_prompt + context_block
    
    return base_prompt


def migrate_legacy_project(project_path: Path) -> None:
    """
    Migrate legacy project to new context system.
    
    Adds default style and empty context for existing projects
    to maintain backward compatibility.
    """
    context_manager = ContextManager(project_path)
    
    # Check if project already has metadata
    if not context_manager.project_metadata_file.exists():
        # Create default project context
        default_context = ProjectContext(
            style="documentary",  # Default to documentary for existing projects
            project_context="",
            ordering_strategy="llm"
        )
        context_manager.save_project_context(default_context)
        logger.info(f"Migrated legacy project at {project_path} to documentary style")
    
    # Initialize clip contexts if input directory exists
    input_dir = project_path / "input"
    if input_dir.exists():
        clip_files = [f.name for f in input_dir.glob("*.mp4")]
        if clip_files:
            context_manager.initialize_clips(clip_files)
            logger.info(f"Initialized context for {len(clip_files)} clips")