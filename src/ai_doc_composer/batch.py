"""
Batch processing module for AI Documentary Composer.

Enables processing multiple project flavors with different styles and models
from a simple YAML configuration.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import typer
import yaml
from datetime import datetime


def parse_project_name(project_name: str) -> Dict[str, str]:
    """
    Parse project folder name to extract configuration parameters.

    Pattern: <root>_<style>_<plan_model>_<tts_spec>

    Examples:
        dublin_doc_gemini_xtts -> root=dublin, style=doc, plan=gemini, tts=xtts, voice=None
        porto_vlog_gpt4o_xtts2 -> root=porto, style=vlog, plan=gpt4o, tts=xtts, voice=2
        switzerland_memories_llama_voice3 -> root=switzerland, style=memories, plan=llama, tts=xtts, voice=3
    """
    # Split by underscore
    parts = project_name.split('_')

    if len(parts) < 4:
        raise ValueError(f"Invalid project name format: {project_name}. Expected: <root>_<style>_<plan>_<tts>")

    # Extract base components
    root_parts = []
    style = None
    plan_model = None
    tts_spec = None

    # Style keywords
    styles = {'doc', 'documentary', 'vlog', 'memories'}
    # Plan model keywords
    plan_models = {'gemini', 'gpt4o', 'llama', 'ollama'}
    # TTS keywords
    tts_providers = {'xtts', 'gemini', 'voice'}

    # Parse from end backwards to identify components
    remaining_parts = parts.copy()

    # Last part is TTS spec
    tts_part = remaining_parts.pop()
    tts_provider = None
    voice_num = None

    # Parse TTS specification
    if tts_part.startswith('xtts'):
        tts_provider = 'xtts'
        if len(tts_part) > 4:
            voice_num = tts_part[4:]
    elif tts_part.startswith('voice'):
        tts_provider = 'xtts'
        voice_num = tts_part[5:]
    elif tts_part == 'gemini':
        tts_provider = 'gemini'
    else:
        raise ValueError(f"Unknown TTS specification: {tts_part}")

    # Second to last should be plan model
    if remaining_parts:
        plan_part = remaining_parts.pop()
        if plan_part in plan_models:
            plan_model = plan_part
        elif plan_part == 'gptoss':  # Legacy naming - maps to ollama with gpt-oss model
            plan_model = 'ollama'
        else:
            raise ValueError(f"Unknown plan model: {plan_part}")

    # Third to last should be style
    if remaining_parts:
        style_part = remaining_parts.pop()
        if style_part in styles:
            style = style_part
        else:
            raise ValueError(f"Unknown style: {style_part}")

    # Everything else is the root project name
    root = '_'.join(remaining_parts) if remaining_parts else None

    if not root:
        raise ValueError(f"Could not extract root project from: {project_name}")

    # Map short style names to full names
    style_mapping = {
        'doc': 'documentary',
        'vlog': 'travel_vlog',
        'memories': 'personal_memories'
    }

    full_style = style_mapping.get(style, style)

    # Handle special model mappings for ollama
    model_override = None
    if plan_model == 'ollama':
        # Map certain project names to specific ollama models
        if 'gptoss' in project_name:
            model_override = 'gpt-oss:20b'
        elif 'llama' in parts:
            model_override = 'llama3.3:latest'

    return {
        'root': root,
        'style': full_style,
        'plan_model': plan_model,
        'model_override': model_override,
        'tts_provider': tts_provider,
        'voice_num': voice_num,
        'project_name': project_name
    }


def setup_flavor_project(project_name: str, root_project: str, copy_input: str = 'preview') -> Path:
    """
    Set up directory structure for a flavor project.

    Args:
        project_name: Full project name (e.g., dublin_doc_gemini_xtts)
        root_project: Root project name (e.g., dublin)
        copy_input: 'preview' to copy input_preview, 'full' to copy input

    Returns:
        Path to the flavor project directory
    """
    projects_dir = Path('projects')
    root_dir = projects_dir / root_project
    flavor_dir = projects_dir / project_name

    # Verify root project exists
    if not root_dir.exists():
        raise FileNotFoundError(f"Root project not found: {root_dir}")

    # Create flavor directory structure
    flavor_dir.mkdir(parents=True, exist_ok=True)
    (flavor_dir / 'json').mkdir(exist_ok=True)
    (flavor_dir / 'output').mkdir(exist_ok=True)
    (flavor_dir / 'output' / 'audio').mkdir(exist_ok=True)

    # Copy input files
    if copy_input == 'preview' and (root_dir / 'input_preview').exists():
        src_input = root_dir / 'input_preview'
        dst_input = flavor_dir / 'input'
    else:
        src_input = root_dir / 'input'
        dst_input = flavor_dir / 'input'

    if src_input.exists():
        if dst_input.exists():
            shutil.rmtree(dst_input)
        shutil.copytree(src_input, dst_input)

    # Copy preview files if they exist (for Streamlit UI)
    if (root_dir / 'input_preview').exists():
        src_preview = root_dir / 'input_preview'
        dst_preview = flavor_dir / 'input_preview'
        if not dst_preview.exists():
            shutil.copytree(src_preview, dst_preview)

    # Copy captions.json from root project
    src_captions = root_dir / 'json' / 'captions.json'
    dst_captions = flavor_dir / 'json' / 'captions.json'
    if src_captions.exists():
        shutil.copy2(src_captions, dst_captions)

    # Copy project context files if they exist
    for context_file in ['project_metadata.json', 'clip_contexts.json']:
        src_context = root_dir / 'json' / context_file
        dst_context = flavor_dir / 'json' / context_file
        if src_context.exists():
            shutil.copy2(src_context, dst_context)

    return flavor_dir


def copy_voice_file(root_project: str, flavor_project: str, voice_num: Optional[str]) -> Optional[Path]:
    """
    Copy appropriate voice file to flavor project.

    Returns path to the voice file in the flavor project, or None if using default voice.
    """
    if not voice_num:
        return None

    projects_dir = Path('projects')
    root_voices = projects_dir / root_project / 'voices'

    # Look for voice file
    voice_file = root_voices / f'voice{voice_num}.wav'
    if not voice_file.exists():
        voice_file = root_voices / f'user_voice_en.wav'  # Fallback to default custom voice
        if not voice_file.exists():
            typer.echo(f"WARNING: Voice file not found for voice{voice_num}, using default")
            return None

    # Copy to flavor project
    flavor_dir = projects_dir / flavor_project
    dst_voice = flavor_dir / 'temp_speaker_voice.wav'
    shutil.copy2(voice_file, dst_voice)

    return dst_voice


def run_flavor_pipeline(
    project_name: str,
    config: Dict[str, str],
    skip_existing: bool = True,
    quality_modes: List[str] = ['preview']
) -> None:
    """
    Run the processing pipeline for a single flavor project.

    Args:
        project_name: Full project name
        config: Parsed configuration from project name
        skip_existing: Skip stages if output already exists
        quality_modes: List of quality modes to render ('preview' and/or 'full')
    """
    typer.echo(f"\n{'='*60}")
    typer.echo(f"🎬 Processing: {project_name}")
    typer.echo(f"   Root: {config['root']}")
    typer.echo(f"   Style: {config['style']}")
    typer.echo(f"   Plan Model: {config['plan_model']}")
    typer.echo(f"   TTS Provider: {config['tts_provider']}")
    if config['voice_num']:
        typer.echo(f"   Voice: {config['voice_num']}")
    typer.echo(f"{'='*60}\n")

    # Check if final output already exists before running pipeline
    if skip_existing:
        projects_dir = Path('projects')
        project_output = projects_dir / project_name / 'output'

        # Check for any of the possible output video names
        existing_outputs = []
        for quality in quality_modes:
            if quality == 'full':
                video_path = project_output / f"{project_name}_full_quality.mp4"
            else:
                video_path = project_output / f"{project_name}.mp4"

            if video_path.exists():
                existing_outputs.append(str(video_path.relative_to(projects_dir)))

        if existing_outputs:
            typer.echo(f"⏭️  Skipping - output already exists:")
            for output in existing_outputs:
                typer.echo(f"    {output}")
            return

    # Setup project directory (always use preview for initial setup)
    setup_flavor_project(project_name, config['root'], copy_input='preview')

    # Copy voice file if needed
    voice_file = None
    if config['tts_provider'] == 'xtts' and config['voice_num']:
        voice_file = copy_voice_file(config['root'], project_name, config['voice_num'])

    # Stage 2: Plan
    typer.echo(f"Running plan stage with {config['plan_model']} and style={config['style']}...")
    plan_cmd = [
        'poetry', 'run', 'python', '-m', 'ai_doc_composer.cli',
        'plan-stage', project_name,
        '--provider', config['plan_model'],
        '--style', config['style'],
    ]

    # Add model override for ollama
    if config.get('model_override'):
        plan_cmd.extend(['--model', config['model_override']])
        typer.echo(f"   Using model: {config['model_override']}")

    if skip_existing:
        plan_cmd.append('--skip-existing')

    result = subprocess.run(plan_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(f"ERROR: Plan stage failed: {result.stderr}")
        return

    # Stage 3: TTS
    typer.echo(f"Running TTS stage with {config['tts_provider']}...")
    tts_cmd = [
        'poetry', 'run', 'python', '-m', 'ai_doc_composer.cli',
        'tts-stage', project_name,
        '--provider', config['tts_provider'],
    ]

    # Add voice specification for XTTS
    if config['tts_provider'] == 'xtts' and voice_file:
        tts_cmd.extend(['--speaker', str(voice_file)])

    if skip_existing:
        tts_cmd.append('--skip-existing')

    result = subprocess.run(tts_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(f"ERROR: TTS stage failed: {result.stderr}")
        return

    # Stage 4: Render (for each quality mode)
    for quality_mode in quality_modes:
        typer.echo(f"🎬 Rendering {quality_mode} quality video...")

        # For full quality, copy full resolution files
        if quality_mode == 'full':
            setup_flavor_project(project_name, config['root'], copy_input='full')

        render_cmd = [
            'poetry', 'run', 'python', '-m', 'ai_doc_composer.cli',
            'render-stage', project_name,
        ]

        if quality_mode == 'full':
            render_cmd.append('--full-quality')

        if skip_existing:
            render_cmd.append('--skip-existing')

        result = subprocess.run(render_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            typer.echo(f"ERROR: Render stage failed: {result.stderr}")
        else:
            output_file = f"{project_name}_full_quality.mp4" if quality_mode == 'full' else f"{project_name}.mp4"
            typer.echo(f"Generated: projects/{project_name}/output/{output_file}")


def batch_process(config_path: Path, parallel: bool = False, skip_existing: bool = True) -> None:
    """
    Process multiple project flavors from a YAML configuration file.

    Args:
        config_path: Path to YAML configuration file
        parallel: Run projects in parallel (not implemented yet)
        skip_existing: Skip stages if output already exists
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        batch_config = yaml.safe_load(f)

    projects = batch_config.get('projects', [])
    quality_modes = batch_config.get('quality_modes', ['preview'])

    typer.echo(f"\n🚀 Batch Processing Configuration")
    typer.echo(f"   Projects: {len(projects)}")
    typer.echo(f"   Quality modes: {quality_modes}")
    typer.echo(f"   Skip existing: {skip_existing}\n")

    # Process each project
    successful = []
    failed = []

    for project_name in projects:
        try:
            # Parse project name
            config = parse_project_name(project_name)

            # Run pipeline
            run_flavor_pipeline(project_name, config, skip_existing, quality_modes)
            successful.append(project_name)

        except Exception as e:
            typer.echo(f"ERROR: Failed to process {project_name}: {str(e)}")
            failed.append((project_name, str(e)))

    # Summary
    typer.echo(f"\n{'='*60}")
    typer.echo(f"📊 Batch Processing Complete!")
    typer.echo(f"   Successful: {len(successful)}/{len(projects)}")
    if successful:
        typer.echo(f"   Successful: {', '.join(successful)}")
    if failed:
        typer.echo(f"   Failed: {len(failed)}")
        for proj, error in failed:
            typer.echo(f"      ERROR - {proj}: {error}")
    typer.echo(f"{'='*60}\n")


def migrate_project_name(old_name: str) -> str:
    """
    Migrate old project naming to new convention.

    Examples:
        porto_gem_xtts_doc -> porto_doc_gemini_xtts
        porto_gptoss_xtts_doc -> porto_doc_gpt4o_xtts
        dublin_memories_voice2 -> dublin_memories_gemini_xtts2
    """
    # Special cases
    migrations = {
        'dublin_memories_gemi': 'dublin_memories_gemini_gemini',
        'dublin_memories_voice2': 'dublin_memories_gemini_xtts2',
        'porto_gem_xtts_doc': 'porto_doc_gemini_xtts',
        'porto_gptoss_xtts_doc': 'porto_doc_gpt4o_xtts',
        'porto_memories_xtts2': 'porto_memories_gemini_xtts2',
        'porto_vibevoice_doc': 'porto_doc_gemini_xtts_voice2',  # Unclear, making assumption
        'porto_vlog_xtts2': 'porto_vlog_gemini_xtts2',
        'us_open_memories': 'us_open_memories_gemini_xtts',
        'us_open_vlog': 'us_open_vlog_gemini_xtts',
    }

    if old_name in migrations:
        return migrations[old_name]

    # Already in correct format
    try:
        parse_project_name(old_name)
        return old_name
    except:
        # Cannot migrate automatically
        return None


def update_json_references(json_file: Path, old_name: str, new_name: str) -> bool:
    """
    Update project name references in a JSON file.

    Returns True if file was modified, False otherwise.
    """
    try:
        with open(json_file, 'r') as f:
            content = f.read()
            data = json.loads(content)

        # Track if we made changes
        original_content = content

        # Update project field
        if isinstance(data, dict) and 'project' in data:
            if data['project'] == old_name:
                data['project'] = new_name

        # Update paths in tts_meta.json
        if 'tracks' in data and isinstance(data['tracks'], list):
            for track in data['tracks']:
                if 'clips' in track and isinstance(track['clips'], list):
                    for clip in track['clips']:
                        if 'wav' in clip and isinstance(clip['wav'], str):
                            # Replace old project name in path
                            clip['wav'] = clip['wav'].replace(f"{old_name}/", f"{new_name}/")

        # Convert back to string to check if changed
        new_content = json.dumps(data, indent=2)

        if new_content != original_content:
            with open(json_file, 'w') as f:
                f.write(new_content)
            return True
        return False

    except Exception as e:
        typer.echo(f"      WARNING: Error updating {json_file.name}: {str(e)}")
        return False


def migrate_existing_projects(dry_run: bool = True) -> None:
    """
    Migrate existing projects to new naming convention.

    Args:
        dry_run: If True, only show what would be done without actually renaming
    """
    projects_dir = Path('projects')

    typer.echo(f"\n{'[DRY RUN] ' if dry_run else ''}Migrating existing projects...\n")

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        old_name = project_dir.name
        new_name = migrate_project_name(old_name)

        if new_name and new_name != old_name:
            typer.echo(f"  {old_name} -> {new_name}")

            if not dry_run:
                json_files_updated = 0
                for json_file in project_dir.glob('json/*.json'):
                    if update_json_references(json_file, old_name, new_name):
                        json_files_updated += 1

                if json_files_updated > 0:
                    typer.echo(f"    Updated {json_files_updated} JSON files")

                new_path = projects_dir / new_name
                if new_path.exists():
                    typer.echo(f"    WARNING: Target already exists, skipping rename")
                else:
                    project_dir.rename(new_path)
                    typer.echo(f"    Renamed directory")
            else:
                # In dry run, check which JSON files would be updated
                json_files_to_update = []
                for json_file in project_dir.glob('json/*.json'):
                    try:
                        with open(json_file, 'r') as f:
                            content = f.read()
                        if old_name in content:
                            json_files_to_update.append(json_file.name)
                    except:
                        pass

                if json_files_to_update:
                    typer.echo(f"    Would update: {', '.join(json_files_to_update)}")

        elif new_name == old_name:
            typer.echo(f"  {old_name} (already correct)")
        else:
            typer.echo(f"  {old_name} ⚠️  (cannot auto-migrate, manual review needed)")

    typer.echo()