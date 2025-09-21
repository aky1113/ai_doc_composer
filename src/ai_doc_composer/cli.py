import typer
from typing import Optional

app = typer.Typer(add_completion=False, help="AI Documentary Composer pipeline CLI")


@app.command()
def preprocess_stage(
    project: str = typer.Argument(..., help="Project name under projects/"),
    target_size_mb: int = typer.Option(8, help="Target file size in MB for preview files"),
    force: bool = typer.Option(False, help="Overwrite existing preview files"),
) -> None:
    """Stage 0 – video preprocessing for API compatibility (reduce file sizes)."""
    from . import preprocess
    preprocess.run(project, target_size_mb=target_size_mb, force=force)


@app.command()
def ingest_stage(
    project: str = typer.Argument(..., help="Project name under projects/"),
    frames: int = typer.Option(1, help="Number of frames to sample per clip"),
    prompt: str = typer.Option("", help="Optional prompt to prepend for BLIP"),
    model: str = typer.Option(None, help="Model ID (HF for BLIP provider)"),
    provider: str = typer.Option("blip", help="Caption provider: blip or gemini"),
    skip_existing: bool = typer.Option(False, help="Skip if captions.json already exists"),
    retry_failed: bool = typer.Option(False, help="Only retry clips with failed captions"),
    max_retries: int = typer.Option(3, help="Maximum number of retries for failed API calls"),
) -> None:
    """Stage 1 – vision captioning."""
    from . import ingest
    from pathlib import Path
    
    if skip_existing:
        captions_file = Path("projects") / project / "json" / "captions.json"
        if captions_file.exists():
            typer.echo(f"Skipping ingest stage - captions.json already exists")
            return
    
    ingest.run(project, frames=frames, prompt=prompt, model_id=model, provider=provider, 
               retry_failed=retry_failed, max_retries=max_retries)


@app.command()
def plan_stage(
    project: str = typer.Argument(..., help="Project name under projects/"),
    provider: str = typer.Option("ollama", help="LLM backend: ollama | gemini | static"),
    model: Optional[str] = typer.Option(None, help="Model ID override, e.g. llama3.1:latest"),
    temperature: float = typer.Option(0.15, help="Sampling temperature for the LLM (lower = better duration compliance)"),
    wpm: int = typer.Option(160, help="Words-per-minute used for speech timing"),
    slack: float = typer.Option(0.15, help="Allowed ± slack ratio for narration vs. clip duration"),
    languages: str = typer.Option("en", help="Comma-separated BCP47 codes for narration languages"),
    style: Optional[str] = typer.Option(None, help="Documentary style: documentary | travel_vlog | personal_memories"),
    ordering_strategy: str = typer.Option("llm", help="Clip ordering strategy: llm | exif"),
    skip_existing: bool = typer.Option(False, help="Skip if plan.json already exists"),
    allow_overflow: bool = typer.Option(False, help="Allow narration to overflow between clips with compensation"),
) -> None:
    """Stage 2 – language planning (ordering + narration)."""
    from . import plan
    from pathlib import Path

    if skip_existing:
        plan_file = Path("projects") / project / "json" / "plan.json"
        if plan_file.exists():
            typer.echo(f"Skipping plan stage - plan.json already exists")
            return

    plan.run(
        project,
        provider=provider,
        model=model,
        temperature=temperature,
        wpm=wpm,
        slack=slack,
        languages=languages,
        style=style,
        ordering_strategy=ordering_strategy,
        skip_existing=skip_existing,
        allow_overflow=allow_overflow,
    )


@app.command()
def tts_stage(
    project: str = typer.Argument(..., help="Project name under projects/"),
    languages: str = typer.Option("en", help="Comma-separated BCP-47 codes (e.g. en,fr)"),
    speaker: Optional[str] = typer.Option(None, help="Speaker WAV, voice name, or mapping 'en:voiceA,ru:voiceB'"),
    provider: str = typer.Option("xtts", help="TTS backend: xtts | gemini"),
    model_id: str = typer.Option(None, help="Model ID override (XTTS or Gemini)"),
    gpu: Optional[bool] = typer.Option(None, help="Force GPU on/off (XTTS only)"),
    clean: bool = typer.Option(False, help="Remove existing WAV files before synthesis"),
    skip_existing: bool = typer.Option(False, help="Skip if tts_meta.json already exists"),
) -> None:
    """Stage 3 – text-to-speech synthesis (XTTS local or Gemini cloud)."""
    from . import tts
    from pathlib import Path
    
    if skip_existing:
        tts_file = Path("projects") / project / "json" / "tts_meta.json"
        if tts_file.exists():
            typer.echo(f"Skipping TTS stage - tts_meta.json already exists")
            return
    
    tts.run(
        project,
        provider=provider,
        languages=languages,
        speaker=speaker,
        model_id=model_id or (tts.DEFAULT_GEMINI_MODEL if provider == "gemini" else tts.DEFAULT_MODEL),
        gpu=gpu,
        clean=clean,
    )


@app.command()
def render_stage(
    project: str = typer.Argument(..., help="Project name under projects/"),
    offset: float = typer.Option(1.0, help="Seconds of silence before narration starts for each clip"),
    full_quality: bool = typer.Option(False, help="Use original high-resolution files instead of preview files"),
    skip_existing: bool = typer.Option(False, help="Skip if final video already exists"),
) -> None:
    """Stage 4 – final video render (FFmpeg mux)."""
    from . import render
    from pathlib import Path
    
    if skip_existing:
        project_output = Path("projects") / project / "output"
        final_video = project_output / f"{project}.mp4"
        if not final_video.exists():
            final_video = project_output / "final.mp4"
        
        if final_video.exists():
            typer.echo(f"Skipping render stage - final video already exists: {final_video}")
            return
    
    render.run(project, offset=offset, full_quality=full_quality)


@app.command()
def validate_stage(
    project: str = typer.Argument(..., help="Project name under projects/"),
    languages: str = typer.Option("en", help="Comma-separated language codes to validate"),
    whisper_model: str = typer.Option("base", help="Whisper model size: tiny, base, small, medium, large"),
    output_report: bool = typer.Option(True, help="Save validation report to JSON"),
    use_faster_whisper: bool = typer.Option(True, help="Use faster-whisper instead of openai-whisper"),
) -> None:
    """Stage 5 – quality validation (WER, duration, performance metrics)."""
    from . import validate
    validate.run(
        project=project,
        languages=languages,
        whisper_model=whisper_model,
        output_report=output_report,
        use_faster_whisper=use_faster_whisper
    )


@app.command()
def quality_stage(
    project: str = typer.Argument(..., help="Project name under projects/"),
    languages: str = typer.Option("en", help="Comma-separated language codes"),
    target_wpm: int = typer.Option(160, help="Expected words per minute for sync validation"),
) -> None:
    """Stage 5 – practical quality validation (no heavy dependencies)."""
    from . import quality
    report = quality.validate_project(
        project=project,
        target_wpm=target_wpm,
        languages=languages
    )
    print(f"\nQuality validation completed")
    print(f"📊 Overall Score: {report.overall_score:.3f}")
    print(f"💾 Report saved to: projects/{project}/json/quality_report.json")


@app.command()
def benchmark_stage(
    project: str = typer.Argument(..., help="Project name under projects/"),
    save_report: bool = typer.Option(True, help="Save benchmark report to JSON"),
) -> None:
    """Stage 6 – performance benchmarking."""
    from . import benchmark
    report = benchmark.benchmark_project(project, save_report=save_report)
    print(f"\n🚀 Benchmarking completed!")
    print(f"⚡ Total Time: {report.total_processing_time:.2f}s")
    print(f"💾 Report saved to: projects/{project}/json/benchmark_report.json")


@app.command()
def set_context(
    project: str = typer.Argument(..., help="Project name under projects/"),
    project_context: str = typer.Option("", help="Project-level context description"),
    style: str = typer.Option("documentary", help="Documentary style: documentary | travel_vlog | personal_memories"),
    ordering_strategy: str = typer.Option("llm", help="Clip ordering strategy: llm | exif"),
) -> None:
    """Set project context and configuration for styles feature."""
    from .context import ContextManager
    from pathlib import Path
    
    project_path = Path("projects") / project
    if not project_path.exists():
        typer.secho(f"Error: Project '{project}' not found at {project_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    
    context_manager = ContextManager(project_path)
    proj_context = context_manager.load_project_context()
    
    # Update project context
    proj_context.style = style
    proj_context.project_context = project_context
    proj_context.ordering_strategy = ordering_strategy
    
    context_manager.save_project_context(proj_context)
    
    typer.secho(f"Project context updated for '{project}'", fg=typer.colors.GREEN)
    typer.echo(f"   Style: {style}")
    typer.echo(f"   Ordering: {ordering_strategy}")
    if project_context:
        typer.echo(f"   Context: {project_context[:100]}{'...' if len(project_context) > 100 else ''}")


@app.command()
def set_clip_context(
    project: str = typer.Argument(..., help="Project name under projects/"),
    filename: str = typer.Argument(..., help="Video filename to set context for"),
    context: str = typer.Argument(..., help="Personal context for this specific clip"),
) -> None:
    """Set personal context for a specific video clip."""
    from .context import ContextManager
    from pathlib import Path
    
    project_path = Path("projects") / project
    if not project_path.exists():
        typer.secho(f"Error: Project '{project}' not found at {project_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    
    # Check if video file exists
    input_path = project_path / "input" / filename
    if not input_path.exists():
        typer.secho(f"Error: Video file '{filename}' not found in project input directory", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    
    context_manager = ContextManager(project_path)
    context_manager.update_clip_context(filename, personal_context=context)
    
    typer.secho(f"Context set for '{filename}'", fg=typer.colors.GREEN)
    typer.echo(f"   Context: {context[:100]}{'...' if len(context) > 100 else ''}")




@app.command()
def list_styles() -> None:
    """List all available documentary styles."""
    from .styles import DocumentaryStyles
    
    typer.echo("📽️  Available Documentary Styles:")
    typer.echo()
    
    styles = DocumentaryStyles.list_styles()
    for style_key, display_name in styles.items():
        style_config = DocumentaryStyles.get_style_by_name(style_key)
        typer.echo(f"  🎬 {typer.style(display_name, fg=typer.colors.CYAN, bold=True)}")
        typer.echo(f"     Key: {style_key}")
        typer.echo(f"     Tone: {style_config.tone}")
        typer.echo(f"     Description: {style_config.master_prompt_template[:80]}...")
        typer.echo()


@app.command() 
def migrate_project(
    project: str = typer.Argument(..., help="Project name to migrate to new context system"),
) -> None:
    """Migrate existing project to support styles and context features."""
    from .context import migrate_legacy_project
    from pathlib import Path
    
    project_path = Path("projects") / project
    if not project_path.exists():
        typer.secho(f"Error: Project '{project}' not found at {project_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    
    migrate_legacy_project(project_path)
    typer.secho(f"Project '{project}' migrated to support styles and context features", fg=typer.colors.GREEN)


@app.command()
def ui(
    port: int = typer.Option(8501, help="Port to run the Streamlit web interface on"),
    address: str = typer.Option("localhost", help="Address to bind to (use 'localhost' for microphone)"),
    https: bool = typer.Option(False, help="Enable HTTPS with self-signed certificate"),
) -> None:
    """Launch modern Streamlit web UI for the AI Documentary Composer.

    Note: Microphone access requires HTTPS or localhost. Use default settings
    for local development, or --https for network access with microphone.
    """
    import subprocess
    import sys
    from pathlib import Path
    import os

    # Get the path to the ui.py file (now Streamlit-based)
    ui_file = Path(__file__).parent / "ui.py"

    # Build base command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ui_file),
        "--server.port", str(port),
        "--server.address", address
    ]

    protocol = "http"

    # Handle HTTPS mode
    if https:
        # Check if certificates exist in .streamlit folder
        streamlit_dir = Path.cwd() / ".streamlit"
        cert_file = streamlit_dir / "cert.pem"
        key_file = streamlit_dir / "key.pem"

        if not cert_file.exists() or not key_file.exists():
            print("🔐 Generating self-signed certificate for HTTPS...")
            streamlit_dir.mkdir(exist_ok=True)

            # Generate self-signed certificate using openssl
            gen_cert_cmd = f'openssl req -x509 -newkey rsa:4096 -keyout "{key_file}" -out "{cert_file}" -days 365 -nodes -subj "/CN=localhost"'
            result = os.system(gen_cert_cmd)

            if result != 0:
                print("❌ Failed to generate certificate. Make sure openssl is installed.")
                print("   On macOS: brew install openssl")
                return

        cmd.extend(["--server.sslCertFile", str(cert_file)])
        cmd.extend(["--server.sslKeyFile", str(key_file)])
        protocol = "https"

    print(f"🚀 Launching AI Documentary Composer UI (Streamlit) on port {port}...")
    print(f"🌐 Access at: {protocol}://{address}:{port}")

    if address != "localhost" and not https:
        print("\n⚠️  WARNING: Microphone access will NOT work!")
        print("   Browsers require HTTPS for microphone on non-localhost addresses.")
        print("\n   Solutions:")
        print(f"   1. Access via: http://localhost:{port}")
        print(f"   2. Use HTTPS: poetry run ai-doc-composer ui --https")
        print(f"   3. Use ngrok: ngrok http {port}")

    if https:
        print("\n🔐 Using HTTPS with self-signed certificate")
        print("   Your browser will show a security warning - this is normal.")
        print("   Click 'Advanced' and 'Proceed' to access the app.")

    print("\n🎬 Press Ctrl+C to stop the server")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Streamlit: {e}")
    except KeyboardInterrupt:
        print("\n👋 Streamlit UI stopped")


@app.command()
def gradio_ui(
    port: int = typer.Option(7860, help="Port to run the Gradio web interface on"),
    share: bool = typer.Option(False, help="Create shareable public link via Gradio"),
) -> None:
    """Launch legacy Gradio web UI for the AI Documentary Composer."""
    try:
        from . import ui_gradio_backup as ui_module
        print(f"🚀 Launching AI Documentary Composer UI (Gradio Legacy) on port {port}...")
        if share:
            print("🌐 Creating shareable public link...")
        ui_module.launch_ui(share=share, port=port)
    except ImportError:
        print("Gradio UI backup not available. Use the main 'ui' command for Streamlit instead.")


@app.command()
def batch(
    config: str,
    skip_existing: bool = typer.Option(True, help="Skip stages if output already exists"),
    parallel: bool = typer.Option(False, help="Process projects in parallel (not implemented yet)"),
) -> None:
    """Process multiple project flavors from a YAML configuration file."""
    from pathlib import Path
    from . import batch as batch_module

    config_path = Path(config)
    batch_module.batch_process(config_path, parallel=parallel, skip_existing=skip_existing)


@app.command()
def migrate_projects(
    dry_run: bool = typer.Option(True, help="Show what would be done without actually renaming"),
) -> None:
    """Migrate existing projects to the new naming convention."""
    from . import batch as batch_module
    batch_module.migrate_existing_projects(dry_run=dry_run)


@app.command()
def parse_project_name(
    name: str,
) -> None:
    """Parse and display the components of a project name."""
    from . import batch as batch_module

    try:
        config = batch_module.parse_project_name(name)
        typer.echo(f"\n📋 Project Configuration:")
        typer.echo(f"   Name: {name}")
        typer.echo(f"   Root: {config['root']}")
        typer.echo(f"   Style: {config['style']}")
        typer.echo(f"   Plan Model: {config['plan_model']}")
        typer.echo(f"   TTS Provider: {config['tts_provider']}")
        if config['voice_num']:
            typer.echo(f"   Voice Number: {config['voice_num']}")
    except Exception as e:
        typer.echo(f"Error parsing project name: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()