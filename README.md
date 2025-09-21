# AI Documentary Composer

An automated pipeline for transforming unordered video clips into narrated documentaries using orchestrated AI models.

## Overview

This project implements an end-to-end artificial intelligence pipeline that automatically processes raw video footage to generate professionally narrated documentaries. The system demonstrates effective orchestration of multiple AI models across computer vision, natural language processing, speech synthesis, and video rendering domains.

**Academic Context**: University of London BSc Computer Science Final Project (CM3070)
**Category**: Template 4.1 - Orchestrating AI Models to Achieve a Goal

## Features

- **Multi-stage AI Pipeline**: Orchestrates four distinct AI models in sequence
- **Flexible Provider Support**: Choose between local and cloud-based AI models
- **Multilingual Capabilities**: Generate documentaries in multiple languages
- **Style-aware Narration**: Support for different documentary styles (documentary, travel vlog, personal narrative)
- **Web Interface**: Modern Streamlit-based UI for easy interaction
- **CLI Support**: Comprehensive command-line interface for automation
- **Quality Validation**: Built-in metrics for evaluating output quality

## Architecture

The system consists of five processing stages:

1. **Vision Analysis**: Extracts frames from video clips and generates descriptive captions using BLIP-2 or Gemini Vision API
2. **Narrative Planning**: Orders clips and generates contextual narration using Llama-3 or Gemini Pro
3. **Speech Synthesis**: Converts narration text to natural speech using XTTS or Gemini TTS
4. **Video Rendering**: Assembles final video with synchronized audio using FFmpeg
5. **Quality Validation** (optional): Evaluates output quality using metrics like WER, timing accuracy, and narrative coherence

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg and ffprobe (must be available in PATH)
- Poetry for dependency management

### Setup

1. Clone the repository:
```bash
git clone https://github.com/aky1113/ai_doc_composer
cd ai_doc_composer
```

2. Install dependencies:
```bash
poetry install
```

3. Configure environment variables:

Create a `.env` file in the project root:
```bash
# Create .env file
echo 'GEMINI_API_KEY="your-gemini-api-key-here"' > .env
```

Alternatively, you can export them directly:
```bash
export GEMINI_API_KEY="your-api-key"  # For Gemini providers
export OLLAMA_BASE_URL="http://localhost:11434"  # For Ollama (optional)
```

**Note**: To obtain a Gemini API key:
- Visit https://aistudio.google.com/apikey
- Click "Create API Key"
- Copy the generated key and add it to your `.env` file

4. Install additional models (if using local providers):
```bash
# For Ollama
ollama pull llama3

# For TTS (installed automatically on first use)
poetry run pip install tts
```

## Usage

### Web Interface

Launch the Streamlit interface:
```bash
poetry run python -m ai_doc_composer.cli ui
```

Access the application at `http://localhost:8501`

### Command Line Interface

Process a complete pipeline:
```bash
# Stage 1: Generate captions from videos
poetry run python -m ai_doc_composer.cli ingest-stage switzerland --provider gemini

# Stage 2: Plan narrative structure
poetry run python -m ai_doc_composer.cli plan-stage switzerland --provider gemini --style documentary

# Stage 3: Synthesize speech
poetry run python -m ai_doc_composer.cli tts-stage switzerland --provider xtts

# Stage 4: Render final video
poetry run python -m ai_doc_composer.cli render-stage switzerland

# Stage 5: Validate quality (optional)
poetry run python -m ai_doc_composer.cli validate-stage switzerland
```

### Docker Deployment

Build and run using Docker Compose:
```bash
docker-compose up ui
```

## Project Structure

```
ai_doc_composer/
├── src/ai_doc_composer/       # Core application modules
│   ├── cli.py                 # Command-line interface
│   ├── ui.py                  # Web interface
│   ├── ingest.py              # Vision captioning stage
│   ├── plan.py                # Narrative planning stage
│   ├── tts.py                 # Speech synthesis stage
│   ├── render.py              # Video rendering stage
│   ├── validate.py            # Quality validation stage
│   ├── quality.py             # Quality metrics and scoring
│   └── styles.py              # Documentary style configurations
├── projects/                  # Sample datasets and outputs
│   └── switzerland/           # Example project
├── tests/                     # Test suite
├── docs/                      # Additional documentation
└── pyproject.toml            # Project dependencies
```

## Performance Metrics

### Quality Validation Metrics
The optional quality validation stage evaluates:
- **Word Error Rate (WER)**: Measures speech-to-text accuracy
- **Timing Accuracy**: Validates narration-to-video synchronization
- **Narrative Coherence**: Checks for duplicate phrases and flow
- **Technical Quality**: Audio levels, video integrity, format compliance

## API Providers

### Vision Models
- **BLIP-2**: Local Hugging Face model (Salesforce/blip2-opt-2.7b)
- **Gemini Vision**: Google Cloud API (gemini-1.5-flash)

### Language Models
- **Llama-3**: Local via Ollama (llama3:8b)
- **Gemini Pro**: Google Cloud API (gemini-1.5-pro)

### Speech Synthesis
- **XTTS**: Local Coqui TTS model
- **Gemini TTS**: Google Cloud API (en-US-Wavenet)
