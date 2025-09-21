#!/bin/bash

# VibeVoice Setup Script (Poetry Version)
# This script installs VibeVoice dependencies using Poetry

echo "==================================="
echo "VibeVoice TTS Setup Script (Poetry)"
echo "==================================="

# Check if we're in the right directory
if [ ! -d "external/vibevoice" ]; then
    echo "Error: external/vibevoice directory not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed!"
    echo "Please install Poetry first: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected. VibeVoice will run on CPU (slower)"
fi

# Install VibeVoice dependencies with Poetry
echo ""
echo "Installing VibeVoice package with Poetry..."
poetry install --with vibevoice

if [ $? -eq 0 ]; then
    echo "✓ VibeVoice dependencies installed via Poetry"
else
    echo "✗ Failed to install VibeVoice dependencies"
    echo "Try running: poetry install --with vibevoice"
    exit 1
fi

# Create model directory if it doesn't exist
echo ""
echo "Creating model directory..."
mkdir -p models/vibevoice

# Print download instructions
echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. The VibeVoice model will be automatically downloaded on first use"
echo "   (approximately 3GB download from HuggingFace)"
echo ""
echo "2. To test the installation, run:"
echo "   poetry run python scripts/test_vibevoice_basic.py"
echo ""
echo "3. For unit tests, run:"
echo "   poetry run pytest tests/test_vibevoice/ -v"
echo ""
echo "4. For the complete test suite:"
echo "   poetry run python scripts/test_vibevoice_basic.py      # Basic synthesis"
echo "   poetry run python scripts/test_vibevoice_speakers.py   # Multi-speaker test"
echo "   poetry run python scripts/test_vibevoice_longform.py   # Long-form generation"
echo ""
echo "Note: First run will download the model which may take several minutes."
echo "The model will be cached for future use."
echo ""