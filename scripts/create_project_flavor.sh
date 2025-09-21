#!/bin/bash

# Create a new project flavor from an existing project
# Usage: ./create_project_flavor.sh SOURCE_PROJECT NEW_PROJECT
# Example: ./create_project_flavor.sh porto_vlog porto_family_trip

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 SOURCE_PROJECT NEW_PROJECT"
    echo "Example: $0 porto_vlog porto_family_trip"
    exit 1
fi

SOURCE=$1
NEW=$2
PROJECTS_DIR="$(dirname "$0")/../projects"

# Check if source project exists
if [ ! -d "$PROJECTS_DIR/$SOURCE" ]; then
    echo "Error: Source project '$SOURCE' not found in $PROJECTS_DIR"
    exit 1
fi

# Check if new project already exists
if [ -d "$PROJECTS_DIR/$NEW" ]; then
    echo "Error: Project '$NEW' already exists"
    exit 1
fi

echo "Creating new project flavor: $NEW from $SOURCE"

# Create directory structure
mkdir -p "$PROJECTS_DIR/$NEW"/{input,json,output}

# Create symlinks to input files
cd "$PROJECTS_DIR/$NEW/input"
ln -sf "../../$SOURCE/input/"* .
cd - > /dev/null

echo "✓ Created symlinks for input files"

# Copy captions.json if it exists
if [ -f "$PROJECTS_DIR/$SOURCE/json/captions.json" ]; then
    cp "$PROJECTS_DIR/$SOURCE/json/captions.json" "$PROJECTS_DIR/$NEW/json/"
    echo "✓ Copied captions.json"
else
    echo "⚠ No captions.json found in source project"
fi

# Copy project metadata if it exists
if [ -f "$PROJECTS_DIR/$SOURCE/json/project_metadata.json" ]; then
    cp "$PROJECTS_DIR/$SOURCE/json/project_metadata.json" "$PROJECTS_DIR/$NEW/json/"
    echo "✓ Copied project_metadata.json"
fi

echo ""
echo "✅ Project '$NEW' created successfully!"
echo ""
echo "Next steps:"
echo "1. Set the style and context for the new project:"
echo "   poetry run python -m ai_doc_composer.cli set-context $NEW --style <style> --project-context \"<context>\""
echo ""
echo "2. Generate a new plan with the desired style:"
echo "   poetry run python -m ai_doc_composer.cli plan-stage $NEW --style <style>"
echo ""
echo "3. Run TTS with your chosen provider:"
echo "   poetry run python -m ai_doc_composer.cli tts-stage $NEW --provider <provider>"
echo ""
echo "4. Render the final video:"
echo "   poetry run python -m ai_doc_composer.cli render-stage $NEW"