#!/bin/bash

# sync_projects.sh - Bidirectional sync for AI Documentary Composer projects
# Syncs projects folder between laptop and Linux server
# Author: AI Doc Composer Team
# Usage: ./scripts/sync_projects.sh [options]

set -euo pipefail

# Configuration
LOCAL_BASE="/Users/andrii/projects/uol/ai_doc_composer"
LOCAL_DIR="${LOCAL_BASE}/projects/"
REMOTE_HOST="tea@home"
REMOTE_DIR="/home/tea/projects/uol/ai_doc_composer/projects/"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
DRY_RUN=true
DELETE_ORPHANS=false
BACKUP_FIRST=false
EXCLUDE_VIDEO=false
VERBOSE=false
QUIET=false
SPECIFIC_PROJECT=""
FAST_MODE=false

# Exclusion patterns (always excluded)
EXCLUDE_PATTERNS=(
    "--exclude=.DS_Store"
    "--exclude=__pycache__/"
    "--exclude=*.pyc"
    "--exclude=*.pyo"
    "--exclude=*.tmp"
    "--exclude=*~"
    "--exclude=.*.swp"
    "--exclude=temp_speaker_voice.wav"
    "--exclude=.git/"
)

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Bidirectional sync of projects folder between laptop and Linux server.
By default runs in dry-run mode (safe preview).

OPTIONS:
    -e, --execute       Execute the sync (default is dry-run)
    -d, --delete        Delete orphaned files on destination
    -b, --backup        Create backup before syncing
    -n, --no-video      Exclude .MOV video files (large files)
    -p, --project NAME  Sync only specific project (e.g. dublin)
    -f, --fast          Fast mode (no compression, parallel transfers)
    -v, --verbose       Show detailed output
    -q, --quiet         Suppress non-error output
    -h, --help          Show this help message

EXAMPLES:
    $0                  # Preview what would be synced (dry-run)
    $0 -e               # Execute bidirectional sync
    $0 -e -p dublin     # Sync only dublin project
    $0 -e -f            # Fast sync (no compression)
    $0 -e -f -p dublin  # Fast sync of dublin only
    $0 -e -n            # Execute sync excluding .MOV files

SYNC BEHAVIOR:
    - Bidirectional: local→remote, then remote→local
    - Conflict resolution: newer file wins (based on modification time)
    - Preserves timestamps for accurate conflict resolution

PATHS:
    Local:  ${LOCAL_DIR}
    Remote: ${REMOTE_HOST}:${REMOTE_DIR}

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--execute)
            DRY_RUN=false
            shift
            ;;
        -d|--delete)
            DELETE_ORPHANS=true
            shift
            ;;
        -b|--backup)
            BACKUP_FIRST=true
            shift
            ;;
        -n|--no-video)
            EXCLUDE_VIDEO=true
            shift
            ;;
        -p|--project)
            SPECIFIC_PROJECT="$2"
            shift 2
            ;;
        -f|--fast)
            FAST_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_color "$RED" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Add video exclusion if requested
if [ "$EXCLUDE_VIDEO" = true ]; then
    EXCLUDE_PATTERNS+=("--exclude=*.MOV" "--exclude=*.mov")
    [ "$QUIET" = false ] && print_color "$YELLOW" "📹 Excluding .MOV video files from sync"
fi

# Build rsync options
RSYNC_OPTS=(
    "--archive"           # -rlptgoD (recursive, links, perms, times, group, owner, devices)
    "--update"            # Skip files that are newer on receiver
    "--times"             # Preserve modification times
    "--partial"           # Keep partial transfers for resume
    "--progress"          # Show progress during transfer
    "--human-readable"    # Human-readable file sizes
    "${EXCLUDE_PATTERNS[@]}"
)

# Add compression unless in fast mode
if [ "$FAST_MODE" = false ]; then
    RSYNC_OPTS+=("--compress")  # Compress during transfer
else
    # Fast mode optimizations
    RSYNC_OPTS+=("--whole-file")  # No delta-transfer algorithm
    RSYNC_OPTS+=("-W")            # Copy whole files
    [ "$QUIET" = false ] && print_color "$YELLOW" "⚡ Fast mode enabled (no compression)"
fi

# Add verbose flag if requested
if [ "$VERBOSE" = true ]; then
    RSYNC_OPTS+=("--verbose")
fi

# Add delete flag if requested
if [ "$DELETE_ORPHANS" = true ]; then
    RSYNC_OPTS+=("--delete" "--delete-excluded")
    [ "$QUIET" = false ] && print_color "$YELLOW" "🗑️  Will delete orphaned files"
fi

# Add dry-run flag if in preview mode
if [ "$DRY_RUN" = true ]; then
    RSYNC_OPTS+=("--dry-run")
fi

# Function to create backup
backup_projects() {
    local backup_dir="${LOCAL_BASE}/backups/projects_$(date +%Y%m%d_%H%M%S)"
    print_color "$CYAN" "📦 Creating backup at: $backup_dir"

    mkdir -p "$(dirname "$backup_dir")"

    if [ -d "$LOCAL_DIR" ]; then
        cp -r "$LOCAL_DIR" "$backup_dir"
        print_color "$GREEN" "✅ Backup completed"
    else
        print_color "$YELLOW" "⚠️  No local projects directory to backup"
    fi
}

# Function to check connectivity
check_connection() {
    [ "$QUIET" = false ] && print_color "$CYAN" "🔍 Checking connection to ${REMOTE_HOST}..."

    if ssh -o ConnectTimeout=5 -q "$REMOTE_HOST" exit 2>/dev/null; then
        [ "$QUIET" = false ] && print_color "$GREEN" "✅ Connection successful"
        return 0
    else
        print_color "$RED" "❌ Cannot connect to ${REMOTE_HOST}"
        print_color "$YELLOW" "Please ensure:"
        print_color "$YELLOW" "  1. The server is running"
        print_color "$YELLOW" "  2. SSH is configured correctly"
        print_color "$YELLOW" "  3. You have network connectivity"
        exit 1
    fi
}

# Function to perform sync
perform_sync() {
    local direction=$1
    local source=$2
    local dest=$3
    local desc=$4

    [ "$QUIET" = false ] && print_color "$BLUE" "\n━━━ $desc ━━━"

    if [ "$DRY_RUN" = true ]; then
        [ "$QUIET" = false ] && print_color "$YELLOW" "🔍 Preview mode (dry-run)"
    fi

    # Run rsync
    if rsync "${RSYNC_OPTS[@]}" "$source" "$dest"; then
        [ "$QUIET" = false ] && print_color "$GREEN" "✅ $desc completed"
    else
        local exit_code=$?
        if [ $exit_code -eq 23 ] || [ $exit_code -eq 24 ]; then
            # Partial transfer due to vanished files (23) or partial transfer (24)
            print_color "$YELLOW" "⚠️  $desc completed with warnings (some files changed during transfer)"
        else
            print_color "$RED" "❌ $desc failed with exit code: $exit_code"
            exit $exit_code
        fi
    fi
}

# Main execution
main() {
    print_color "$CYAN" "🔄 AI Documentary Composer - Project Sync"
    print_color "$CYAN" "========================================="

    # Show mode
    if [ "$DRY_RUN" = true ]; then
        print_color "$YELLOW" "🛡️  Running in SAFE MODE (dry-run)"
        print_color "$YELLOW" "   Use -e or --execute to perform actual sync"
    else
        print_color "$GREEN" "⚡ Running in EXECUTE MODE"
    fi

    # Check connection first
    check_connection

    # Create backup if requested
    if [ "$BACKUP_FIRST" = true ] && [ "$DRY_RUN" = false ]; then
        backup_projects
    fi

    # Handle specific project sync
    if [ -n "$SPECIFIC_PROJECT" ]; then
        LOCAL_DIR="${LOCAL_BASE}/projects/${SPECIFIC_PROJECT}/"
        REMOTE_DIR="/home/tea/projects/uol/ai_doc_composer/projects/${SPECIFIC_PROJECT}/"
        print_color "$CYAN" "🎯 Syncing only: ${SPECIFIC_PROJECT}"

        # Exclude other projects (comment them out effectively)
        for proj in porto switzerland us_open; do
            if [ "$proj" != "$SPECIFIC_PROJECT" ]; then
                EXCLUDE_PATTERNS+=("--exclude=${proj}/")
            fi
        done
    fi

    # Ensure local directory exists
    if [ ! -d "$LOCAL_DIR" ]; then
        print_color "$YELLOW" "📁 Creating local projects directory..."
        mkdir -p "$LOCAL_DIR"
    fi

    # Ensure remote directory exists
    [ "$QUIET" = false ] && print_color "$CYAN" "📁 Ensuring remote directory exists..."
    ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'" || true

    # Phase 1: Push local changes to remote (newer local files win)
    perform_sync "push" "$LOCAL_DIR" "${REMOTE_HOST}:${REMOTE_DIR}" "Push local → remote"

    # Phase 2: Pull remote changes to local (newer remote files win)
    perform_sync "pull" "${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR" "Pull remote → local"

    # Summary
    echo
    if [ "$DRY_RUN" = true ]; then
        print_color "$CYAN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_color "$YELLOW" "📋 Preview complete!"
        print_color "$YELLOW" "   Review the changes above"
        print_color "$YELLOW" "   Run with -e to execute sync"
    else
        print_color "$CYAN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        print_color "$GREEN" "✅ Sync completed successfully!"
        print_color "$GREEN" "   Projects are now synchronized"
    fi
}

# Run main function
main