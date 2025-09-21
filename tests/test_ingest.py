"""Tests for the ingest module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from ai_doc_composer import ingest


class TestIngestStage:
    """Test suite for video ingestion and captioning."""

    def test_ffprobe_duration(self, tmp_path):
        """Test video duration extraction."""
        # Create mock video file
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        with patch('subprocess.run') as mock_run:
            # Mock ffprobe output
            mock_run.return_value = Mock(
                stdout='{"format": {"duration": "10.5"}}',
                returncode=0
            )

            duration = ingest._ffprobe_duration(str(video_path))

            assert duration == 10.5
            mock_run.assert_called_once()

    def test_extract_frame_at_timestamp(self, tmp_path):
        """Test frame extraction at specific timestamp."""
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        with patch('subprocess.run') as mock_run:
            with patch('PIL.Image.open') as mock_image:
                # Mock successful frame extraction
                mock_run.return_value = Mock(returncode=0)
                mock_image.return_value = MagicMock()

                frame = ingest._extract_frame_at(str(video_path), 5.0)

                assert frame is not None
                mock_run.assert_called_once()

    def test_extract_exif_timestamp(self, tmp_path):
        """Test EXIF timestamp extraction from video metadata."""
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        with patch('subprocess.run') as mock_run:
            # Mock ffprobe output with creation time
            mock_run.return_value = Mock(
                stdout='{"format": {"tags": {"creation_time": "2024-08-15T14:30:00.000000Z"}}}',
                returncode=0
            )

            timestamp = ingest._extract_exif_timestamp(str(video_path))

            assert timestamp == "2024-08-15T14:30:00.000000Z"

    def test_caption_generation_blip(self):
        """Test BLIP caption generation."""
        with patch('ai_doc_composer.ingest._processor') as mock_processor:
            with patch('ai_doc_composer.ingest._model') as mock_model:
                # Mock BLIP model pipeline
                mock_processor.return_value = {"input_ids": [[1, 2, 3]]}
                mock_model.generate.return_value = [[4, 5, 6]]
                mock_processor.decode.return_value = "A mountain landscape"

                mock_image = MagicMock()
                caption = ingest._caption_image(mock_image, "")

                assert caption == "A mountain landscape"

    @pytest.mark.parametrize("provider", ["blip", "gemini"])
    def test_run_with_different_providers(self, tmp_path, provider):
        """Test ingestion with different caption providers."""
        project_path = tmp_path / "test_project"
        input_path = project_path / "input"
        json_path = project_path / "json"
        input_path.mkdir(parents=True)
        json_path.mkdir(parents=True)

        # Create test video
        video_file = input_path / "test.mp4"
        video_file.touch()

        with patch('ai_doc_composer.ingest._ffprobe_duration', return_value=10.0):
            with patch('ai_doc_composer.ingest._extract_exif_timestamp', return_value="2024-01-01T00:00:00Z"):
                if provider == "blip":
                    with patch('ai_doc_composer.ingest._caption_image', return_value="Test caption"):
                        with patch('ai_doc_composer.ingest._extract_first_frame', return_value=MagicMock()):
                            with patch('ai_doc_composer.ingest.DATA_ROOT', tmp_path):
                                ingest.run("test_project", provider=provider)
                else:
                    with patch('ai_doc_composer.ingest._gemini_caption', return_value="Test caption"):
                        with patch('ai_doc_composer.ingest.DATA_ROOT', tmp_path):
                            ingest.run("test_project", provider=provider)

        # Check output file
        output_file = json_path / "captions.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert len(data["clips"]) == 1
            assert data["clips"][0]["caption"] == "Test caption"

    def test_retry_failed_clips(self, tmp_path):
        """Test retry mechanism for failed captions."""
        project_path = tmp_path / "test_project"
        input_path = project_path / "input"
        json_path = project_path / "json"
        input_path.mkdir(parents=True)
        json_path.mkdir(parents=True)

        # Create existing captions with one failed
        existing_captions = {
            "clips": [
                {"clip_id": "test1", "caption": "Success"},
                {"clip_id": "test2", "caption": "[FAILED]"}
            ]
        }

        captions_file = json_path / "captions.json"
        with open(captions_file, 'w') as f:
            json.dump(existing_captions, f)

        # Create video files
        (input_path / "test1.mp4").touch()
        (input_path / "test2.mp4").touch()

        with patch('ai_doc_composer.ingest.DATA_ROOT', tmp_path):
            with patch('ai_doc_composer.ingest._ffprobe_duration', return_value=10.0):
                with patch('ai_doc_composer.ingest._caption_image', return_value="Retry caption"):
                    with patch('ai_doc_composer.ingest._extract_first_frame', return_value=MagicMock()):
                        ingest.run("test_project", retry_failed=True)

        # Check that only failed clip was reprocessed
        with open(captions_file) as f:
            data = json.load(f)
            assert data["clips"][0]["caption"] == "Success"  # Unchanged
            assert data["clips"][1]["caption"] == "Retry caption"  # Updated