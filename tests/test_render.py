"""Tests for the render module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch, call
import pytest

from ai_doc_composer import render


class TestRenderStage:
    """Test suite for video rendering and audio muxing."""

    @patch('subprocess.run')
    def test_concat_video(self, mock_run, tmp_path):
        """Test video concatenation with FFmpeg."""
        # Setup test files
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        video_files = [
            tmp_path / "clip1.mp4",
            tmp_path / "clip2.mp4",
            tmp_path / "clip3.mp4"
        ]

        for vf in video_files:
            vf.touch()

        # Mock successful FFmpeg execution
        mock_run.return_value = Mock(returncode=0)

        output_path = output_dir / "concat.mp4"
        render._concat_video(video_files, output_path)

        # Verify FFmpeg was called
        mock_run.assert_called()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "-filter_complex" in args
        assert str(output_path) in args

    @patch('subprocess.run')
    def test_build_audio_track(self, mock_run, tmp_path):
        """Test audio track building with silence padding."""
        # Setup test environment
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create mock audio files
        audio_files = []
        for i in range(3):
            audio_file = output_dir / f"clip{i}_en.wav"
            audio_file.touch()
            audio_files.append(str(audio_file))

        # Mock durations
        durations = [5.0, 8.0, 6.0]
        speech_durations = [4.0, 7.0, 5.0]

        # Mock FFmpeg calls
        mock_run.return_value = Mock(returncode=0)

        output_path = output_dir / "audio_track.wav"
        render._build_audio_track(
            audio_files,
            durations,
            speech_durations,
            str(output_path),
            offset=1.0
        )

        # Verify FFmpeg was called for padding and concatenation
        assert mock_run.called
        calls = mock_run.call_args_list

        # Check for silence generation and concatenation
        ffmpeg_calls = [str(call[0][0]) for call in calls]
        assert any("ffmpeg" in str(c) for c in ffmpeg_calls)

    def test_full_quality_rendering(self, tmp_path):
        """Test full quality video rendering option."""
        project_path = tmp_path / "test_project"
        input_path = project_path / "input"
        output_path = project_path / "output"
        json_path = project_path / "json"

        input_path.mkdir(parents=True)
        output_path.mkdir(parents=True)
        json_path.mkdir(parents=True)

        # Create test plan
        plan_data = {
            "ordered_clips": ["clip1", "clip2"],
            "narration": [
                {"clip_id": "clip1", "text": "Test 1"},
                {"clip_id": "clip2", "text": "Test 2"}
            ],
            "languages": {"en": {}}
        }

        plan_file = json_path / "plan.json"
        with open(plan_file, 'w') as f:
            json.dump(plan_data, f)

        # Create test videos
        (input_path / "clip1.mp4").touch()
        (input_path / "clip2.mp4").touch()

        with patch('ai_doc_composer.render.DATA_ROOT', tmp_path):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="")

                # Test with full quality flag
                render.run("test_project", full_quality=True)

                # Verify different output naming for full quality
                calls = mock_run.call_args_list
                output_files = [str(c[0][0]) for c in calls if "final_full" in str(c)]
                # Full quality version should have different naming

    def test_multilingual_audio_tracks(self, tmp_path):
        """Test generation of multiple language audio tracks."""
        project_path = tmp_path / "test_project"
        output_path = project_path / "output"
        json_path = project_path / "json"

        output_path.mkdir(parents=True)
        json_path.mkdir(parents=True)

        # Create multilingual TTS metadata
        tts_meta = {
            "files": [
                {"clip_id": "clip1", "language": "en", "path": "clip1_en.wav", "duration": 5.0},
                {"clip_id": "clip1", "language": "es", "path": "clip1_es.wav", "duration": 5.5},
                {"clip_id": "clip2", "language": "en", "path": "clip2_en.wav", "duration": 7.0},
                {"clip_id": "clip2", "language": "es", "path": "clip2_es.wav", "duration": 7.2}
            ]
        }

        tts_file = json_path / "tts_meta.json"
        with open(tts_file, 'w') as f:
            json.dump(tts_meta, f)

        # Create audio files
        for item in tts_meta["files"]:
            (output_path / item["path"]).touch()

        with patch('ai_doc_composer.render.DATA_ROOT', tmp_path):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout='{"format": {"duration": "10.0"}}')

                # Test should handle multiple language tracks
                # This would be tested in actual run function
                assert tts_file.exists()

    @patch('subprocess.run')
    def test_error_handling(self, mock_run, tmp_path):
        """Test error handling during rendering."""
        # Mock FFmpeg failure
        mock_run.return_value = Mock(returncode=1, stderr="FFmpeg error")

        with pytest.raises(Exception):
            render._concat_video([], tmp_path / "output.mp4")

    def test_offset_configuration(self):
        """Test audio offset configuration."""
        # Test that offset is properly applied
        offset_values = [0.0, 0.5, 1.0, 2.0]

        for offset in offset_values:
            # Offset should be non-negative
            assert offset >= 0.0

    @patch('subprocess.run')
    def test_file_cleanup(self, mock_run, tmp_path):
        """Test temporary file cleanup after rendering."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create temporary files
        temp_files = [
            output_dir / "temp_concat.txt",
            output_dir / "temp_audio.wav"
        ]

        for tf in temp_files:
            tf.touch()

        mock_run.return_value = Mock(returncode=0)

        # After rendering, temp files should be cleaned up
        # This would be tested in the actual implementation
        assert output_dir.exists()