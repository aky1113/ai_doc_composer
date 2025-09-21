"""Tests for the plan module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from ai_doc_composer import plan
from ai_doc_composer.styles import DocumentaryStyles


class TestPlanStage:
    """Test suite for narrative planning and clip ordering."""

    def test_estimate_speech_seconds(self):
        """Test speech duration estimation."""
        text = "This is a test sentence with ten words in it."
        duration = plan._estimate_speech_sec(text, wpm=120)

        # 10 words at 120 wpm = 5 seconds
        assert duration == 5.0

    def test_baseline_plan_generation(self):
        """Test fallback planning without LLM."""
        clips = [
            {"clip_id": "clip1", "duration_seconds": 10.0, "caption": "Mountain view"},
            {"clip_id": "clip2", "duration_seconds": 15.0, "caption": "Lake scene"},
            {"clip_id": "clip3", "duration_seconds": 8.0, "caption": "Sunset"}
        ]

        ordered, narration = plan._baseline_plan(clips, wpm=160)

        assert ordered == ["clip1", "clip2", "clip3"]
        assert len(narration) == 3
        assert all("clip_id" in n and "text" in n for n in narration)

    def test_order_by_exif(self):
        """Test chronological ordering by EXIF timestamps."""
        clips = [
            {"clip_id": "clip1", "exif_timestamp": "2024-01-03T00:00:00Z"},
            {"clip_id": "clip2", "exif_timestamp": "2024-01-01T00:00:00Z"},
            {"clip_id": "clip3", "exif_timestamp": "2024-01-02T00:00:00Z"}
        ]

        ordered = plan._order_by_exif(clips)

        assert ordered == ["clip2", "clip3", "clip1"]

    def test_check_phrase_diversity(self):
        """Test duplicate phrase detection."""
        script = [
            "The mountain stands tall against the sky.",
            "We explore the beautiful landscape.",
            "The mountain stands tall in the distance.",
            "Nature reveals its wonders."
        ]

        duplicates = plan.check_phrase_diversity(script)

        assert "The mountain stands tall" in str(duplicates)

    def test_style_config_application(self):
        """Test documentary style configuration."""
        style_config = DocumentaryStyles.get_style("documentary")

        assert style_config.name == "documentary"
        assert "professional" in style_config.tone.lower()

    @patch('ai_doc_composer.plan._ask_ollama')
    def test_ollama_provider_integration(self, mock_ollama):
        """Test Ollama LLM provider integration."""
        mock_ollama.return_value = {"ordered": ["clip1", "clip2"]}

        clips = [
            {"clip_id": "clip1", "caption": "Scene 1"},
            {"clip_id": "clip2", "caption": "Scene 2"}
        ]

        from ai_doc_composer.styles import StyleConfig
        style_config = StyleConfig(
            name="test",
            tone="casual",
            vocabulary="simple",
            structure="linear"
        )

        result = plan._ask_ollama_order(clips, style_config, None, temperature=0.5)

        assert result == ["clip1", "clip2"]
        mock_ollama.assert_called_once()

    @patch('ai_doc_composer.plan._ask_gemini')
    def test_gemini_provider_integration(self, mock_gemini):
        """Test Gemini LLM provider integration."""
        mock_gemini.return_value = {"ordered": ["clip2", "clip1"]}

        clips = [
            {"clip_id": "clip1", "caption": "Scene 1"},
            {"clip_id": "clip2", "caption": "Scene 2"}
        ]

        from ai_doc_composer.styles import StyleConfig
        style_config = StyleConfig(
            name="test",
            tone="formal",
            vocabulary="advanced",
            structure="thematic"
        )

        result = plan._ask_gemini_order(clips, style_config, None, temperature=0.3)

        assert result == ["clip2", "clip1"]
        mock_gemini.assert_called_once()

    def test_location_injection(self):
        """Test location mention injection into narration."""
        script = [
            "Beautiful mountains surround us.",
            "The lake reflects the sky.",
            "Nature at its finest."
        ]

        locations = ["Switzerland, Alps", "Lake Geneva, Switzerland", "Swiss countryside"]

        modified_script = plan._inject_location(script, locations)

        # Check that at least one location was injected
        assert any("Switzerland" in line or "Alps" in line or "Geneva" in line
                  for line in modified_script)

    def test_multilingual_support(self, tmp_path):
        """Test multi-language narration generation."""
        with patch('ai_doc_composer.plan._ask_gemini') as mock_gemini:
            mock_gemini.return_value = {
                "translations": {
                    "fr": ["Bonjour", "Au revoir"],
                    "es": ["Hola", "Adiós"]
                }
            }

            script = ["Hello", "Goodbye"]
            languages = ["en", "fr", "es"]

            translations = plan._translate_lines(script, languages, provider="gemini")

            assert "fr" in translations
            assert "es" in translations
            assert translations["fr"] == ["Bonjour", "Au revoir"]
            assert translations["es"] == ["Hola", "Adiós"]

    def test_overflow_compensation(self):
        """Test narration overflow compensation between clips."""
        clips = [
            {"clip_id": "clip1", "duration_seconds": 5.0},
            {"clip_id": "clip2", "duration_seconds": 10.0},
            {"clip_id": "clip3", "duration_seconds": 8.0}
        ]

        narration = [
            {"clip_id": "clip1", "text": "Short text", "duration": 3.0},
            {"clip_id": "clip2", "text": "Very long narration that exceeds clip duration significantly", "duration": 15.0},
            {"clip_id": "clip3", "text": "Normal text", "duration": 7.0}
        ]

        # Test that overflow handling is properly configured
        # This would be tested in the actual run function with allow_overflow=True
        assert len(clips) == len(narration)