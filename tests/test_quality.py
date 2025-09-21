#!/usr/bin/env python3
"""Test script for practical quality validation."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_doc_composer.quality import validate_project

def test_quality_validation_switzerland():
    """Test quality validation on Switzerland dataset."""
    print("🧪 Testing practical quality validation system...")
    
    report = validate_project(
        project="switzerland",
        target_wpm=150,
        languages="en,es"
    )
    
    # Basic assertions
    assert report.overall_score >= 0.0
    assert report.pipeline_completeness >= 0.0
    assert isinstance(report.audio_sync_scores, dict)
    assert isinstance(report.file_integrity, dict)
    assert isinstance(report.performance_metrics, dict)
    
    print(f"✅ Quality validation test passed!")
    print(f"📊 Overall Score: {report.overall_score:.3f}")
    print(f"🔧 Pipeline Completeness: {report.pipeline_completeness:.3f}")

def test_quality_validation_components():
    """Test individual quality validation components."""
    from ai_doc_composer.quality import PipelineValidator
    
    validator = PipelineValidator()
    project_path = Path("projects/switzerland")
    
    if project_path.exists():
        # Test file integrity validation
        integrity = validator.validate_file_integrity(project_path)
        assert isinstance(integrity, dict)
        assert len(integrity) > 0
        
        # Test performance benchmarking
        performance = validator.benchmark_performance(project_path)
        assert isinstance(performance, dict)
        assert "clips_processed" in performance
        
        print("✅ Quality validation components test passed!")

if __name__ == "__main__":
    # Run tests directly
    test_quality_validation_switzerland()
    test_quality_validation_components()