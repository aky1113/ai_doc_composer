"""Test Whisper-based validation system (requires optional dependencies)."""

import sys
from pathlib import Path
import pytest

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_validation_module_import():
    """Test that validation module can be imported."""
    from ai_doc_composer.validate import ValidationMetrics, QualityValidator
    
    # Test basic instantiation
    validator = QualityValidator("tiny")
    assert validator.whisper_model_name == "tiny"
    assert validator.whisper_model is None  # Should be lazy-loaded
    
    print("✅ Validation module import test passed!")

@pytest.mark.skipif(True, reason="Requires Whisper dependencies which have compatibility issues")
def test_whisper_validation_switzerland():
    """Test Whisper WER validation on Switzerland dataset (requires Whisper)."""
    from ai_doc_composer.validate import run
    
    print("🧪 Testing Whisper WER validation system...")
    
    # Run validation on Switzerland dataset
    metrics = run(
        project="switzerland",  
        languages="en,es",
        whisper_model="tiny",  # Use smallest model for speed
        output_report=True
    )
    
    # Basic assertions
    assert isinstance(metrics.wer_scores, dict)
    assert isinstance(metrics.duration_accuracy, dict)
    assert metrics.overall_quality_score >= 0.0
    
    print(f"✅ Whisper validation test passed!")
    print(f"📊 Overall Quality Score: {metrics.overall_quality_score:.3f}")

def test_validation_performance_benchmark():
    """Test performance benchmarking component."""
    from ai_doc_composer.validate import QualityValidator
    
    validator = QualityValidator()
    project_path = Path("projects/switzerland")
    
    if project_path.exists():
        performance = validator.benchmark_performance(project_path)
        
        assert isinstance(performance, dict)
        assert "total_clips" in performance
        assert "memory_peak_mb" in performance
        assert performance["total_clips"] >= 0
        
        print("✅ Performance benchmark test passed!")

if __name__ == "__main__":
    # Run tests directly
    test_validation_module_import()
    test_validation_performance_benchmark()
    print("✅ Validation tests completed (Whisper tests skipped due to dependencies)")