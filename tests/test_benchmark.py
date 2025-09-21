"""Test performance benchmarking functionality."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_doc_composer.benchmark import benchmark_project, PerformanceBenchmarker

def test_benchmark_project_switzerland():
    """Test benchmarking on Switzerland dataset."""
    print("🚀 Testing performance benchmarking system...")
    
    report = benchmark_project("switzerland", save_report=True)
    
    # Basic assertions
    assert report.project_name == "switzerland"
    assert report.total_clips >= 0
    assert report.total_processing_time >= 0
    assert len(report.stage_performance) >= 0
    assert isinstance(report.scaling_metrics, dict)
    assert isinstance(report.provider_comparison, dict)
    assert isinstance(report.system_info, dict)
    
    print(f"✅ Benchmark test passed!")
    print(f"🏁 Total Processing Time: {report.total_processing_time:.2f}s")
    print(f"📊 Clips: {report.total_clips}")

def test_performance_benchmarker_components():
    """Test individual benchmarker components."""
    benchmarker = PerformanceBenchmarker()
    project_path = Path("projects/switzerland")
    
    if project_path.exists():
        # Test system info collection
        system_info = benchmarker.get_system_info()
        assert isinstance(system_info, dict)
        assert "cpu_count" in system_info
        assert "memory_total_gb" in system_info
        assert "platform" in system_info
        
        # Test scaling metrics
        scaling = benchmarker.calculate_scaling_metrics(project_path)
        assert isinstance(scaling, dict)
        
        print("✅ Benchmarker components test passed!")

def test_benchmark_context_manager():
    """Test the performance monitoring context manager."""
    benchmarker = PerformanceBenchmarker()
    
    # Test context manager
    with benchmarker.monitor_stage("test_stage", "test_provider") as ctx:
        ctx['items_processed'] = 5
        # Simulate some processing
        import time
        time.sleep(0.01)
    
    # Check that stage was recorded
    assert len(benchmarker.stage_results) == 1
    stage = benchmarker.stage_results[0]
    assert stage.stage_name == "test_stage"
    assert stage.provider_used == "test_provider"
    assert stage.items_processed == 5
    assert stage.success is True
    assert stage.processing_time > 0
    
    print("✅ Context manager test passed!")

if __name__ == "__main__":
    # Run tests directly
    test_benchmark_project_switzerland()
    test_performance_benchmarker_components()
    test_benchmark_context_manager()