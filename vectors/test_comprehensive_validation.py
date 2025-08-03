#!/usr/bin/env python3
"""
Test Script for Comprehensive Validation Framework

This script demonstrates and validates the comprehensive validation framework
functionality, ensuring it correctly identifies quality issues and provides
accurate monitoring capabilities.
"""

import json
import time
import traceback
from pathlib import Path

# Import the framework
from comprehensive_validation_framework import (
    ComprehensiveValidationFramework,
    GroundTruthExample,
    run_quick_validation,
    validate_against_custom_examples
)


def test_ground_truth_manager():
    """Test ground truth management functionality"""
    print("Testing Ground Truth Manager...")
    
    framework = ComprehensiveValidationFramework()
    gt_manager = framework.ground_truth_manager
    
    # Test getting datasets
    rust_examples = gt_manager.get_dataset('rust_examples')
    python_examples = gt_manager.get_dataset('python_examples')
    edge_cases = gt_manager.get_dataset('edge_cases')
    
    print(f"  - Rust examples: {len(rust_examples)}")
    print(f"  - Python examples: {len(python_examples)}")
    print(f"  - Edge cases: {len(edge_cases)}")
    
    # Test adding custom example
    custom_example = GroundTruthExample(
        content='''/**
 * Test function for validation
 * @param x The input value
 * @returns The processed result
 */
function testValidation(x) {
    return x * 2;
}''',
        language='javascript',
        has_documentation=True,
        declaration_line=5,
        documentation_lines=[0, 1, 2, 3, 4],
        difficulty_level='test'
    )
    
    gt_manager.add_example(custom_example, 'test_dataset')
    test_examples = gt_manager.get_dataset('test_dataset')
    print(f"  - Test examples added: {len(test_examples)}")
    
    # Verify example was added correctly
    assert len(test_examples) == 1
    assert test_examples[0].language == 'javascript'
    assert test_examples[0].has_documentation == True
    
    print("  [PASS] Ground Truth Manager tests passed")


def test_quality_metrics_calculator():
    """Test quality metrics calculation"""
    print("Testing Quality Metrics Calculator...")
    
    framework = ComprehensiveValidationFramework()
    calculator = framework.metrics_calculator
    
    # Create test data
    ground_truth = [
        GroundTruthExample("test1", "python", True, 0, [0, 1], difficulty_level="easy"),
        GroundTruthExample("test2", "python", False, 0, [], difficulty_level="easy"),
        GroundTruthExample("test3", "rust", True, 0, [0, 1, 2], difficulty_level="medium"),
        GroundTruthExample("test4", "rust", False, 0, [], difficulty_level="hard"),
    ]
    
    predictions = [
        {'has_documentation': True, 'confidence': 0.9, 'processing_time': 0.01},   # TP
        {'has_documentation': False, 'confidence': 0.1, 'processing_time': 0.005}, # TN
        {'has_documentation': True, 'confidence': 0.8, 'processing_time': 0.015},  # TP
        {'has_documentation': True, 'confidence': 0.6, 'processing_time': 0.008},  # FP
    ]
    
    # Calculate metrics
    metrics = calculator.calculate_comprehensive_metrics(predictions, ground_truth)
    
    # Verify calculations
    print(f"  - Accuracy: {metrics.accuracy:.3f}")
    print(f"  - Precision: {metrics.precision:.3f}")
    print(f"  - Recall: {metrics.recall:.3f}")
    print(f"  - F1 Score: {metrics.f1_score:.3f}")
    print(f"  - Calibration Error: {metrics.calibration_error:.3f}")
    
    # Expected values
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.accuracy == 0.75  # (2+1)/4
    assert metrics.precision == 2/3   # 2/(2+1)
    assert metrics.recall == 1.0      # 2/(2+0)
    
    print("  [PASS] Quality Metrics Calculator tests passed")


def test_regression_detector():
    """Test regression detection functionality"""
    print("Testing Regression Detector...")
    
    framework = ComprehensiveValidationFramework()
    detector = framework.regression_detector
    
    # Create baseline metrics
    from comprehensive_validation_framework import ValidationMetrics
    baseline_metrics = ValidationMetrics(
        total_examples=100,
        correct_predictions=95,
        accuracy=0.95,
        true_positives=45,
        true_negatives=50,
        false_positives=3,
        false_negatives=2,
        precision=0.94,
        recall=0.96,
        f1_score=0.95,
        specificity=0.94,
        calibration_error=0.05,
        reliability_correlation=0.85,
        overconfidence_rate=0.02,
        underconfidence_rate=0.03,
        avg_processing_time=0.01,
        throughput_items_per_sec=100.0,
        memory_usage_mb=256.0,
        high_confidence_accuracy=0.98,
        medium_confidence_accuracy=0.92,
        low_confidence_accuracy=0.80
    )
    
    # Create degraded metrics
    current_metrics = ValidationMetrics(
        total_examples=100,
        correct_predictions=85,  # Degraded
        accuracy=0.85,           # Degraded
        true_positives=40,
        true_negatives=45,
        false_positives=8,       # Increased
        false_negatives=7,       # Increased
        precision=0.83,          # Degraded
        recall=0.85,             # Degraded
        f1_score=0.84,           # Degraded
        specificity=0.85,
        calibration_error=0.12,  # Degraded
        reliability_correlation=0.75,
        overconfidence_rate=0.08,
        underconfidence_rate=0.07,
        avg_processing_time=0.025,  # Degraded
        throughput_items_per_sec=40.0,
        memory_usage_mb=512.0,      # Degraded
        high_confidence_accuracy=0.95,
        medium_confidence_accuracy=0.85,
        low_confidence_accuracy=0.70
    )
    
    # Detect regressions
    alerts = detector.detect_regressions(current_metrics, baseline_metrics)
    
    print(f"  - Regression alerts found: {len(alerts)}")
    for alert in alerts:
        print(f"    {alert.severity}: {alert.metric_name} - {alert.description}")
    
    # Should detect multiple regressions
    assert len(alerts) > 0
    
    # Check for specific expected alerts
    alert_metrics = [alert.metric_name for alert in alerts]
    assert 'accuracy' in alert_metrics
    assert 'avg_processing_time' in alert_metrics
    
    print("  [PASS] Regression Detector tests passed")


def test_comprehensive_validation():
    """Test full comprehensive validation pipeline"""
    print("Testing Comprehensive Validation Pipeline...")
    
    try:
        # Run a small validation
        framework = ComprehensiveValidationFramework(enable_advanced_confidence=True)
        
        # Use only a subset for faster testing
        result = framework.run_comprehensive_validation(
            dataset_names=['rust_examples'], 
            include_performance_benchmark=False
        )
        
        print(f"  - Validated {result.dataset_size} examples")
        print(f"  - Overall accuracy: {result.validation_metrics.accuracy:.1%}")
        print(f"  - F1 Score: {result.validation_metrics.f1_score:.3f}")
        print(f"  - Performance rating: {result.performance_rating}")
        print(f"  - Regression alerts: {len(result.regression_alerts)}")
        
        # Verify results structure
        assert result.dataset_size > 0
        assert 0.0 <= result.validation_metrics.accuracy <= 1.0
        assert 0.0 <= result.validation_metrics.f1_score <= 1.0
        assert result.performance_rating in ["Excellent", "Good", "Fair", "Poor", "Critical"]
        
        # Test report generation
        report = framework.generate_validation_report(result)
        assert len(report) > 100  # Should be a substantial report
        assert "COMPREHENSIVE VALIDATION FRAMEWORK REPORT" in report
        
        print("  [PASS] Comprehensive Validation Pipeline tests passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Comprehensive Validation Pipeline test failed: {e}")
        print(traceback.format_exc())
        return False


def test_custom_examples_validation():
    """Test validation with custom examples"""
    print("Testing Custom Examples Validation...")
    
    # Create custom examples that test specific edge cases
    custom_examples = [
        # Well-documented Python function - should be detected
        GroundTruthExample(
            content='''def well_documented_function(x, y):
    """
    Add two numbers together.
    
    Args:
        x (int): First number
        y (int): Second number
        
    Returns:
        int: Sum of x and y
    """
    return x + y''',
            language='python',
            has_documentation=True,
            declaration_line=0,
            documentation_lines=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            difficulty_level='easy',
            annotation_notes='Well-documented Python function with full docstring'
        ),
        
        # Undocumented function - should not be detected
        GroundTruthExample(
            content='''def simple_function(a, b):
    return a * b''',
            language='python',
            has_documentation=False,
            declaration_line=0,
            documentation_lines=[],
            difficulty_level='easy',
            annotation_notes='Simple undocumented function'
        ),
        
        # False positive case - TODO comment should not count
        GroundTruthExample(
            content='''# TODO: Add proper documentation
def todo_function():
    pass''',
            language='python',
            has_documentation=False,
            declaration_line=1,
            documentation_lines=[],
            difficulty_level='hard',
            annotation_notes='TODO comment should not be considered documentation'
        ),
        
        # Rust documentation example
        GroundTruthExample(
            content='''/// Calculate factorial recursively
/// 
/// # Arguments
/// * `n` - Non-negative integer
/// 
/// # Returns
/// Factorial of n
pub fn factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n * factorial(n - 1),
    }
}''',
            language='rust',
            has_documentation=True,
            declaration_line=7,
            documentation_lines=[0, 1, 2, 3, 4, 5, 6],
            difficulty_level='medium',
            annotation_notes='Rust function with comprehensive documentation'
        )
    ]
    
    try:
        # Run validation on custom examples
        result = validate_against_custom_examples(custom_examples)
        
        print(f"  - Validated {result.dataset_size} custom examples")
        print(f"  - Accuracy: {result.validation_metrics.accuracy:.1%}")
        print(f"  - Precision: {result.validation_metrics.precision:.3f}")
        print(f"  - Recall: {result.validation_metrics.recall:.3f}")
        
        # Analyze results
        detailed = result.detailed_results
        false_positives = detailed.get('false_positives', [])
        false_negatives = detailed.get('false_negatives', [])
        
        print(f"  - False positives: {len(false_positives)}")
        print(f"  - False negatives: {len(false_negatives)}")
        
        if false_positives:
            print("    False positive cases:")
            for fp in false_positives:
                print(f"      - {fp.get('annotation_notes', 'No notes')}")
        
        if false_negatives:
            print("    False negative cases:")
            for fn in false_negatives:
                print(f"      - {fn.get('annotation_notes', 'No notes')}")
        
        # Expected: Should correctly identify documented vs undocumented
        # Perfect score not expected due to complexity of edge cases
        assert result.validation_metrics.accuracy >= 0.5  # At least 50% accuracy
        
        print("  [PASS] Custom Examples Validation tests passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Custom Examples Validation test failed: {e}")
        print(traceback.format_exc())
        return False


def test_performance_benchmarking():
    """Test performance benchmarking functionality"""
    print("Testing Performance Benchmarking...")
    
    try:
        framework = ComprehensiveValidationFramework()
        
        # Run with performance benchmarking enabled
        result = framework.run_comprehensive_validation(
            dataset_names=['python_examples'],  # Use smaller dataset
            include_performance_benchmark=True
        )
        
        # Check performance benchmark results
        perf_results = result.detailed_results.get('performance_benchmark', {})
        print(f"  - Benchmark configurations tested: {len(perf_results)}")
        
        for config_name, config_results in perf_results.items():
            batch_size = config_results['batch_size']
            throughput = config_results['throughput_items_per_sec']
            avg_time = config_results['avg_time_per_item'] * 1000  # ms
            
            print(f"    {config_name}: {throughput:.1f} items/sec, {avg_time:.2f}ms/item")
        
        # Verify performance metrics
        assert len(perf_results) > 0
        assert result.validation_metrics.avg_processing_time > 0
        assert result.validation_metrics.throughput_items_per_sec > 0
        
        print("  [PASS] Performance Benchmarking tests passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Performance Benchmarking test failed: {e}")
        print(traceback.format_exc())
        return False


def test_report_generation():
    """Test validation report generation"""
    print("Testing Report Generation...")
    
    try:
        # Run quick validation
        result = run_quick_validation(include_performance=False)
        
        # Generate report
        framework = ComprehensiveValidationFramework()
        report = framework.generate_validation_report(result)
        
        print(f"  - Report length: {len(report)} characters")
        
        # Verify report contains expected sections
        expected_sections = [
            "COMPREHENSIVE VALIDATION FRAMEWORK REPORT",
            "OVERALL METRICS",
            "CONFIDENCE CALIBRATION", 
            "PERFORMANCE METRICS",
            "CLASSIFICATION BREAKDOWN",
            "ACCURACY BY CONFIDENCE LEVEL"
        ]
        
        for section in expected_sections:
            assert section in report, f"Missing section: {section}"
        
        # Save report to file for manual inspection
        report_file = Path("test_validation_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"  - Sample report saved to: {report_file}")
        print("  [PASS] Report Generation tests passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Report Generation test failed: {e}")
        print(traceback.format_exc())
        return False


def test_framework_accuracy():
    """Test that the framework accurately detects validation issues"""
    print("Testing Framework Accuracy...")
    
    try:
        # Create examples with known issues that should be detected
        problematic_examples = [
            # Example that should trigger false positive (system detects docs where there aren't any)
            GroundTruthExample(
                content='''// Just a regular comment
def no_docs_function():
    return 42''',
                language='python',
                has_documentation=False,
                declaration_line=1,
                documentation_lines=[],
                difficulty_level='medium',
                annotation_notes='Regular comment should not be documentation'
            ),
            
            # Example that should trigger false negative (system misses actual docs)
            GroundTruthExample(
                content='''"""
This is comprehensive documentation for the function.
It explains what the function does in detail.
"""
def well_documented_function():
    return "documented"''',
                language='python',
                has_documentation=True,
                declaration_line=4,
                documentation_lines=[0, 1, 2, 3],
                difficulty_level='easy',
                annotation_notes='Clear docstring should be detected'
            )
        ]
        
        # Run validation
        result = validate_against_custom_examples(problematic_examples)
        
        # Analyze accuracy
        accuracy = result.validation_metrics.accuracy
        print(f"  - Accuracy on test cases: {accuracy:.1%}")
        
        # Check if framework detected any issues correctly
        false_positives = result.detailed_results.get('false_positives', [])
        false_negatives = result.detailed_results.get('false_negatives', [])
        
        print(f"  - False positives detected: {len(false_positives)}")
        print(f"  - False negatives detected: {len(false_negatives)}")
        
        # The framework should be able to handle these cases reasonably well
        # Perfect accuracy not expected due to complexity, but should be reasonable
        assert accuracy >= 0.4  # At least 40% accuracy on challenging cases
        
        print("  [PASS] Framework Accuracy tests passed")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Framework Accuracy test failed: {e}")
        print(traceback.format_exc())
        return False


def run_all_tests():
    """Run all validation framework tests"""
    print("COMPREHENSIVE VALIDATION FRAMEWORK TESTS")
    print("=" * 60)
    
    tests = [
        test_ground_truth_manager,
        test_quality_metrics_calculator,
        test_regression_detector,
        test_comprehensive_validation,
        test_custom_examples_validation,
        test_performance_benchmarking,
        test_report_generation,
        test_framework_accuracy
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print()
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("*** All tests passed! Comprehensive Validation Framework is ready. ***")
        return True
    else:
        print("*** Some tests failed. Please review the issues above. ***")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("COMPREHENSIVE VALIDATION FRAMEWORK READY FOR PRODUCTION")
        print("=" * 60)
        print()
        print("The framework provides:")
        print("* Ground truth management with curated datasets")
        print("* Comprehensive quality metrics (precision, recall, F1, calibration)")
        print("* Performance benchmarking and monitoring")
        print("* Automated regression detection")
        print("* Detailed analysis and reporting")
        print("* Continuous monitoring capabilities")
        print()
        print("Usage:")
        print("  from comprehensive_validation_framework import run_quick_validation")
        print("  result = run_quick_validation()")
        print("  print(f'Accuracy: {result.validation_metrics.accuracy:.1%}')")
    else:
        exit(1)