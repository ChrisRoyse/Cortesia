#!/usr/bin/env python3
"""
Test and Validation Suite for Advanced Confidence Engine
Validates confidence calibration accuracy and multi-factor analysis
"""

import sys
import os
import json
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_confidence_engine import AdvancedConfidenceEngine, ConfidenceLevel
from ultra_reliable_core import UniversalDocumentationDetector


@dataclass
class TestCase:
    """Test case for confidence validation"""
    content: str
    language: str
    expected_has_docs: bool
    expected_confidence_range: Tuple[float, float]
    description: str
    declaration_line: int = None


class AdvancedConfidenceEngineValidator:
    """Validator for advanced confidence engine functionality"""
    
    def __init__(self):
        self.confidence_engine = AdvancedConfidenceEngine()
        self.doc_detector = UniversalDocumentationDetector(use_advanced_confidence=True)
        
    def create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases for validation"""
        
        return [
            # High-confidence documented cases
            TestCase(
                content='''/// High-quality Rust documentation
/// This function performs complex mathematical operations
/// with comprehensive error handling and optimization.
/// 
/// # Arguments
/// * `x` - First operand (must be positive)
/// * `y` - Second operand (must be non-zero)
/// 
/// # Returns
/// Result of the mathematical operation
/// 
/// # Examples
/// ```rust
/// let result = complex_math(5.0, 2.0);
/// assert!(result > 0.0);
/// ```
/// 
/// # Panics
/// Panics if y is zero or x is negative
pub fn complex_math(x: f64, y: f64) -> f64 {
    if y == 0.0 { panic!("Division by zero"); }
    if x < 0.0 { panic!("Negative input"); }
    (x * y) + (x / y).sqrt()
}''',
                language='rust',
                expected_has_docs=True,
                expected_confidence_range=(0.85, 1.0),
                description="Comprehensive Rust function documentation",
                declaration_line=18
            ),
            
            TestCase(
                content='''class DataProcessor:
    """Advanced data processing class for machine learning pipelines.
    
    This class provides comprehensive data processing capabilities including:
    - Data cleaning and validation with statistical methods
    - Feature engineering using advanced transformations
    - Statistical analysis and comprehensive reporting
    - Integration with popular ML frameworks
    
    The processor supports multiple data formats and provides
    extensive configuration options for customization.
    
    Attributes:
        config (dict): Configuration parameters for processing
        transformers (list): List of data transformers to apply
        validators (list): List of validation rules
        stats (dict): Processing statistics and metrics
    
    Example:
        >>> processor = DataProcessor({'normalize': True})
        >>> cleaned_data = processor.clean(raw_data)
        >>> features = processor.extract_features(cleaned_data)
    
    Note:
        This class requires pandas and scikit-learn for full functionality.
    """
    
    def __init__(self, config: dict):
        """Initialize the data processing pipeline.
        
        Args:
            config: Configuration dictionary with processing parameters.
                   Must include 'normalize', 'handle_missing', and 'feature_types'.
        
        Raises:
            ValueError: If required configuration keys are missing.
            TypeError: If config is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")
        
        required_keys = ['normalize', 'handle_missing', 'feature_types']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        self.config = config
        self.transformers = []
        self.validators = []
        self.stats = {}''',
                language='python',
                expected_has_docs=True,
                expected_confidence_range=(0.90, 1.0),
                description="Comprehensive Python class documentation",
                declaration_line=1
            ),
            
            TestCase(
                content='''/**
 * Advanced API client for external services with comprehensive features
 * 
 * This class provides a robust HTTP client with the following capabilities:
 * - Automatic retry logic with exponential backoff
 * - Request/response logging and debugging
 * - Authentication token management
 * - Rate limiting and throttling
 * - Response caching with TTL support
 * - Request/response transformation
 * - Error handling and recovery
 * 
 * @class ApiClient
 * @example
 * const client = new ApiClient({
 *   baseUrl: 'https://api.example.com',
 *   timeout: 30000,
 *   retries: 3
 * });
 * 
 * const response = await client.get('/users/123', {
 *   headers: { 'Authorization': 'Bearer token' }
 * });
 */
class ApiClient {
    /**
     * Create a new API client instance
     * 
     * @param {Object} config - Configuration object
     * @param {string} config.baseUrl - Base URL for API requests
     * @param {number} [config.timeout=30000] - Request timeout in milliseconds
     * @param {number} [config.retries=3] - Number of retry attempts
     * @param {Object} [config.headers={}] - Default headers for all requests
     * @param {boolean} [config.cache=false] - Enable response caching
     * @throws {Error} Throws error if baseUrl is not provided
     */
    constructor(config) {
        if (!config || !config.baseUrl) {
            throw new Error('baseUrl is required in configuration');
        }
        
        this.baseUrl = config.baseUrl;
        this.timeout = config.timeout || 30000;
        this.retries = config.retries || 3;
        this.headers = config.headers || {};
        this.cache = config.cache || false;
    }
}''',
                language='javascript',
                expected_has_docs=True,
                expected_confidence_range=(0.85, 1.0),
                description="Comprehensive JavaScript class documentation",
                declaration_line=24
            ),
            
            # Medium-confidence cases
            TestCase(
                content='''/// Basic function for calculations
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}''',
                language='rust',
                expected_has_docs=True,
                expected_confidence_range=(0.4, 0.7),
                description="Basic Rust function documentation",
                declaration_line=1
            ),
            
            TestCase(
                content='''def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)''',
                language='python',
                expected_has_docs=True,
                expected_confidence_range=(0.5, 0.8),
                description="Basic Python function documentation",
                declaration_line=1
            ),
            
            # Low-confidence cases (false positives)
            TestCase(
                content='''// TODO: implement this function later
// FIXME: handle edge cases
pub fn incomplete_function() {
    // debug print for testing
    println!("not implemented");
}''',
                language='rust',
                expected_has_docs=False,
                expected_confidence_range=(0.0, 0.3),
                description="False positive - TODO/FIXME comments",
                declaration_line=2
            ),
            
            TestCase(
                content='''# Configuration values
DEBUG = True
API_KEY = "secret-123"
DATABASE_URL = "postgresql://localhost:5432/db"''',
                language='python',
                expected_has_docs=False,
                expected_confidence_range=(0.0, 0.2),
                description="No documentation - configuration values",
            ),
            
            # Undocumented cases
            TestCase(
                content='''pub struct SimpleStruct {
    field1: i32,
    field2: String,
}

impl SimpleStruct {
    pub fn new() -> Self {
        Self {
            field1: 0,
            field2: String::new(),
        }
    }
}''',
                language='rust',
                expected_has_docs=False,
                expected_confidence_range=(0.0, 0.2),
                description="Undocumented Rust struct",
                declaration_line=1
            ),
            
            TestCase(
                content='''function processData(data) {
    const results = [];
    for (let item of data) {
        results.push(item * 2);
    }
    return results;
}''',
                language='javascript',
                expected_has_docs=False,
                expected_confidence_range=(0.0, 0.2),
                description="Undocumented JavaScript function",
                declaration_line=1
            ),
            
            # Edge cases
            TestCase(
                content='''/**
 * @param data
 */
function minimalDoc(data) {
    return data;
}''',
                language='javascript',
                expected_has_docs=True,
                expected_confidence_range=(0.3, 0.6),
                description="Minimal JSDoc documentation",
                declaration_line=4
            ),
            
            TestCase(
                content='''/// Very long line of documentation that provides extensive detail about the function's purpose, parameters, return values, error conditions, performance characteristics, thread safety guarantees, memory usage patterns, and various other implementation details that developers need to understand when using this function effectively in their applications.
pub fn detailed_function() { }''',
                language='rust',
                expected_has_docs=True,
                expected_confidence_range=(0.6, 0.9),
                description="Single long documentation line",
                declaration_line=1
            ),
        ]
    
    def validate_confidence_accuracy(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Validate confidence accuracy against expected results"""
        
        print("Advanced Confidence Engine Validation")
        print("=" * 50)
        
        results = []
        correct_detections = 0
        confidence_errors = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {test_case.description}")
            print("-" * 30)
            
            # Get detection results
            detection_results = self.doc_detector.detect_documentation_multi_pass(
                test_case.content, test_case.language, test_case.declaration_line
            )
            
            has_docs = detection_results['has_documentation']
            confidence = detection_results.get('advanced_confidence', detection_results.get('confidence', 0.0))
            
            # Check detection accuracy
            detection_correct = has_docs == test_case.expected_has_docs
            if detection_correct:
                correct_detections += 1
            
            # Check confidence calibration
            min_conf, max_conf = test_case.expected_confidence_range
            confidence_correct = min_conf <= confidence <= max_conf
            
            if not confidence_correct:
                confidence_errors.append({
                    'test_case': i + 1,
                    'expected_range': test_case.expected_confidence_range,
                    'actual_confidence': confidence,
                    'description': test_case.description
                })
            
            # Print results
            print(f"  Expected docs: {test_case.expected_has_docs}")
            print(f"  Detected docs: {has_docs} ({'PASS' if detection_correct else 'FAIL'})")
            print(f"  Expected confidence: {min_conf:.2f} - {max_conf:.2f}")
            print(f"  Actual confidence: {confidence:.3f} ({'PASS' if confidence_correct else 'FAIL'})")
            
            if 'confidence_level' in detection_results:
                print(f"  Confidence level: {detection_results['confidence_level']}")
            
            if 'dominant_factors' in detection_results:
                factors = detection_results['dominant_factors'][:2]
                print(f"  Main factors: {', '.join(factors) if factors else 'none'}")
            
            if 'confidence_warnings' in detection_results and detection_results['confidence_warnings']:
                warnings = detection_results['confidence_warnings'][:2]
                print(f"  Warnings: {', '.join(warnings)}")
            
            results.append({
                'test_case': i + 1,
                'description': test_case.description,
                'expected_has_docs': test_case.expected_has_docs,
                'detected_has_docs': has_docs,
                'detection_correct': detection_correct,
                'expected_confidence_range': test_case.expected_confidence_range,
                'actual_confidence': confidence,
                'confidence_correct': confidence_correct,
                'confidence_level': detection_results.get('confidence_level', 'unknown'),
                'dominant_factors': detection_results.get('dominant_factors', []),
                'warnings': detection_results.get('confidence_warnings', [])
            })
        
        # Calculate overall metrics
        detection_accuracy = correct_detections / len(test_cases)
        confidence_accuracy = (len(test_cases) - len(confidence_errors)) / len(test_cases)
        
        print(f"\n{'='*50}")
        print("VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total test cases: {len(test_cases)}")
        print(f"Detection accuracy: {detection_accuracy:.1%} ({correct_detections}/{len(test_cases)})")
        print(f"Confidence calibration accuracy: {confidence_accuracy:.1%}")
        
        if confidence_errors:
            print(f"\nConfidence Calibration Issues:")
            for error in confidence_errors[:5]:  # Show first 5 errors
                print(f"  Test {error['test_case']}: Expected {error['expected_range']}, got {error['actual_confidence']:.3f}")
                print(f"    {error['description']}")
        
        # Calculate confidence metrics
        all_confidences = [r['actual_confidence'] for r in results]
        avg_confidence = statistics.mean(all_confidences)
        confidence_std = statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0
        
        print(f"\nConfidence Statistics:")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Confidence std dev: {confidence_std:.3f}")
        print(f"  Min confidence: {min(all_confidences):.3f}")
        print(f"  Max confidence: {max(all_confidences):.3f}")
        
        return {
            'detection_accuracy': detection_accuracy,
            'confidence_calibration_accuracy': confidence_accuracy,
            'test_results': results,
            'confidence_errors': confidence_errors,
            'confidence_stats': {
                'mean': avg_confidence,
                'std': confidence_std,
                'min': min(all_confidences),
                'max': max(all_confidences)
            }
        }
    
    def test_confidence_factors(self) -> Dict[str, Any]:
        """Test individual confidence factors"""
        
        print("\nTesting Individual Confidence Factors")
        print("=" * 40)
        
        # Test case with strong pattern matching
        rust_code = '''/// This is documented
pub fn test_function() {}'''
        
        detection_results = self.doc_detector.detect_documentation_multi_pass(
            rust_code, 'rust', declaration_line=1
        )
        
        if 'confidence_factors' in detection_results:
            factors = detection_results['confidence_factors']
            print("\nConfidence Factor Analysis:")
            print(f"  Pattern Match: {factors.get('pattern_match', 0):.3f}")
            print(f"  Semantic Richness: {factors.get('semantic_richness', 0):.3f}")
            print(f"  Context Appropriateness: {factors.get('context_appropriateness', 0):.3f}")
            print(f"  Cross Validation: {factors.get('cross_validation', 0):.3f}")
            print(f"  Language Specific: {factors.get('language_specific', 0):.3f}")
            print(f"  Quality Bonus: {factors.get('quality_bonus', 0):.3f}")
            print(f"  False Positive Penalty: {factors.get('false_positive_penalty', 0):.3f}")
            
            return factors
        else:
            print("Advanced confidence factors not available")
            return {}
    
    def test_uncertainty_quantification(self) -> Dict[str, Any]:
        """Test uncertainty quantification for ambiguous cases"""
        
        print("\nTesting Uncertainty Quantification")
        print("=" * 35)
        
        # Ambiguous test case
        ambiguous_code = '''// This might be documentation
// or it might just be comments
pub fn ambiguous_function() {}'''
        
        detection_results = self.doc_detector.detect_documentation_multi_pass(
            ambiguous_code, 'rust', declaration_line=2
        )
        
        if 'uncertainty_range' in detection_results:
            lower, upper = detection_results['uncertainty_range']
            confidence = detection_results.get('advanced_confidence', 0)
            uncertainty = upper - lower
            
            print(f"Confidence: {confidence:.3f}")
            print(f"Uncertainty range: {lower:.3f} - {upper:.3f}")
            print(f"Uncertainty magnitude: {uncertainty:.3f}")
            
            return {
                'confidence': confidence,
                'uncertainty_range': (lower, upper),
                'uncertainty_magnitude': uncertainty
            }
        else:
            print("Uncertainty information not available")
            return {}
    
    def benchmark_performance(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark performance of advanced confidence engine"""
        
        print(f"\nBenchmarking Performance ({iterations} iterations)")
        print("=" * 30)
        
        test_code = '''/// Example function with documentation
/// Performs basic arithmetic operations
pub fn benchmark_function(x: i32, y: i32) -> i32 {
    x + y
}'''
        
        # Time basic detection
        start_time = time.time()
        basic_detector = UniversalDocumentationDetector(use_advanced_confidence=False)
        for _ in range(iterations):
            basic_detector.detect_documentation_multi_pass(test_code, 'rust', 2)
        basic_time = time.time() - start_time
        
        # Time advanced detection
        start_time = time.time()
        for _ in range(iterations):
            self.doc_detector.detect_documentation_multi_pass(test_code, 'rust', 2)
        advanced_time = time.time() - start_time
        
        # Calculate metrics
        basic_per_sec = iterations / basic_time if basic_time > 0 else 0
        advanced_per_sec = iterations / advanced_time if advanced_time > 0 else 0
        overhead = ((advanced_time - basic_time) / basic_time * 100) if basic_time > 0 else 0
        
        print(f"Basic detection: {basic_time:.3f}s ({basic_per_sec:.1f} ops/sec)")
        print(f"Advanced detection: {advanced_time:.3f}s ({advanced_per_sec:.1f} ops/sec)")
        print(f"Performance overhead: {overhead:.1f}%")
        
        return {
            'basic_time': basic_time,
            'advanced_time': advanced_time,
            'basic_ops_per_sec': basic_per_sec,
            'advanced_ops_per_sec': advanced_per_sec,
            'overhead_percentage': overhead
        }


def main():
    """Main validation function"""
    validator = AdvancedConfidenceEngineValidator()
    
    print("ADVANCED CONFIDENCE ENGINE VALIDATION SUITE")
    print("=" * 60)
    print("Testing sophisticated confidence analysis and calibration")
    print("=" * 60)
    
    # Create and run test cases
    test_cases = validator.create_test_cases()
    validation_results = validator.validate_confidence_accuracy(test_cases)
    
    # Test individual components
    factor_results = validator.test_confidence_factors()
    uncertainty_results = validator.test_uncertainty_quantification()
    performance_results = validator.benchmark_performance()
    
    # Final assessment
    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    
    detection_acc = validation_results['detection_accuracy']
    confidence_acc = validation_results['confidence_calibration_accuracy']
    overhead = performance_results['overhead_percentage']
    
    print(f"Detection Accuracy: {detection_acc:.1%}")
    print(f"Confidence Calibration: {confidence_acc:.1%}")
    print(f"Performance Overhead: {overhead:.1f}%")
    
    # Success criteria
    success_criteria = {
        'detection_accuracy': detection_acc >= 0.90,  # 90%+ detection accuracy
        'confidence_calibration': confidence_acc >= 0.70,  # 70%+ confidence calibration
        'performance_overhead': overhead <= 50.0,  # <50% performance overhead
    }
    
    all_passed = all(success_criteria.values())
    
    print(f"\nSuccess Criteria:")
    for criterion, passed in success_criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {criterion.replace('_', ' ').title()}: {status}")
    
    if all_passed:
        print(f"\nADVANCED CONFIDENCE ENGINE VALIDATION PASSED!")
        print("   System is ready for production use with sophisticated confidence analysis.")
    else:
        print(f"\nSome validation criteria need improvement.")
        print("   Consider tuning confidence factors or calibration parameters.")
    
    # Save detailed results
    results_file = 'advanced_confidence_validation_results.json'
    detailed_results = {
        'validation_results': validation_results,
        'factor_analysis': factor_results,
        'uncertainty_analysis': uncertainty_results,
        'performance_benchmark': performance_results,
        'success_criteria': success_criteria,
        'overall_success': all_passed
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)