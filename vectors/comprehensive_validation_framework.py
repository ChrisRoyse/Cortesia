#!/usr/bin/env python3
"""
Comprehensive Validation Framework for Documentation Detection System

This framework provides ground truth comparison, continuous quality monitoring,
and comprehensive validation capabilities for the advanced documentation
detection system with confidence scoring.

Features:
1. Ground Truth Management - Curated datasets with manual labels
2. Quality Metrics Suite - Precision, Recall, F1, Calibration metrics
3. Performance Benchmarking - Speed, memory, scalability testing
4. Automated Testing Pipeline - Continuous integration validation
5. Regression Detection - Quality degradation monitoring
6. Analysis and Reporting - Detailed performance insights

Integrates with:
- SmartChunkerOptimized
- AdvancedConfidenceEngine  
- UniversalDocumentationDetector
"""

import json
import time
import statistics
import hashlib
import logging
import os
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Import the existing components
try:
    from smart_chunker_optimized import SmartChunkerOptimized, PerformanceMetrics
    from advanced_confidence_engine import AdvancedConfidenceEngine, ConfidenceMetrics
    from ultra_reliable_core import UniversalDocumentationDetector
except ImportError as e:
    print(f"Warning: Could not import required components: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GroundTruthExample:
    """Single ground truth example for validation"""
    content: str
    language: str
    has_documentation: bool
    declaration_line: Optional[int] = None
    documentation_lines: List[int] = field(default_factory=list)
    file_path: str = "synthetic"
    example_id: str = ""
    difficulty_level: str = "medium"  # easy, medium, hard, edge_case
    annotation_notes: str = ""
    
    def __post_init__(self):
        if not self.example_id:
            # Generate unique ID from content hash
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.example_id = f"{self.language}_{content_hash}"


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    # Basic accuracy metrics
    total_examples: int
    correct_predictions: int
    accuracy: float
    
    # Classification metrics
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    specificity: float
    
    # Confidence calibration metrics
    calibration_error: float
    reliability_correlation: float
    overconfidence_rate: float
    underconfidence_rate: float
    
    # Performance metrics
    avg_processing_time: float
    throughput_items_per_sec: float
    memory_usage_mb: float
    
    # Quality distribution
    high_confidence_accuracy: float
    medium_confidence_accuracy: float
    low_confidence_accuracy: float
    
    # Language-specific metrics
    language_accuracies: Dict[str, float] = field(default_factory=dict)
    difficulty_accuracies: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegressionAlert:
    """Alert for performance regression"""
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_percentage: float
    severity: str  # low, medium, high, critical
    timestamp: datetime
    description: str


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark results"""
    benchmark_name: str
    timestamp: datetime
    dataset_size: int
    validation_metrics: ValidationMetrics
    performance_rating: str
    regression_alerts: List[RegressionAlert] = field(default_factory=list)
    detailed_results: Dict[str, Any] = field(default_factory=dict)


class GroundTruthManager:
    """Manages curated ground truth datasets"""
    
    def __init__(self, data_directory: str = "ground_truth_data"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)
        
        # Initialize with curated examples
        self.datasets = {
            'rust_examples': self._create_rust_ground_truth(),
            'python_examples': self._create_python_ground_truth(),
            'javascript_examples': self._create_javascript_ground_truth(),
            'edge_cases': self._create_edge_case_ground_truth(),
            'real_world_samples': []  # Populated from actual codebase
        }
        
        logger.info(f"GroundTruthManager initialized with {sum(len(ds) for ds in self.datasets.values())} examples")
    
    def _create_rust_ground_truth(self) -> List[GroundTruthExample]:
        """Create curated Rust documentation examples"""
        examples = []
        
        # Example 1: Well-documented function
        examples.append(GroundTruthExample(
            content='''/// Calculates the factorial of a number
/// 
/// # Arguments
/// * `n` - A non-negative integer
/// 
/// # Returns
/// The factorial of n
/// 
/// # Examples
/// ```
/// let result = factorial(5);
/// assert_eq!(result, 120);
/// ```
pub fn factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n * factorial(n - 1),
    }
}''',
            language='rust',
            has_documentation=True,
            declaration_line=14,
            documentation_lines=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            difficulty_level='easy',
            annotation_notes='Complete Rust documentation with examples'
        ))
        
        # Example 2: Undocumented function
        examples.append(GroundTruthExample(
            content='''pub fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}''',
            language='rust',
            has_documentation=False,
            declaration_line=0,
            documentation_lines=[],
            difficulty_level='easy',
            annotation_notes='Simple undocumented function'
        ))
        
        # Example 3: Struct with documentation
        examples.append(GroundTruthExample(
            content='''/// Represents a neural network layer
/// 
/// This structure contains the weights and biases
/// for a single layer in a neural network.
pub struct NeuralLayer {
    /// Weight matrix for the layer
    weights: Vec<f64>,
    /// Bias vector for the layer  
    biases: Vec<f64>,
}''',
            language='rust',
            has_documentation=True,
            declaration_line=4,
            documentation_lines=[0, 1, 2, 3, 5, 7],
            difficulty_level='medium',
            annotation_notes='Struct with field documentation'
        ))
        
        # Example 4: False positive case (TODO comment)
        examples.append(GroundTruthExample(
            content='''// TODO: Implement error handling
pub fn process_data(data: &[u8]) -> Result<Vec<u8>, String> {
    Ok(data.to_vec())
}''',
            language='rust',
            has_documentation=False,
            declaration_line=1,
            documentation_lines=[],
            difficulty_level='hard',
            annotation_notes='TODO comment should not be considered documentation'
        ))
        
        # Example 5: Module documentation with inner docs
        examples.append(GroundTruthExample(
            content='''//! Neural network implementation module
//! 
//! This module provides core neural network functionality
//! including forward and backward propagation algorithms.

pub mod neural_core {
    // Module implementation
}''',
            language='rust',
            has_documentation=True,
            declaration_line=5,
            documentation_lines=[0, 1, 2, 3],
            difficulty_level='medium',
            annotation_notes='Module-level inner documentation'
        ))
        
        return examples
    
    def _create_python_ground_truth(self) -> List[GroundTruthExample]:
        """Create curated Python documentation examples"""
        examples = []
        
        # Example 1: Function with docstring
        examples.append(GroundTruthExample(
            content='''def calculate_mean(numbers):
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Args:
        numbers (list): A list of numeric values
        
    Returns:
        float: The arithmetic mean of the input numbers
        
    Raises:
        ValueError: If the input list is empty
        
    Examples:
        >>> calculate_mean([1, 2, 3, 4, 5])
        3.0
    """
    if not numbers:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(numbers) / len(numbers)''',
            language='python',
            has_documentation=True,
            declaration_line=0,
            documentation_lines=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            difficulty_level='easy',
            annotation_notes='Complete Python docstring with all sections'
        ))
        
        # Example 2: Class with docstring
        examples.append(GroundTruthExample(
            content='''class DataProcessor:
    """
    A class for processing and analyzing data.
    
    This class provides methods for data cleaning,
    transformation, and statistical analysis.
    """
    
    def __init__(self, data):
        self.data = data
        
    def clean_data(self):
        # Implementation here
        pass''',
            language='python',
            has_documentation=True,
            declaration_line=0,
            documentation_lines=[1, 2, 3, 4, 5, 6],
            difficulty_level='easy',
            annotation_notes='Class docstring documentation'
        ))
        
        # Example 3: Undocumented function
        examples.append(GroundTruthExample(
            content='''def helper_function(x, y):
    return x + y * 2''',
            language='python',
            has_documentation=False,
            declaration_line=0,
            documentation_lines=[],
            difficulty_level='easy',
            annotation_notes='Simple undocumented function'
        ))
        
        # Example 4: Function with only comment (not docstring)
        examples.append(GroundTruthExample(
            content='''# This is just a regular comment
def process_item(item):
    return item.upper()''',
            language='python',
            has_documentation=False,
            declaration_line=1,
            documentation_lines=[],
            difficulty_level='medium',
            annotation_notes='Regular comment is not documentation'
        ))
        
        # Example 5: Raw docstring  
        examples.append(GroundTruthExample(
            content='''def regex_matcher(pattern, text):
    r"""
    Match a regular expression pattern against text.
    
    Args:
        pattern (str): The regex pattern to match
        text (str): The text to search in
        
    Returns:
        Match object or None
        
    Note:
        This function uses raw strings for regex patterns:
        pattern = r"\\d+\\.\\d+"
    """
    import re
    return re.search(pattern, text)''',
            language='python',
            has_documentation=True,
            declaration_line=0,
            documentation_lines=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            difficulty_level='medium',
            annotation_notes='Raw docstring with regex examples'
        ))
        
        return examples
    
    def _create_javascript_ground_truth(self) -> List[GroundTruthExample]:
        """Create curated JavaScript documentation examples"""
        examples = []
        
        # Example 1: JSDoc function
        examples.append(GroundTruthExample(
            content='''/**
 * Calculates the area of a circle
 * @param {number} radius - The radius of the circle
 * @returns {number} The area of the circle
 * @example
 * const area = calculateCircleArea(5);
 * console.log(area); // 78.54
 */
function calculateCircleArea(radius) {
    return Math.PI * radius * radius;
}''',
            language='javascript',
            has_documentation=True,
            declaration_line=8,
            documentation_lines=[0, 1, 2, 3, 4, 5, 6, 7],
            difficulty_level='easy',
            annotation_notes='Complete JSDoc with params and examples'
        ))
        
        # Example 2: Undocumented arrow function
        examples.append(GroundTruthExample(
            content='''const multiply = (a, b) => a * b;''',
            language='javascript',
            has_documentation=False,
            declaration_line=0,
            documentation_lines=[],
            difficulty_level='easy',
            annotation_notes='Simple undocumented arrow function'
        ))
        
        # Example 3: Class with JSDoc
        examples.append(GroundTruthExample(
            content='''/**
 * Represents a user account
 * @class UserAccount
 */
class UserAccount {
    /**
     * Create a user account
     * @param {string} username - The username
     * @param {string} email - The email address
     */
    constructor(username, email) {
        this.username = username;
        this.email = email;
    }
}''',
            language='javascript',
            has_documentation=True,
            declaration_line=4,
            documentation_lines=[0, 1, 2, 3, 5, 6, 7, 8, 9],
            difficulty_level='medium',
            annotation_notes='Class and constructor documentation'
        ))
        
        return examples
    
    def _create_edge_case_ground_truth(self) -> List[GroundTruthExample]:
        """Create edge case examples for robust testing"""
        examples = []
        
        # Edge case 1: Mixed comment styles
        examples.append(GroundTruthExample(
            content='''// Regular comment
/// Documentation comment for the function below
/// This function does important work
pub fn mixed_comments() {
    // Internal comment
    println!("Hello");
}''',
            language='rust',
            has_documentation=True,
            declaration_line=3,
            documentation_lines=[1, 2],
            difficulty_level='edge_case',
            annotation_notes='Mixed comment styles - only /// should count'
        ))
        
        # Edge case 2: Empty docstring
        examples.append(GroundTruthExample(
            content='''def empty_docstring():
    """
    """
    pass''',
            language='python',
            has_documentation=False,  # Empty docstring should not count
            declaration_line=0,
            documentation_lines=[],
            difficulty_level='edge_case',
            annotation_notes='Empty docstring should not be considered documentation'
        ))
        
        # Edge case 3: Very long documentation
        examples.append(GroundTruthExample(
            content='''"""
This is a very comprehensive docstring that explains in great detail
what this function does, how it works, what parameters it takes,
what it returns, what exceptions it might raise, provides multiple
examples of usage, discusses performance characteristics, explains
the algorithm being used, provides references to relevant literature,
and generally contains way more information than most docstrings
would ever contain in real world scenarios.

Parameters:
    x (int): The first parameter with a very long explanation
    y (int): The second parameter with another long explanation
    
Returns:
    int: A detailed explanation of what gets returned
    
Raises:
    ValueError: When something goes wrong
    TypeError: When types don't match
    
Examples:
    >>> result = long_documented_function(1, 2)
    >>> print(result)
    3
    
    >>> result = long_documented_function(10, 20)  
    >>> print(result)
    30
    
See Also:
    other_function: Related functionality
    some_module: More information
    
Notes:
    This function has been optimized for performance
    and handles edge cases carefully.
"""
def long_documented_function(x, y):
    return x + y''',
            language='python',
            has_documentation=True,
            declaration_line=34,
            documentation_lines=list(range(0, 34)),
            difficulty_level='edge_case',
            annotation_notes='Very long comprehensive documentation'
        ))
        
        return examples
    
    def get_dataset(self, name: str) -> List[GroundTruthExample]:
        """Get a specific dataset by name"""
        return self.datasets.get(name, [])
    
    def get_all_examples(self) -> List[GroundTruthExample]:
        """Get all ground truth examples"""
        all_examples = []
        for dataset in self.datasets.values():
            all_examples.extend(dataset)
        return all_examples
    
    def add_example(self, example: GroundTruthExample, dataset_name: str = 'custom'):
        """Add a new ground truth example"""
        if dataset_name not in self.datasets:
            self.datasets[dataset_name] = []
        self.datasets[dataset_name].append(example)
    
    def save_datasets(self) -> None:
        """Save datasets to disk"""
        for name, dataset in self.datasets.items():
            file_path = self.data_directory / f"{name}.json"
            with open(file_path, 'w') as f:
                json.dump([{
                    'content': ex.content,
                    'language': ex.language,
                    'has_documentation': ex.has_documentation,
                    'declaration_line': ex.declaration_line,
                    'documentation_lines': ex.documentation_lines,
                    'file_path': ex.file_path,
                    'example_id': ex.example_id,
                    'difficulty_level': ex.difficulty_level,
                    'annotation_notes': ex.annotation_notes
                } for ex in dataset], f, indent=2)
    
    def load_datasets(self) -> None:
        """Load datasets from disk"""
        for file_path in self.data_directory.glob("*.json"):
            dataset_name = file_path.stem
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.datasets[dataset_name] = [
                        GroundTruthExample(**example) for example in data
                    ]
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")


class QualityMetricsCalculator:
    """Calculates comprehensive quality metrics"""
    
    def __init__(self):
        self.confidence_bins = 10
    
    def calculate_comprehensive_metrics(self, 
                                      predictions: List[Dict[str, Any]], 
                                      ground_truth: List[GroundTruthExample]) -> ValidationMetrics:
        """
        Calculate comprehensive validation metrics
        
        Args:
            predictions: List of prediction results from detection system
            ground_truth: List of ground truth examples
            
        Returns:
            ValidationMetrics with comprehensive analysis
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Basic classification metrics
        tp = sum(1 for pred, gt in zip(predictions, ground_truth) 
                if pred['has_documentation'] and gt.has_documentation)
        tn = sum(1 for pred, gt in zip(predictions, ground_truth) 
                if not pred['has_documentation'] and not gt.has_documentation)
        fp = sum(1 for pred, gt in zip(predictions, ground_truth) 
                if pred['has_documentation'] and not gt.has_documentation)
        fn = sum(1 for pred, gt in zip(predictions, ground_truth) 
                if not pred['has_documentation'] and gt.has_documentation)
        
        total = len(predictions)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Confidence calibration metrics
        confidences = [pred.get('confidence', 0.0) for pred in predictions]
        actual_positives = [gt.has_documentation for gt in ground_truth]
        
        calibration_error = self._calculate_expected_calibration_error(confidences, actual_positives)
        reliability_correlation = self._calculate_reliability_correlation(confidences, actual_positives)
        overconfidence_rate, underconfidence_rate = self._calculate_calibration_rates(confidences, actual_positives)
        
        # Performance metrics
        processing_times = [pred.get('processing_time', 0.0) for pred in predictions]
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
        throughput = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
        
        # Memory usage (current process)
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Quality distribution by confidence level
        high_conf_acc = self._calculate_confidence_band_accuracy(predictions, ground_truth, 0.7, 1.0)
        med_conf_acc = self._calculate_confidence_band_accuracy(predictions, ground_truth, 0.3, 0.7)
        low_conf_acc = self._calculate_confidence_band_accuracy(predictions, ground_truth, 0.0, 0.3)
        
        # Language-specific accuracies
        language_accuracies = {}
        for lang in set(gt.language for gt in ground_truth):
            lang_predictions = [pred for pred, gt in zip(predictions, ground_truth) if gt.language == lang]
            lang_ground_truth = [gt for gt in ground_truth if gt.language == lang]
            if lang_predictions:
                lang_tp = sum(1 for pred, gt in zip(lang_predictions, lang_ground_truth) 
                             if pred['has_documentation'] and gt.has_documentation)
                lang_tn = sum(1 for pred, gt in zip(lang_predictions, lang_ground_truth) 
                             if not pred['has_documentation'] and not gt.has_documentation)
                language_accuracies[lang] = (lang_tp + lang_tn) / len(lang_predictions)
        
        # Difficulty-specific accuracies
        difficulty_accuracies = {}
        for diff in set(gt.difficulty_level for gt in ground_truth):
            diff_predictions = [pred for pred, gt in zip(predictions, ground_truth) if gt.difficulty_level == diff]
            diff_ground_truth = [gt for gt in ground_truth if gt.difficulty_level == diff]
            if diff_predictions:
                diff_tp = sum(1 for pred, gt in zip(diff_predictions, diff_ground_truth) 
                             if pred['has_documentation'] and gt.has_documentation)
                diff_tn = sum(1 for pred, gt in zip(diff_predictions, diff_ground_truth) 
                             if not pred['has_documentation'] and not gt.has_documentation)
                difficulty_accuracies[diff] = (diff_tp + diff_tn) / len(diff_predictions)
        
        return ValidationMetrics(
            total_examples=total,
            correct_predictions=tp + tn,
            accuracy=accuracy,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            specificity=specificity,
            calibration_error=calibration_error,
            reliability_correlation=reliability_correlation,
            overconfidence_rate=overconfidence_rate,
            underconfidence_rate=underconfidence_rate,
            avg_processing_time=avg_processing_time,
            throughput_items_per_sec=throughput,
            memory_usage_mb=memory_usage,
            high_confidence_accuracy=high_conf_acc,
            medium_confidence_accuracy=med_conf_acc,
            low_confidence_accuracy=low_conf_acc,
            language_accuracies=language_accuracies,
            difficulty_accuracies=difficulty_accuracies
        )
    
    def _calculate_expected_calibration_error(self, confidences: List[float], 
                                            actual_positives: List[bool]) -> float:
        """Calculate Expected Calibration Error"""
        if not confidences:
            return 0.0
        
        # Create bins
        bin_boundaries = [i / self.confidence_bins for i in range(self.confidence_bins + 1)]
        ece = 0.0
        
        for i in range(self.confidence_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin = []
            for j, conf in enumerate(confidences):
                if i == self.confidence_bins - 1:  # Last bin includes upper boundary
                    in_bin.append(bin_lower <= conf <= bin_upper)
                else:
                    in_bin.append(bin_lower <= conf < bin_upper)
            
            if not any(in_bin):
                continue
                
            bin_confidences = [confidences[j] for j, in_b in enumerate(in_bin) if in_b]
            bin_actuals = [actual_positives[j] for j, in_b in enumerate(in_bin) if in_b]
            
            if bin_confidences:
                bin_accuracy = sum(bin_actuals) / len(bin_actuals)
                bin_confidence = sum(bin_confidences) / len(bin_confidences)
                bin_weight = len(bin_confidences) / len(confidences)
                
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _calculate_reliability_correlation(self, confidences: List[float], 
                                         actual_positives: List[bool]) -> float:
        """Calculate correlation between confidence and accuracy"""
        if len(confidences) < 2:
            return 0.0
        
        try:
            # Create individual accuracy scores
            individual_accuracies = []
            for conf, actual in zip(confidences, actual_positives):
                # Predict positive if confidence > 0.5
                predicted = conf > 0.5
                individual_accuracies.append(1.0 if predicted == actual else 0.0)
            
            return statistics.correlation(confidences, individual_accuracies)
        except:
            return 0.0
    
    def _calculate_calibration_rates(self, confidences: List[float], 
                                   actual_positives: List[bool]) -> Tuple[float, float]:
        """Calculate overconfidence and underconfidence rates"""
        if not confidences:
            return 0.0, 0.0
        
        overconfident_count = 0
        underconfident_count = 0
        
        for conf, actual in zip(confidences, actual_positives):
            predicted = conf > 0.5
            
            if predicted == actual:  # Correct prediction
                continue
            elif predicted and not actual:  # False positive - overconfident
                overconfident_count += 1
            elif not predicted and actual:  # False negative - underconfident
                underconfident_count += 1
        
        total = len(confidences)
        return overconfident_count / total, underconfident_count / total
    
    def _calculate_confidence_band_accuracy(self, predictions: List[Dict[str, Any]], 
                                          ground_truth: List[GroundTruthExample],
                                          min_conf: float, max_conf: float) -> float:
        """Calculate accuracy for a specific confidence band"""
        band_predictions = []
        band_ground_truth = []
        
        for pred, gt in zip(predictions, ground_truth):
            conf = pred.get('confidence', 0.0)
            if min_conf <= conf <= max_conf:
                band_predictions.append(pred)
                band_ground_truth.append(gt)
        
        if not band_predictions:
            return 0.0
        
        correct = sum(1 for pred, gt in zip(band_predictions, band_ground_truth)
                     if pred['has_documentation'] == gt.has_documentation)
        
        return correct / len(band_predictions)


class RegressionDetector:
    """Detects performance regressions"""
    
    def __init__(self, baseline_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize regression detector
        
        Args:
            baseline_thresholds: Dictionary of metric -> threshold values
        """
        self.baseline_thresholds = baseline_thresholds or {
            'accuracy': 0.95,  # 95% accuracy baseline
            'precision': 0.90,
            'recall': 0.90,
            'f1_score': 0.90,
            'calibration_error': 0.10,  # Max 10% calibration error
            'avg_processing_time': 0.01,  # Max 10ms per item
            'memory_usage_mb': 512  # Max 512MB memory
        }
        
        self.degradation_thresholds = {
            'accuracy': 0.02,      # 2% accuracy drop = alert
            'precision': 0.05,     # 5% precision drop = alert  
            'recall': 0.05,        # 5% recall drop = alert
            'f1_score': 0.05,      # 5% F1 drop = alert
            'calibration_error': 0.05,  # 5% calibration error increase = alert
            'avg_processing_time': 2.0,  # 2x processing time = alert
            'memory_usage_mb': 1.5       # 1.5x memory usage = alert
        }
    
    def detect_regressions(self, current_metrics: ValidationMetrics, 
                          baseline_metrics: Optional[ValidationMetrics] = None) -> List[RegressionAlert]:
        """
        Detect performance regressions
        
        Args:
            current_metrics: Current validation metrics
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            List of regression alerts
        """
        alerts = []
        
        # Compare against baseline metrics if available
        if baseline_metrics:
            alerts.extend(self._compare_metrics(current_metrics, baseline_metrics))
        
        # Compare against absolute thresholds
        alerts.extend(self._check_absolute_thresholds(current_metrics))
        
        return sorted(alerts, key=lambda x: self._get_severity_priority(x.severity))
    
    def _compare_metrics(self, current: ValidationMetrics, 
                        baseline: ValidationMetrics) -> List[RegressionAlert]:
        """Compare current metrics against baseline"""
        alerts = []
        
        metrics_to_check = [
            ('accuracy', current.accuracy, baseline.accuracy, False),
            ('precision', current.precision, baseline.precision, False),
            ('recall', current.recall, baseline.recall, False),
            ('f1_score', current.f1_score, baseline.f1_score, False),
            ('calibration_error', current.calibration_error, baseline.calibration_error, True),  # Higher is worse
            ('avg_processing_time', current.avg_processing_time, baseline.avg_processing_time, True),  # Higher is worse
            ('memory_usage_mb', current.memory_usage_mb, baseline.memory_usage_mb, True)  # Higher is worse
        ]
        
        for metric_name, current_val, baseline_val, higher_is_worse in metrics_to_check:
            if baseline_val == 0:
                continue  # Skip if baseline is 0
            
            if higher_is_worse:
                degradation = (current_val - baseline_val) / baseline_val
                threshold = self.degradation_thresholds.get(metric_name, 0.1)
            else:
                degradation = (baseline_val - current_val) / baseline_val
                threshold = self.degradation_thresholds.get(metric_name, 0.05)
            
            if degradation > threshold:
                severity = self._determine_severity(degradation, threshold)
                description = f"{metric_name} degraded by {degradation:.1%} (threshold: {threshold:.1%})"
                
                alerts.append(RegressionAlert(
                    metric_name=metric_name,
                    current_value=current_val,
                    baseline_value=baseline_val,
                    degradation_percentage=degradation * 100,
                    severity=severity,
                    timestamp=datetime.now(),
                    description=description
                ))
        
        return alerts
    
    def _check_absolute_thresholds(self, metrics: ValidationMetrics) -> List[RegressionAlert]:
        """Check metrics against absolute thresholds"""
        alerts = []
        
        thresholds_to_check = [
            ('accuracy', metrics.accuracy, self.baseline_thresholds['accuracy'], False),
            ('precision', metrics.precision, self.baseline_thresholds['precision'], False),
            ('recall', metrics.recall, self.baseline_thresholds['recall'], False),
            ('f1_score', metrics.f1_score, self.baseline_thresholds['f1_score'], False),
            ('calibration_error', metrics.calibration_error, self.baseline_thresholds['calibration_error'], True),
            ('avg_processing_time', metrics.avg_processing_time, self.baseline_thresholds['avg_processing_time'], True),
            ('memory_usage_mb', metrics.memory_usage_mb, self.baseline_thresholds['memory_usage_mb'], True)
        ]
        
        for metric_name, current_val, threshold, higher_is_worse in thresholds_to_check:
            failed_threshold = False
            
            if higher_is_worse and current_val > threshold:
                failed_threshold = True
                degradation = (current_val - threshold) / threshold
            elif not higher_is_worse and current_val < threshold:
                failed_threshold = True
                degradation = (threshold - current_val) / threshold
            
            if failed_threshold:
                severity = self._determine_severity(degradation, 0.1)  # 10% threshold for severity
                description = f"{metric_name} ({current_val:.3f}) failed absolute threshold ({threshold:.3f})"
                
                alerts.append(RegressionAlert(
                    metric_name=metric_name,
                    current_value=current_val,
                    baseline_value=threshold,
                    degradation_percentage=degradation * 100,
                    severity=severity,
                    timestamp=datetime.now(),
                    description=description
                ))
        
        return alerts
    
    def _determine_severity(self, degradation: float, threshold: float) -> str:
        """Determine alert severity based on degradation"""
        if degradation >= threshold * 5:  # 5x threshold
            return 'critical'
        elif degradation >= threshold * 3:  # 3x threshold
            return 'high'
        elif degradation >= threshold * 2:  # 2x threshold
            return 'medium'
        else:
            return 'low'
    
    def _get_severity_priority(self, severity: str) -> int:
        """Get numeric priority for severity"""
        priorities = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        return priorities.get(severity, 0)


class ComprehensiveValidationFramework:
    """
    Main validation framework coordinating all validation components
    """
    
    def __init__(self, 
                 enable_advanced_confidence: bool = True,
                 max_workers: int = 4,
                 results_directory: str = "validation_results"):
        """
        Initialize comprehensive validation framework
        
        Args:
            enable_advanced_confidence: Use advanced confidence engine
            max_workers: Maximum worker threads for parallel processing
            results_directory: Directory to store validation results
        """
        self.enable_advanced_confidence = enable_advanced_confidence
        self.max_workers = max_workers
        self.results_directory = Path(results_directory)
        self.results_directory.mkdir(exist_ok=True)
        
        # Initialize components
        self.ground_truth_manager = GroundTruthManager()
        self.metrics_calculator = QualityMetricsCalculator()
        self.regression_detector = RegressionDetector()
        
        # Initialize detection system components
        self.chunker = SmartChunkerOptimized(enable_parallel=True, max_workers=max_workers)
        self.doc_detector = UniversalDocumentationDetector(use_advanced_confidence=enable_advanced_confidence)
        
        if enable_advanced_confidence:
            try:
                self.confidence_engine = AdvancedConfidenceEngine()
            except Exception as e:
                logger.warning(f"Could not initialize advanced confidence engine: {e}")
                self.confidence_engine = None
        else:
            self.confidence_engine = None
        
        # Performance tracking
        self.validation_history = []
        
        logger.info(f"ComprehensiveValidationFramework initialized with {len(self.ground_truth_manager.get_all_examples())} ground truth examples")
    
    def run_comprehensive_validation(self, 
                                   dataset_names: Optional[List[str]] = None,
                                   include_performance_benchmark: bool = True) -> BenchmarkResult:
        """
        Run comprehensive validation on specified datasets
        
        Args:
            dataset_names: List of dataset names to validate (None = all datasets)
            include_performance_benchmark: Include performance benchmarking
            
        Returns:
            BenchmarkResult with comprehensive validation results
        """
        start_time = time.time()
        
        # Get ground truth examples
        if dataset_names:
            examples = []
            for name in dataset_names:
                examples.extend(self.ground_truth_manager.get_dataset(name))
        else:
            examples = self.ground_truth_manager.get_all_examples()
        
        if not examples:
            raise ValueError("No ground truth examples available for validation")
        
        logger.info(f"Starting comprehensive validation on {len(examples)} examples")
        
        # Run predictions
        predictions = self._run_predictions(examples)
        
        # Calculate comprehensive metrics
        validation_metrics = self.metrics_calculator.calculate_comprehensive_metrics(predictions, examples)
        
        # Performance benchmarking
        performance_results = {}
        if include_performance_benchmark:
            performance_results = self._run_performance_benchmark(examples)
        
        # Regression detection
        baseline_metrics = self._get_baseline_metrics()
        regression_alerts = self.regression_detector.detect_regressions(validation_metrics, baseline_metrics)
        
        # Determine performance rating
        performance_rating = self._calculate_performance_rating(validation_metrics, regression_alerts)
        
        # Create comprehensive result
        benchmark_result = BenchmarkResult(
            benchmark_name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            dataset_size=len(examples),
            validation_metrics=validation_metrics,
            performance_rating=performance_rating,
            regression_alerts=regression_alerts,
            detailed_results={
                'processing_time_seconds': time.time() - start_time,
                'examples_by_language': self._group_by_language(examples),
                'examples_by_difficulty': self._group_by_difficulty(examples),
                'false_positives': self._analyze_false_positives(predictions, examples),
                'false_negatives': self._analyze_false_negatives(predictions, examples),
                'confidence_distribution': self._analyze_confidence_distribution(predictions),
                'performance_benchmark': performance_results
            }
        )
        
        # Save results
        self._save_benchmark_result(benchmark_result)
        
        # Update validation history
        self.validation_history.append(benchmark_result)
        
        logger.info(f"Comprehensive validation completed in {time.time() - start_time:.2f}s")
        logger.info(f"Overall accuracy: {validation_metrics.accuracy:.1%}")
        logger.info(f"F1 Score: {validation_metrics.f1_score:.3f}")
        logger.info(f"Calibration Error: {validation_metrics.calibration_error:.3f}")
        logger.info(f"Performance Rating: {performance_rating}")
        
        if regression_alerts:
            logger.warning(f"Found {len(regression_alerts)} regression alerts")
            for alert in regression_alerts[:3]:  # Show top 3
                logger.warning(f"  {alert.severity.upper()}: {alert.description}")
        
        return benchmark_result
    
    def _run_predictions(self, examples: List[GroundTruthExample]) -> List[Dict[str, Any]]:
        """Run predictions on ground truth examples"""
        predictions = []
        
        for i, example in enumerate(examples):
            start_time = time.time()
            
            try:
                # Run documentation detection
                result = self.doc_detector.detect_documentation_multi_pass(
                    example.content, example.language, example.declaration_line
                )
                
                processing_time = time.time() - start_time
                
                # Create prediction record
                prediction = {
                    'example_id': example.example_id,
                    'has_documentation': result.get('has_documentation', False),
                    'confidence': result.get('confidence', 0.0),
                    'advanced_confidence': result.get('advanced_confidence'),
                    'documentation_lines': result.get('documentation_lines', []),
                    'patterns_found': result.get('patterns_found', []),
                    'detection_methods': result.get('detection_methods', []),
                    'processing_time': processing_time,
                    'language': example.language,
                    'difficulty_level': example.difficulty_level
                }
                
                # Add advanced confidence details if available
                if self.enable_advanced_confidence and 'confidence_factors' in result:
                    prediction.update({
                        'confidence_level': result.get('confidence_level'),
                        'confidence_factors': result.get('confidence_factors'),
                        'uncertainty_range': result.get('uncertainty_range'),
                        'dominant_factors': result.get('dominant_factors'),
                        'confidence_warnings': result.get('confidence_warnings')
                    })
                
                predictions.append(prediction)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(examples)} examples")
                    
            except Exception as e:
                logger.error(f"Error processing example {example.example_id}: {e}")
                # Add failed prediction
                predictions.append({
                    'example_id': example.example_id,
                    'has_documentation': False,
                    'confidence': 0.0,
                    'processing_time': time.time() - start_time,
                    'error': str(e),
                    'language': example.language,
                    'difficulty_level': example.difficulty_level
                })
        
        return predictions
    
    def _run_performance_benchmark(self, examples: List[GroundTruthExample]) -> Dict[str, Any]:
        """Run performance benchmarking"""
        logger.info("Running performance benchmark...")
        
        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100]
        benchmark_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(examples):
                continue
                
            batch_examples = examples[:batch_size]
            
            # Multiple runs for statistical significance
            run_times = []
            memory_usages = []
            
            for _ in range(3):  # 3 runs per batch size
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Run predictions
                self._run_predictions(batch_examples)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                run_times.append(end_time - start_time)
                memory_usages.append(end_memory - start_memory)
            
            benchmark_results[f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'avg_total_time': statistics.mean(run_times),
                'avg_time_per_item': statistics.mean(run_times) / batch_size,
                'throughput_items_per_sec': batch_size / statistics.mean(run_times),
                'avg_memory_delta_mb': statistics.mean(memory_usages),
                'std_total_time': statistics.stdev(run_times) if len(run_times) > 1 else 0.0
            }
        
        return benchmark_results
    
    def _get_baseline_metrics(self) -> Optional[ValidationMetrics]:
        """Get baseline metrics from previous validation runs"""
        if not self.validation_history:
            return None
        
        # Use the most recent validation as baseline
        return self.validation_history[-1].validation_metrics
    
    def _calculate_performance_rating(self, metrics: ValidationMetrics, 
                                    alerts: List[RegressionAlert]) -> str:
        """Calculate overall performance rating"""
        score = 0
        max_score = 100
        
        # Accuracy component (40 points)
        score += metrics.accuracy * 40
        
        # F1 Score component (20 points)
        score += metrics.f1_score * 20
        
        # Calibration component (20 points) - lower error is better
        calibration_score = max(0, 20 - (metrics.calibration_error * 100))
        score += calibration_score
        
        # Performance component (20 points)
        if metrics.avg_processing_time < 0.01:  # < 10ms
            score += 20
        elif metrics.avg_processing_time < 0.05:  # < 50ms
            score += 15
        elif metrics.avg_processing_time < 0.1:  # < 100ms
            score += 10
        else:
            score += 5
        
        # Regression penalty
        critical_alerts = sum(1 for alert in alerts if alert.severity == 'critical')
        high_alerts = sum(1 for alert in alerts if alert.severity == 'high')
        
        score -= critical_alerts * 15  # -15 for each critical alert
        score -= high_alerts * 10     # -10 for each high alert
        
        score = max(0, min(100, score))  # Clamp to 0-100
        
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 60:
            return "Poor"
        else:
            return "Critical"
    
    def _group_by_language(self, examples: List[GroundTruthExample]) -> Dict[str, int]:
        """Group examples by programming language"""
        counts = {}
        for example in examples:
            counts[example.language] = counts.get(example.language, 0) + 1
        return counts
    
    def _group_by_difficulty(self, examples: List[GroundTruthExample]) -> Dict[str, int]:
        """Group examples by difficulty level"""
        counts = {}
        for example in examples:
            counts[example.difficulty_level] = counts.get(example.difficulty_level, 0) + 1
        return counts
    
    def _analyze_false_positives(self, predictions: List[Dict[str, Any]], 
                                examples: List[GroundTruthExample]) -> List[Dict[str, Any]]:
        """Analyze false positive predictions"""
        false_positives = []
        
        for pred, example in zip(predictions, examples):
            if pred['has_documentation'] and not example.has_documentation:
                false_positives.append({
                    'example_id': example.example_id,
                    'language': example.language,
                    'difficulty_level': example.difficulty_level,
                    'confidence': pred['confidence'],
                    'patterns_found': pred.get('patterns_found', []),
                    'annotation_notes': example.annotation_notes,
                    'content_preview': example.content[:200] + "..." if len(example.content) > 200 else example.content
                })
        
        return false_positives
    
    def _analyze_false_negatives(self, predictions: List[Dict[str, Any]], 
                               examples: List[GroundTruthExample]) -> List[Dict[str, Any]]:
        """Analyze false negative predictions"""
        false_negatives = []
        
        for pred, example in zip(predictions, examples):
            if not pred['has_documentation'] and example.has_documentation:
                false_negatives.append({
                    'example_id': example.example_id,
                    'language': example.language,
                    'difficulty_level': example.difficulty_level,
                    'confidence': pred['confidence'],
                    'expected_doc_lines': example.documentation_lines,
                    'annotation_notes': example.annotation_notes,
                    'content_preview': example.content[:200] + "..." if len(example.content) > 200 else example.content
                })
        
        return false_negatives
    
    def _analyze_confidence_distribution(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence score distribution"""
        confidences = [pred['confidence'] for pred in predictions]
        
        if not confidences:
            return {}
        
        return {
            'mean': statistics.mean(confidences),
            'median': statistics.median(confidences),
            'std_dev': statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            'min': min(confidences),
            'max': max(confidences),
            'quartiles': {
                'q1': statistics.quantiles(confidences, n=4)[0] if len(confidences) >= 4 else 0.0,
                'q2': statistics.median(confidences),
                'q3': statistics.quantiles(confidences, n=4)[2] if len(confidences) >= 4 else 0.0
            },
            'confidence_bands': {
                'very_low_0_0.2': sum(1 for c in confidences if 0.0 <= c < 0.2),
                'low_0.2_0.4': sum(1 for c in confidences if 0.2 <= c < 0.4),
                'medium_0.4_0.6': sum(1 for c in confidences if 0.4 <= c < 0.6),
                'high_0.6_0.8': sum(1 for c in confidences if 0.6 <= c < 0.8),
                'very_high_0.8_1.0': sum(1 for c in confidences if 0.8 <= c <= 1.0)
            }
        }
    
    def _save_benchmark_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to disk"""
        file_path = self.results_directory / f"{result.benchmark_name}.json"
        
        # Convert to serializable format
        result_dict = {
            'benchmark_name': result.benchmark_name,
            'timestamp': result.timestamp.isoformat(),
            'dataset_size': result.dataset_size,
            'performance_rating': result.performance_rating,
            'validation_metrics': {
                'total_examples': result.validation_metrics.total_examples,
                'correct_predictions': result.validation_metrics.correct_predictions,
                'accuracy': result.validation_metrics.accuracy,
                'true_positives': result.validation_metrics.true_positives,
                'true_negatives': result.validation_metrics.true_negatives,
                'false_positives': result.validation_metrics.false_positives,
                'false_negatives': result.validation_metrics.false_negatives,
                'precision': result.validation_metrics.precision,
                'recall': result.validation_metrics.recall,
                'f1_score': result.validation_metrics.f1_score,
                'specificity': result.validation_metrics.specificity,
                'calibration_error': result.validation_metrics.calibration_error,
                'reliability_correlation': result.validation_metrics.reliability_correlation,
                'overconfidence_rate': result.validation_metrics.overconfidence_rate,
                'underconfidence_rate': result.validation_metrics.underconfidence_rate,
                'avg_processing_time': result.validation_metrics.avg_processing_time,
                'throughput_items_per_sec': result.validation_metrics.throughput_items_per_sec,
                'memory_usage_mb': result.validation_metrics.memory_usage_mb,
                'high_confidence_accuracy': result.validation_metrics.high_confidence_accuracy,
                'medium_confidence_accuracy': result.validation_metrics.medium_confidence_accuracy,
                'low_confidence_accuracy': result.validation_metrics.low_confidence_accuracy,
                'language_accuracies': result.validation_metrics.language_accuracies,
                'difficulty_accuracies': result.validation_metrics.difficulty_accuracies
            },
            'regression_alerts': [{
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'baseline_value': alert.baseline_value,
                'degradation_percentage': alert.degradation_percentage,
                'severity': alert.severity,
                'timestamp': alert.timestamp.isoformat(),
                'description': alert.description
            } for alert in result.regression_alerts],
            'detailed_results': result.detailed_results
        }
        
        with open(file_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Benchmark result saved to {file_path}")
    
    def generate_validation_report(self, result: BenchmarkResult) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE VALIDATION FRAMEWORK REPORT")
        report.append("=" * 80)
        report.append(f"Benchmark: {result.benchmark_name}")
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Dataset Size: {result.dataset_size} examples")
        report.append(f"Performance Rating: {result.performance_rating}")
        report.append("")
        
        # Overall metrics
        metrics = result.validation_metrics
        report.append("OVERALL METRICS")
        report.append("-" * 40)
        report.append(f"Accuracy:      {metrics.accuracy:.1%}")
        report.append(f"Precision:     {metrics.precision:.3f}")
        report.append(f"Recall:        {metrics.recall:.3f}")
        report.append(f"F1 Score:      {metrics.f1_score:.3f}")
        report.append(f"Specificity:   {metrics.specificity:.3f}")
        report.append("")
        
        # Confidence calibration
        report.append("CONFIDENCE CALIBRATION")
        report.append("-" * 40)
        report.append(f"Calibration Error:      {metrics.calibration_error:.3f}")
        report.append(f"Reliability Correlation: {metrics.reliability_correlation:.3f}")
        report.append(f"Overconfidence Rate:    {metrics.overconfidence_rate:.1%}")
        report.append(f"Underconfidence Rate:   {metrics.underconfidence_rate:.1%}")
        report.append("")
        
        # Performance metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Avg Processing Time:  {metrics.avg_processing_time*1000:.1f}ms")
        report.append(f"Throughput:          {metrics.throughput_items_per_sec:.1f} items/sec")
        report.append(f"Memory Usage:        {metrics.memory_usage_mb:.1f}MB")
        report.append("")
        
        # Classification breakdown
        report.append("CLASSIFICATION BREAKDOWN")
        report.append("-" * 40)
        report.append(f"True Positives:   {metrics.true_positives}")
        report.append(f"True Negatives:   {metrics.true_negatives}")
        report.append(f"False Positives:  {metrics.false_positives}")
        report.append(f"False Negatives:  {metrics.false_negatives}")
        report.append("")
        
        # Confidence bands
        report.append("ACCURACY BY CONFIDENCE LEVEL")
        report.append("-" * 40)
        report.append(f"High Confidence (0.7-1.0):   {metrics.high_confidence_accuracy:.1%}")
        report.append(f"Medium Confidence (0.3-0.7): {metrics.medium_confidence_accuracy:.1%}")
        report.append(f"Low Confidence (0.0-0.3):    {metrics.low_confidence_accuracy:.1%}")
        report.append("")
        
        # Language-specific accuracy
        if metrics.language_accuracies:
            report.append("ACCURACY BY LANGUAGE")
            report.append("-" * 40)
            for lang, accuracy in sorted(metrics.language_accuracies.items()):
                report.append(f"{lang:12}: {accuracy:.1%}")
            report.append("")
        
        # Difficulty-specific accuracy
        if metrics.difficulty_accuracies:
            report.append("ACCURACY BY DIFFICULTY")
            report.append("-" * 40)
            for difficulty, accuracy in sorted(metrics.difficulty_accuracies.items()):
                report.append(f"{difficulty:12}: {accuracy:.1%}")
            report.append("")
        
        # Regression alerts
        if result.regression_alerts:
            report.append("REGRESSION ALERTS")
            report.append("-" * 40)
            for alert in result.regression_alerts:
                report.append(f"{alert.severity.upper()}: {alert.description}")
            report.append("")
        
        # False positives analysis
        false_positives = result.detailed_results.get('false_positives', [])
        if false_positives:
            report.append(f"FALSE POSITIVES ANALYSIS ({len(false_positives)} cases)")
            report.append("-" * 40)
            for fp in false_positives[:5]:  # Show top 5
                report.append(f"- {fp['language']} ({fp['difficulty_level']}): {fp['annotation_notes']}")
            if len(false_positives) > 5:
                report.append(f"  ... and {len(false_positives) - 5} more")
            report.append("")
            
        # False negatives analysis
        false_negatives = result.detailed_results.get('false_negatives', [])
        if false_negatives:
            report.append(f"FALSE NEGATIVES ANALYSIS ({len(false_negatives)} cases)")
            report.append("-" * 40)
            for fn in false_negatives[:5]:  # Show top 5
                report.append(f"- {fn['language']} ({fn['difficulty_level']}): {fn['annotation_notes']}")
            if len(false_negatives) > 5:
                report.append(f"  ... and {len(false_negatives) - 5} more")
            report.append("")
        
        # Performance benchmark
        perf_benchmark = result.detailed_results.get('performance_benchmark', {})
        if perf_benchmark:
            report.append("PERFORMANCE BENCHMARK")
            report.append("-" * 40)
            for batch_name, batch_results in perf_benchmark.items():
                batch_size = batch_results['batch_size']
                throughput = batch_results['throughput_items_per_sec']
                avg_time = batch_results['avg_time_per_item'] * 1000  # Convert to ms
                report.append(f"Batch {batch_size:3d}: {throughput:6.1f} items/sec, {avg_time:5.1f}ms/item")
            report.append("")
        
        report.append("=" * 80)
        
        return '\n'.join(report)
    
    def run_continuous_monitoring(self, 
                                check_interval_hours: int = 24,
                                alert_callback: Optional[callable] = None) -> None:
        """
        Run continuous monitoring with periodic validation
        
        Args:
            check_interval_hours: Hours between validation checks
            alert_callback: Callback function for regression alerts
        """
        logger.info(f"Starting continuous monitoring (checking every {check_interval_hours} hours)")
        
        while True:
            try:
                # Run validation
                result = self.run_comprehensive_validation()
                
                # Check for critical alerts
                critical_alerts = [alert for alert in result.regression_alerts if alert.severity == 'critical']
                
                if critical_alerts and alert_callback:
                    alert_callback(critical_alerts, result)
                
                # Log status
                logger.info(f"Monitoring check completed - Rating: {result.performance_rating}, "
                           f"Accuracy: {result.validation_metrics.accuracy:.1%}, "
                           f"Alerts: {len(result.regression_alerts)}")
                
                # Wait for next check
                time.sleep(check_interval_hours * 3600)
                
            except KeyboardInterrupt:
                logger.info("Continuous monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                logger.error(traceback.format_exc())
                time.sleep(3600)  # Wait 1 hour before retrying


# Convenience functions for easy usage
def run_quick_validation(include_performance: bool = True) -> BenchmarkResult:
    """
    Run a quick validation with default settings
    
    Args:
        include_performance: Include performance benchmarking
        
    Returns:
        BenchmarkResult with validation results
    """
    framework = ComprehensiveValidationFramework()
    return framework.run_comprehensive_validation(include_performance_benchmark=include_performance)


def validate_against_custom_examples(examples: List[GroundTruthExample]) -> BenchmarkResult:
    """
    Validate against custom ground truth examples
    
    Args:
        examples: List of custom ground truth examples
        
    Returns:
        BenchmarkResult with validation results
    """
    framework = ComprehensiveValidationFramework()
    
    # Add examples to ground truth manager
    for example in examples:
        framework.ground_truth_manager.add_example(example, 'custom_validation')
    
    return framework.run_comprehensive_validation(dataset_names=['custom_validation'])


def generate_ground_truth_from_codebase(codebase_path: str, 
                                      sample_size: int = 100) -> List[GroundTruthExample]:
    """
    Generate ground truth examples from a real codebase
    
    Args:
        codebase_path: Path to codebase directory
        sample_size: Number of examples to generate
        
    Returns:
        List of GroundTruthExample objects
    """
    # This would be implemented to scan a codebase and create ground truth examples
    # For now, return empty list as placeholder
    logger.warning("generate_ground_truth_from_codebase not yet implemented")
    return []


if __name__ == "__main__":
    # Example usage and testing
    print("Comprehensive Validation Framework")
    print("=" * 50)
    
    # Initialize framework
    framework = ComprehensiveValidationFramework(enable_advanced_confidence=True)
    
    # Run comprehensive validation
    print("Running comprehensive validation...")
    result = framework.run_comprehensive_validation()
    
    # Generate and display report
    report = framework.generate_validation_report(result)
    print("\n" + report)
    
    # Save detailed results
    print(f"\nDetailed results saved to: validation_results/{result.benchmark_name}.json")
    
    print("\nValidation framework ready for production monitoring!")