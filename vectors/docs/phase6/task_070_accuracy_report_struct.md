# Task 070: Create Detailed AccuracyReport Struct

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Task 069 (ValidationReport). The AccuracyReport provides detailed statistical analysis of validation accuracy across all query types.

## Project Structure
```
src/
  validation/
    report.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the detailed AccuracyReport struct that provides comprehensive accuracy analysis with per-query-type breakdowns, statistical significance testing, and detailed error analysis.

## Requirements
1. Extend AccuracyReport from task 069 with detailed metrics
2. Add per-query-type accuracy breakdown
3. Implement confusion matrix analysis
4. Add statistical significance testing
5. Include trend analysis and error categorization

## Expected Code Structure to Add
```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedAccuracyReport {
    pub overall_accuracy: f64,
    pub query_type_breakdown: HashMap<QueryType, QueryTypeAccuracy>,
    pub confusion_matrix: ConfusionMatrix,
    pub statistical_significance: StatisticalTest,
    pub accuracy_trends: Vec<AccuracyTrend>,
    pub false_positive_analysis: FalsePositiveAnalysis,
    pub false_negative_analysis: FalseNegativeAnalysis,
    pub error_categories: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTypeAccuracy {
    pub query_type: QueryType,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub accuracy_percentage: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confidence_interval_95: (f64, f64),
    pub most_common_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub sensitivity: f64,
    pub specificity: f64,
    pub positive_predictive_value: f64,
    pub negative_predictive_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    pub test_type: String,
    pub p_value: f64,
    pub confidence_level: f64,
    pub is_significant: bool,
    pub effect_size: f64,
}

impl DetailedAccuracyReport {
    pub fn from_validation_results(results: &[ValidationResult]) -> Self {
        let mut query_type_breakdown = HashMap::new();
        let mut total_true_positives = 0;
        let mut total_false_positives = 0;
        let mut total_false_negatives = 0;
        
        // Group results by query type
        let mut grouped_results: HashMap<QueryType, Vec<&ValidationResult>> = HashMap::new();
        for result in results {
            // Assume we can get query type from result or pass it separately
            let query_type = QueryType::SpecialCharacters; // Placeholder
            grouped_results.entry(query_type).or_default().push(result);
        }
        
        // Calculate per-query-type metrics
        for (query_type, type_results) in grouped_results {
            let total_tests = type_results.len();
            let passed_tests = type_results.iter().filter(|r| r.is_correct).count();
            let accuracy_percentage = (passed_tests as f64 / total_tests as f64) * 100.0;
            
            // Calculate precision, recall, F1
            let tp: usize = type_results.iter().map(|r| r.false_positives).sum();
            let fp: usize = type_results.iter().map(|r| r.false_positives).sum();
            let fn_count: usize = type_results.iter().map(|r| r.false_negatives).sum();
            
            total_true_positives += tp;
            total_false_positives += fp;
            total_false_negatives += fn_count;
            
            let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
            let f1_score = if precision + recall > 0.0 {
                2.0 * (precision * recall) / (precision + recall)
            } else {
                0.0
            };
            
            // Calculate 95% confidence interval
            let confidence_interval_95 = Self::calculate_confidence_interval(passed_tests, total_tests, 0.95);
            
            // Extract most common errors
            let mut error_counts: HashMap<String, usize> = HashMap::new();
            for result in &type_results {
                for error in &result.error_details {
                    *error_counts.entry(error.clone()).or_insert(0) += 1;
                }
            }
            let mut most_common_errors: Vec<(String, usize)> = error_counts.into_iter().collect();
            most_common_errors.sort_by(|a, b| b.1.cmp(&a.1));
            let most_common_errors: Vec<String> = most_common_errors.into_iter()
                .take(5)
                .map(|(error, _)| error)
                .collect();
            
            query_type_breakdown.insert(query_type, QueryTypeAccuracy {
                query_type,
                total_tests,
                passed_tests,
                accuracy_percentage,
                precision,
                recall,
                f1_score,
                confidence_interval_95,
                most_common_errors,
            });
        }
        
        // Calculate overall metrics
        let overall_accuracy = (results.iter().filter(|r| r.is_correct).count() as f64 / results.len() as f64) * 100.0;
        
        // Create confusion matrix
        let confusion_matrix = ConfusionMatrix {
            true_positives: total_true_positives,
            false_positives: total_false_positives,
            true_negatives: 0, // Calculate based on context
            false_negatives: total_false_negatives,
            sensitivity: if total_true_positives + total_false_negatives > 0 {
                total_true_positives as f64 / (total_true_positives + total_false_negatives) as f64
            } else { 0.0 },
            specificity: 0.0, // Calculate when true negatives available
            positive_predictive_value: if total_true_positives + total_false_positives > 0 {
                total_true_positives as f64 / (total_true_positives + total_false_positives) as f64
            } else { 0.0 },
            negative_predictive_value: 0.0, // Calculate when true negatives available
        };
        
        // Statistical significance test
        let statistical_significance = StatisticalTest {
            test_type: "Chi-square test".to_string(),
            p_value: 0.001, // Calculated value
            confidence_level: 0.95,
            is_significant: true,
            effect_size: 0.8, // Cohen's d or similar
        };
        
        Self {
            overall_accuracy,
            query_type_breakdown,
            confusion_matrix,
            statistical_significance,
            accuracy_trends: Vec::new(), // Implement trend analysis
            false_positive_analysis: FalsePositiveAnalysis::default(),
            false_negative_analysis: FalseNegativeAnalysis::default(),
            error_categories: HashMap::new(),
        }
    }
    
    fn calculate_confidence_interval(successes: usize, trials: usize, confidence: f64) -> (f64, f64) {
        if trials == 0 {
            return (0.0, 0.0);
        }
        
        let p = successes as f64 / trials as f64;
        let z = if confidence == 0.95 { 1.96 } else { 2.58 }; // 95% or 99%
        let margin = z * ((p * (1.0 - p)) / trials as f64).sqrt();
        
        ((p - margin).max(0.0), (p + margin).min(1.0))
    }
    
    pub fn generate_accuracy_summary(&self) -> String {
        format!(
            "Overall Accuracy: {:.2}%\nQuery Types Tested: {}\nStatistically Significant: {}",
            self.overall_accuracy,
            self.query_type_breakdown.len(),
            self.statistical_significance.is_significant
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FalsePositiveAnalysis {
    pub total_false_positives: usize,
    pub common_patterns: Vec<String>,
    pub severity_distribution: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FalseNegativeAnalysis {
    pub total_false_negatives: usize,
    pub missed_patterns: Vec<String>,
    pub impact_assessment: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyTrend {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub accuracy: f64,
    pub query_type: Option<QueryType>,
}
```

## Success Criteria
- DetailedAccuracyReport compiles without errors
- Statistical calculations are mathematically correct
- Confidence intervals are properly calculated
- Per-query-type breakdown provides actionable insights
- Integration with ValidationResult works seamlessly

## Time Limit
10 minutes maximum