# Task 012: Advanced Precision, Recall, and F1 Calculation with Statistical Analysis

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 005-011, specifically extending the CorrectnessValidator with advanced statistical metrics beyond the basic ValidationResult. The AdvancedMetrics struct provides micro/macro averaging, statistical significance testing, and performance regression detection.

## Project Structure
```
src/
  validation/
    correctness.rs  <- Extend this file
  lib.rs
```

## Task Description
Create the `AdvancedMetrics` struct that provides sophisticated statistical analysis including micro/macro/weighted averaging, confusion matrix generation, and statistical significance testing for validation results.

## Requirements
1. Add to existing `src/validation/correctness.rs`
2. Implement `AdvancedMetrics` struct with comprehensive metrics
3. Add confusion matrix generation and analysis
4. Include statistical significance testing capabilities
5. Add performance regression detection methods
6. Support micro/macro/weighted averaging across query types
7. Provide detailed statistical reporting methods

## Expected Code Structure to Add
```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub per_class: HashMap<String, ClassMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatTest {
    pub p_value: f64,
    pub is_significant: bool,
    pub confidence_interval: (f64, f64),
    pub effect_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    pub precision_change: f64,
    pub recall_change: f64,
    pub f1_change: f64,
    pub is_regression: bool,
    pub significance_test: StatTest,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMetrics {
    pub micro_precision: f64,
    pub macro_precision: f64,
    pub weighted_precision: f64,
    pub micro_recall: f64,
    pub macro_recall: f64,
    pub weighted_recall: f64,
    pub micro_f1: f64,
    pub macro_f1: f64,
    pub weighted_f1: f64,
    pub confusion_matrix: ConfusionMatrix,
    pub statistical_significance: StatTest,
    pub query_type_breakdown: HashMap<QueryType, ClassMetrics>,
    pub total_samples: usize,
}

impl AdvancedMetrics {
    pub fn calculate_from_results(results: &[ValidationResult]) -> Self {
        let mut total_tp = 0;
        let mut total_fp = 0;
        let mut total_fn = 0;
        let mut total_tn = 0;
        let mut query_type_stats: HashMap<QueryType, Vec<&ValidationResult>> = HashMap::new();
        
        // Aggregate results by query type
        for result in results {
            let query_type = QueryType::from_query(&result.query); // Assuming we have access to query
            query_type_stats.entry(query_type).or_default().push(result);
            
            // Calculate true negatives (assuming we have this data)
            let tp = (result.result_count - result.false_positives) as i32;
            let fp = result.false_positives as i32;
            let fn_count = result.false_negatives as i32;
            let tn = (result.expected_count - tp as usize - fn_count as usize) as i32;
            
            total_tp += tp;
            total_fp += fp;
            total_fn += fn_count;
            total_tn += tn;
        }
        
        // Calculate micro averages
        let micro_precision = if total_tp + total_fp > 0 {
            total_tp as f64 / (total_tp + total_fp) as f64
        } else { 0.0 };
        
        let micro_recall = if total_tp + total_fn > 0 {
            total_tp as f64 / (total_tp + total_fn) as f64
        } else { 0.0 };
        
        let micro_f1 = if micro_precision + micro_recall > 0.0 {
            2.0 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        } else { 0.0 };
        
        // Calculate macro averages
        let mut class_metrics = HashMap::new();
        let mut macro_precision = 0.0;
        let mut macro_recall = 0.0;
        let mut macro_f1 = 0.0;
        let mut weighted_precision = 0.0;
        let mut weighted_recall = 0.0;
        let mut weighted_f1 = 0.0;
        let mut total_support = 0;
        
        for (query_type, type_results) in &query_type_stats {
            let type_tp: usize = type_results.iter()
                .map(|r| r.result_count - r.false_positives)
                .sum();
            let type_fp: usize = type_results.iter()
                .map(|r| r.false_positives)
                .sum();
            let type_fn: usize = type_results.iter()
                .map(|r| r.false_negatives)
                .sum();
            
            let precision = if type_tp + type_fp > 0 {
                type_tp as f64 / (type_tp + type_fp) as f64
            } else { 0.0 };
            
            let recall = if type_tp + type_fn > 0 {
                type_tp as f64 / (type_tp + type_fn) as f64
            } else { 0.0 };
            
            let f1 = if precision + recall > 0.0 {
                2.0 * (precision * recall) / (precision + recall)
            } else { 0.0 };
            
            let support = type_results.len();
            total_support += support;
            
            class_metrics.insert(*query_type, ClassMetrics {
                precision,
                recall,
                f1_score: f1,
                support,
            });
            
            macro_precision += precision;
            macro_recall += recall;
            macro_f1 += f1;
            
            weighted_precision += precision * support as f64;
            weighted_recall += recall * support as f64;
            weighted_f1 += f1 * support as f64;
        }
        
        let num_classes = query_type_stats.len() as f64;
        macro_precision /= num_classes;
        macro_recall /= num_classes;
        macro_f1 /= num_classes;
        
        if total_support > 0 {
            weighted_precision /= total_support as f64;
            weighted_recall /= total_support as f64;
            weighted_f1 /= total_support as f64;
        }
        
        // Create confusion matrix
        let confusion_matrix = ConfusionMatrix {
            true_positives: total_tp as usize,
            false_positives: total_fp as usize,
            true_negatives: total_tn as usize,
            false_negatives: total_fn as usize,
            per_class: class_metrics.clone(),
        };
        
        // Calculate statistical significance (simplified chi-square test)
        let statistical_significance = Self::calculate_significance_test(&confusion_matrix);
        
        Self {
            micro_precision,
            macro_precision,
            weighted_precision,
            micro_recall,
            macro_recall,
            weighted_recall,
            micro_f1,
            macro_f1,
            weighted_f1,
            confusion_matrix,
            statistical_significance,
            query_type_breakdown: class_metrics,
            total_samples: results.len(),
        }
    }
    
    fn calculate_significance_test(matrix: &ConfusionMatrix) -> StatTest {
        let n = (matrix.true_positives + matrix.false_positives + 
                matrix.true_negatives + matrix.false_negatives) as f64;
        
        if n == 0.0 {
            return StatTest {
                p_value: 1.0,
                is_significant: false,
                confidence_interval: (0.0, 0.0),
                effect_size: 0.0,
            };
        }
        
        // Simplified chi-square calculation
        let expected = n / 4.0;
        let chi_square = [
            matrix.true_positives as f64,
            matrix.false_positives as f64,
            matrix.true_negatives as f64,
            matrix.false_negatives as f64,
        ].iter()
            .map(|&observed| (observed - expected).powi(2) / expected)
            .sum::<f64>();
        
        // Simplified p-value calculation (would use proper distribution in real implementation)
        let p_value = (-chi_square / 2.0).exp();
        let is_significant = p_value < 0.05;
        
        // Calculate effect size (Cohen's w for contingency tables)
        let effect_size = (chi_square / n).sqrt();
        
        // Simple confidence interval for F1 score
        let f1 = if matrix.true_positives > 0 {
            let precision = matrix.true_positives as f64 / 
                (matrix.true_positives + matrix.false_positives) as f64;
            let recall = matrix.true_positives as f64 / 
                (matrix.true_positives + matrix.false_negatives) as f64;
            2.0 * (precision * recall) / (precision + recall)
        } else { 0.0 };
        
        let margin = 1.96 * (f1 * (1.0 - f1) / n).sqrt(); // 95% CI
        let confidence_interval = ((f1 - margin).max(0.0), (f1 + margin).min(1.0));
        
        StatTest {
            p_value,
            is_significant,
            confidence_interval,
            effect_size,
        }
    }
    
    pub fn compare_with_baseline(&self, baseline: &AdvancedMetrics) -> RegressionReport {
        let precision_change = self.macro_precision - baseline.macro_precision;
        let recall_change = self.macro_recall - baseline.macro_recall;
        let f1_change = self.macro_f1 - baseline.macro_f1;
        
        // Determine if this represents a regression (>5% drop in any metric)
        let threshold = -0.05;
        let is_regression = precision_change < threshold || 
                          recall_change < threshold || 
                          f1_change < threshold;
        
        // Simple significance test for the difference
        let combined_samples = self.total_samples + baseline.total_samples;
        let pooled_f1 = (self.macro_f1 * self.total_samples as f64 + 
                        baseline.macro_f1 * baseline.total_samples as f64) / combined_samples as f64;
        let se = (pooled_f1 * (1.0 - pooled_f1) / combined_samples as f64).sqrt();
        let z_score = f1_change / se;
        let p_value = 2.0 * (1.0 - 0.5 * (1.0 + (z_score.abs() / 1.414).tanh())); // Approximate
        
        let significance_test = StatTest {
            p_value,
            is_significant: p_value < 0.05,
            confidence_interval: (
                f1_change - 1.96 * se,
                f1_change + 1.96 * se
            ),
            effect_size: z_score.abs(),
        };
        
        let mut recommendations = Vec::new();
        if is_regression {
            recommendations.push("Performance regression detected. Review recent changes.".to_string());
            if precision_change < threshold {
                recommendations.push("Precision dropped significantly. Check for false positive increases.".to_string());
            }
            if recall_change < threshold {
                recommendations.push("Recall dropped significantly. Check for false negative increases.".to_string());
            }
        } else if f1_change > 0.02 {
            recommendations.push("Performance improvement detected. Consider this as new baseline.".to_string());
        }
        
        RegressionReport {
            precision_change,
            recall_change,
            f1_change,
            is_regression,
            significance_test,
            recommendations,
        }
    }
    
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("=== Advanced Metrics Report ===\n"));
        report.push_str(&format!("Total Samples: {}\n\n", self.total_samples));
        
        report.push_str("Micro Averages:\n");
        report.push_str(&format!("  Precision: {:.4}\n", self.micro_precision));
        report.push_str(&format!("  Recall: {:.4}\n", self.micro_recall));
        report.push_str(&format!("  F1-Score: {:.4}\n\n", self.micro_f1));
        
        report.push_str("Macro Averages:\n");
        report.push_str(&format!("  Precision: {:.4}\n", self.macro_precision));
        report.push_str(&format!("  Recall: {:.4}\n", self.macro_recall));
        report.push_str(&format!("  F1-Score: {:.4}\n\n", self.macro_f1));
        
        report.push_str("Weighted Averages:\n");
        report.push_str(&format!("  Precision: {:.4}\n", self.weighted_precision));
        report.push_str(&format!("  Recall: {:.4}\n", self.weighted_recall));
        report.push_str(&format!("  F1-Score: {:.4}\n\n", self.weighted_f1));
        
        report.push_str("Statistical Significance:\n");
        report.push_str(&format!("  P-value: {:.4}\n", self.statistical_significance.p_value));
        report.push_str(&format!("  Significant: {}\n", self.statistical_significance.is_significant));
        report.push_str(&format!("  Effect Size: {:.4}\n", self.statistical_significance.effect_size));
        report.push_str(&format!("  95% CI: ({:.4}, {:.4})\n\n", 
                                self.statistical_significance.confidence_interval.0,
                                self.statistical_significance.confidence_interval.1));
        
        report.push_str("Confusion Matrix:\n");
        report.push_str(&format!("  TP: {}, FP: {}\n", 
                                self.confusion_matrix.true_positives,
                                self.confusion_matrix.false_positives));
        report.push_str(&format!("  FN: {}, TN: {}\n\n", 
                                self.confusion_matrix.false_negatives,
                                self.confusion_matrix.true_negatives));
        
        report.push_str("Per-Class Breakdown:\n");
        for (query_type, metrics) in &self.query_type_breakdown {
            report.push_str(&format!("  {:?}: P={:.3}, R={:.3}, F1={:.3}, Support={}\n",
                                   query_type, metrics.precision, metrics.recall, 
                                   metrics.f1_score, metrics.support));
        }
        
        report
    }
}
```

## Success Criteria
- AdvancedMetrics struct compiles without errors
- All averaging methods (micro/macro/weighted) calculate correctly
- Confusion matrix generation works properly
- Statistical significance testing provides meaningful results
- Performance regression detection identifies issues accurately
- Detailed reporting method provides comprehensive output
- All methods handle edge cases (empty results, division by zero)

## Time Limit
10 minutes maximum