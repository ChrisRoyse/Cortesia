# Task 006: Create ValidationResult Struct with Metrics

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Task 005. The ValidationResult struct captures the results of validating a single query against ground truth data.

## Project Structure
```
src/
  validation/
    correctness.rs  <- Extend this file
  lib.rs
```

## Task Description
Create the `ValidationResult` struct that captures detailed metrics about validation accuracy, including precision, recall, F1 score, and error details.

## Requirements
1. Add to existing `src/validation/correctness.rs`
2. Implement `ValidationResult` struct with all metrics
3. Add calculation helper methods
4. Include detailed error reporting
5. Add serialization support for reporting

## Expected Code Structure to Add
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_correct: bool,
    pub accuracy: f64,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub error_details: Vec<String>,
    pub execution_time_ms: u64,
    pub result_count: usize,
    pub expected_count: usize,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_correct: true,
            accuracy: 0.0,
            false_positives: 0,
            false_negatives: 0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            error_details: Vec::new(),
            execution_time_ms: 0,
            result_count: 0,
            expected_count: 0,
        }
    }
    
    pub fn calculate_metrics(&mut self, true_positives: usize, false_positives: usize, false_negatives: usize) {
        self.false_positives = false_positives;
        self.false_negatives = false_negatives;
        
        // Set correctness
        self.is_correct = false_positives == 0 && false_negatives == 0;
        
        // Calculate precision
        if true_positives + false_positives > 0 {
            self.precision = true_positives as f64 / (true_positives + false_positives) as f64;
        } else {
            self.precision = 0.0;
        }
        
        // Calculate recall
        if true_positives + false_negatives > 0 {
            self.recall = true_positives as f64 / (true_positives + false_negatives) as f64;
        } else {
            self.recall = 0.0;
        }
        
        // Calculate F1 score
        if self.precision + self.recall > 0.0 {
            self.f1_score = 2.0 * (self.precision * self.recall) / (self.precision + self.recall);
        } else {
            self.f1_score = 0.0;
        }
        
        // Calculate accuracy (100% if perfect, 0% otherwise for strict validation)
        self.accuracy = if self.is_correct { 100.0 } else { 0.0 };
    }
    
    pub fn add_error(&mut self, error: String) {
        self.error_details.push(error);
        self.is_correct = false;
        self.accuracy = 0.0;
    }
    
    pub fn set_timing(&mut self, duration_ms: u64) {
        self.execution_time_ms = duration_ms;
    }
    
    pub fn set_counts(&mut self, result_count: usize, expected_count: usize) {
        self.result_count = result_count;
        self.expected_count = expected_count;
    }
    
    pub fn summary(&self) -> String {
        format!(
            "Correct: {}, Accuracy: {:.1}%, Precision: {:.3}, Recall: {:.3}, F1: {:.3}, Time: {}ms",
            self.is_correct, self.accuracy, self.precision, self.recall, self.f1_score, self.execution_time_ms
        )
    }
    
    pub fn has_errors(&self) -> bool {
        !self.error_details.is_empty()
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}
```

## Success Criteria
- ValidationResult struct compiles without errors
- All metrics are calculated correctly
- Error handling is comprehensive
- Timing and count tracking work properly
- Summary method provides useful output
- Serialization works for reporting

## Time Limit
10 minutes maximum