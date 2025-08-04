# Task 005: Add Ground Truth Validation Methods

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 001-004. The GroundTruthDataset needs comprehensive validation methods to ensure data integrity and catch configuration errors before running expensive validation tests.

## Project Structure
```
src/
  validation/
    ground_truth.rs  <- Extend this file
  lib.rs
Cargo.toml
```

## Task Description
Add comprehensive validation methods to the GroundTruthDataset and GroundTruthCase structs. These methods will validate test case integrity, detect configuration errors, and provide detailed diagnostic information for debugging test failures.

## Requirements
1. Add to existing `src/validation/ground_truth.rs`
2. Implement validation methods for individual test cases
3. Add dataset-level validation with detailed error reporting
4. Create validation statistics and health checks
5. Add methods to detect duplicate and conflicting test cases
6. Implement query syntax validation
7. Add file existence validation (optional mode)

## Expected Code Structure to Add
```rust
use std::collections::{HashMap, HashSet};
use anyhow::{anyhow, Result, Context};

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub case_index: Option<usize>,
    pub query: Option<String>,
    pub error_type: ValidationErrorType,
    pub message: String,
}

#[derive(Debug, Clone)]
pub enum ValidationErrorType {
    EmptyQuery,
    InvalidQuerySyntax,
    DuplicateQuery,
    ConflictingExpectations,
    InvalidFileReference,
    MissingRequiredFields,
    QueryTypeMismatch,
    LogicalInconsistency,
}

impl GroundTruthCase {
    pub fn validate(&self) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        // Validate query is not empty
        if self.query.trim().is_empty() {
            errors.push(ValidationError {
                case_index: None,
                query: Some(self.query.clone()),
                error_type: ValidationErrorType::EmptyQuery,
                message: "Query cannot be empty".to_string(),
            });
        }
        
        // Validate expected_count matches expected_files length
        if self.expected_count != self.expected_files.len() {
            errors.push(ValidationError {
                case_index: None,
                query: Some(self.query.clone()),
                error_type: ValidationErrorType::LogicalInconsistency,
                message: format!(
                    "Expected count ({}) doesn't match expected files length ({})",
                    self.expected_count,
                    self.expected_files.len()
                ),
            });
        }
        
        // Validate query type matches query content
        let detected_type = QueryType::from_query(&self.query);
        if detected_type != self.query_type {
            errors.push(ValidationError {
                case_index: None,
                query: Some(self.query.clone()),
                error_type: ValidationErrorType::QueryTypeMismatch,
                message: format!(
                    "Query type mismatch: expected {:?}, detected {:?}",
                    self.query_type, detected_type
                ),
            });
        }
        
        // Validate must_contain and must_not_contain don't conflict
        for must_contain in &self.must_contain {
            if self.must_not_contain.contains(must_contain) {
                errors.push(ValidationError {
                    case_index: None,
                    query: Some(self.query.clone()),
                    error_type: ValidationErrorType::ConflictingExpectations,
                    message: format!(
                        "Content requirement conflict: '{}' in both must_contain and must_not_contain",
                        must_contain
                    ),
                });
            }
        }
        
        Ok(errors)
    }
    
    pub fn validate_query_syntax(&self) -> Result<()> {
        // Basic query syntax validation based on query type
        match self.query_type {
            QueryType::BooleanAnd => {
                if !self.query.contains("AND") && !self.query.contains("&&") {
                    return Err(anyhow!("Boolean AND query must contain AND or && operators"));
                }
            }
            QueryType::BooleanOr => {
                if !self.query.contains("OR") && !self.query.contains("||") {
                    return Err(anyhow!("Boolean OR query must contain OR or || operators"));
                }
            }
            QueryType::BooleanNot => {
                if !self.query.contains("NOT") && !self.query.contains("!") {
                    return Err(anyhow!("Boolean NOT query must contain NOT or ! operators"));
                }
            }
            QueryType::Phrase => {
                if !self.query.contains('"') {
                    return Err(anyhow!("Phrase query must contain quoted strings"));
                }
            }
            QueryType::Wildcard => {
                if !self.query.contains('*') && !self.query.contains('?') {
                    return Err(anyhow!("Wildcard query must contain * or ? characters"));
                }
            }
            _ => {} // Other types have more complex validation
        }
        
        Ok(())
    }
    
    pub fn health_score(&self) -> f64 {
        let mut score = 100.0;
        let errors = self.validate().unwrap_or_default();
        
        // Deduct points for each error type
        for error in errors {
            match error.error_type {
                ValidationErrorType::EmptyQuery => score -= 50.0,
                ValidationErrorType::QueryTypeMismatch => score -= 20.0,
                ValidationErrorType::ConflictingExpectations => score -= 30.0,
                ValidationErrorType::LogicalInconsistency => score -= 25.0,
                _ => score -= 10.0,
            }
        }
        
        score.max(0.0)
    }
}

impl GroundTruthDataset {
    pub fn validate_comprehensive(&self) -> Result<Vec<ValidationError>> {
        let mut all_errors = Vec::new();
        let mut seen_queries = HashMap::new();
        
        // Validate each test case individually
        for (index, case) in self.test_cases.iter().enumerate() {
            let mut case_errors = case.validate()?;
            for error in &mut case_errors {
                error.case_index = Some(index);
            }
            all_errors.extend(case_errors);
            
            // Check for duplicate queries
            if let Some(existing_index) = seen_queries.get(&case.query) {
                all_errors.push(ValidationError {
                    case_index: Some(index),
                    query: Some(case.query.clone()),
                    error_type: ValidationErrorType::DuplicateQuery,
                    message: format!(
                        "Duplicate query found at indices {} and {}",
                        existing_index, index
                    ),
                });
            } else {
                seen_queries.insert(case.query.clone(), index);
            }
        }
        
        Ok(all_errors)
    }
    
    pub fn validate_files_exist(&self, base_path: &std::path::Path) -> Result<Vec<ValidationError>> {
        let mut errors = Vec::new();
        
        for (index, case) in self.test_cases.iter().enumerate() {
            for expected_file in &case.expected_files {
                let file_path = base_path.join(expected_file);
                if !file_path.exists() {
                    errors.push(ValidationError {
                        case_index: Some(index),
                        query: Some(case.query.clone()),
                        error_type: ValidationErrorType::InvalidFileReference,
                        message: format!(
                            "Expected file does not exist: {}",
                            file_path.display()
                        ),
                    });
                }
            }
        }
        
        Ok(errors)
    }
    
    pub fn health_report(&self) -> ValidationHealthReport {
        let errors = self.validate_comprehensive().unwrap_or_default();
        let total_cases = self.test_cases.len();
        let error_count = errors.len();
        let cases_with_errors = errors.iter()
            .filter_map(|e| e.case_index)
            .collect::<HashSet<_>>()
            .len();
        
        let overall_health = if total_cases == 0 {
            0.0
        } else {
            ((total_cases - cases_with_errors) as f64 / total_cases as f64) * 100.0
        };
        
        ValidationHealthReport {
            total_test_cases: total_cases,
            cases_with_errors,
            total_errors: error_count,
            overall_health_score: overall_health,
            errors,
        }
    }
    
    pub fn query_type_distribution(&self) -> HashMap<QueryType, usize> {
        let mut distribution = HashMap::new();
        for case in &self.test_cases {
            *distribution.entry(case.query_type.clone()).or_insert(0) += 1;
        }
        distribution
    }
}

#[derive(Debug, Clone)]
pub struct ValidationHealthReport {
    pub total_test_cases: usize,
    pub cases_with_errors: usize,
    pub total_errors: usize,
    pub overall_health_score: f64,
    pub errors: Vec<ValidationError>,
}

impl ValidationHealthReport {
    pub fn is_healthy(&self) -> bool {
        self.overall_health_score >= 95.0 && self.total_errors == 0
    }
    
    pub fn summary(&self) -> String {
        format!(
            "Health: {:.1}% ({}/{} cases pass, {} total errors)",
            self.overall_health_score,
            self.total_test_cases - self.cases_with_errors,
            self.total_test_cases,
            self.total_errors
        )
    }
}
```

## Dependencies
Same as previous tasks - should already be in Cargo.toml

## Success Criteria
- All validation methods compile without errors
- ValidationError and ValidationHealthReport provide detailed diagnostics
- Individual test case validation catches common configuration errors
- Dataset-level validation detects duplicates and conflicts
- Health scoring provides actionable metrics
- File existence validation works with real file systems
- Query syntax validation matches query type expectations

## Time Limit
10 minutes maximum