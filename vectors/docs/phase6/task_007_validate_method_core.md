# Task 007: Implement Core Validate Method

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 005-006. The validate method is the core logic that executes queries and compares results against ground truth.

## Project Structure
```
src/
  validation/
    correctness.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the core `validate()` method in CorrectnessValidator that executes a ground truth test case and returns detailed validation results.

## Requirements
1. Add to existing `src/validation/correctness.rs`
2. Implement the main `validate()` async method
3. Add file path comparison logic
4. Add timing measurement
5. Include proper error handling

## Expected Code Structure to Add
```rust
use std::time::Instant;
use std::collections::HashSet;

impl CorrectnessValidator {
    pub async fn validate(&self, case: &GroundTruthCase) -> Result<ValidationResult> {
        let mut result = ValidationResult::new();
        let start_time = Instant::now();
        
        // Determine search mode based on query type
        let search_mode = self.determine_search_mode(&case.query_type);
        
        // Execute the query
        let search_results = match self.search_system.search_hybrid(&case.query, search_mode).await {
            Ok(results) => results,
            Err(e) => {
                result.add_error(format!("Query execution failed: {}", e));
                result.set_timing(start_time.elapsed().as_millis() as u64);
                return Ok(result);
            }
        };
        
        // Set timing and counts
        result.set_timing(start_time.elapsed().as_millis() as u64);
        result.set_counts(search_results.len(), case.expected_count);
        
        // Compare file results
        let actual_files: HashSet<String> = search_results.iter()
            .map(|r| r.file_path.clone())
            .collect();
        let expected_files: HashSet<String> = case.expected_files.iter()
            .cloned()
            .collect();
        
        // Calculate true/false positives/negatives
        let true_positives = actual_files.intersection(&expected_files).count();
        let false_positives = actual_files.difference(&expected_files).count();
        let false_negatives = expected_files.difference(&actual_files).count();
        
        // Report specific file mismatches
        for missing_file in expected_files.difference(&actual_files) {
            result.add_error(format!("Missing expected file: {}", missing_file));
        }
        for unexpected_file in actual_files.difference(&expected_files) {
            result.add_error(format!("Unexpected file in results: {}", unexpected_file));
        }
        
        // Calculate metrics
        result.calculate_metrics(true_positives, false_positives, false_negatives);
        
        // Validate content requirements
        self.validate_content(&search_results, case, &mut result);
        
        Ok(result)
    }
    
    fn validate_content(&self, results: &[UnifiedResult], case: &GroundTruthCase, validation_result: &mut ValidationResult) {
        for result in results {
            // Check must_contain requirements
            for must_contain in &case.must_contain {
                if !result.content.contains(must_contain) {
                    validation_result.add_error(format!(
                        "File '{}' missing required content: '{}'", 
                        result.file_path, 
                        must_contain
                    ));
                }
            }
            
            // Check must_not_contain requirements
            for must_not_contain in &case.must_not_contain {
                if result.content.contains(must_not_contain) {
                    validation_result.add_error(format!(
                        "File '{}' contains forbidden content: '{}'", 
                        result.file_path, 
                        must_not_contain
                    ));
                }
            }
        }
    }
    
    pub async fn validate_batch(&self, cases: &[GroundTruthCase]) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();
        
        for case in cases {
            let validation_result = self.validate(case).await?;
            results.push(validation_result);
        }
        
        Ok(results)
    }
}
```

## Success Criteria
- validate() method compiles and handles async properly
- File comparison logic works correctly
- Content validation catches must_contain/must_not_contain violations
- Timing measurement is accurate
- Error reporting is comprehensive
- Batch validation method works for multiple test cases

## Time Limit
10 minutes maximum