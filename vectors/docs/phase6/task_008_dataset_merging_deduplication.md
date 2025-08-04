# Task 008: Implement Dataset Merging and Deduplication

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 001-007. The GroundTruthDataset needs robust merging and deduplication capabilities to combine test cases from multiple sources while maintaining data integrity and avoiding redundant test cases.

## Project Structure
```
src/
  validation/
    ground_truth.rs  <- Extend this file
  lib.rs
Cargo.toml
```

## Task Description
Implement comprehensive dataset merging and deduplication logic that can combine multiple GroundTruthDataset instances while intelligently handling duplicate queries, conflicting expectations, and maintaining test coverage across all query types.

## Requirements
1. Add to existing `src/validation/ground_truth.rs`
2. Implement intelligent merging with conflict resolution
3. Add sophisticated deduplication logic
4. Create merge strategies for different use cases
5. Handle conflicting test expectations gracefully
6. Maintain test coverage statistics during merging
7. Add merge validation and rollback capabilities

## Expected Code Structure to Add
```rust
use std::collections::{HashMap, HashSet};
use anyhow::{anyhow, Result, Context};

#[derive(Debug, Clone)]
pub enum MergeStrategy {
    KeepFirst,          // Keep first occurrence, ignore duplicates
    KeepLast,           // Keep last occurrence, overwrite duplicates
    KeepMostSpecific,   // Keep test case with more specific expectations
    MergeCombined,      // Attempt to merge expectations intelligently
    RequireExactMatch,  // Fail if duplicates don't match exactly
}

#[derive(Debug, Clone)]
pub struct MergeOptions {
    pub strategy: MergeStrategy,
    pub allow_conflicts: bool,
    pub preserve_order: bool,
    pub min_confidence_threshold: f64,
    pub max_conflicts_allowed: usize,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::KeepMostSpecific,
            allow_conflicts: true,
            preserve_order: true,
            min_confidence_threshold: 0.7,
            max_conflicts_allowed: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MergeResult {
    pub merged_dataset: GroundTruthDataset,
    pub conflicts_resolved: Vec<MergeConflict>,
    pub duplicates_removed: usize,
    pub cases_added: usize,
    pub coverage_report: MergeCoverageReport,
}

#[derive(Debug, Clone)]
pub struct MergeConflict {
    pub query: String,
    pub conflict_type: ConflictType,
    pub original_case: GroundTruthCase,
    pub conflicting_case: GroundTruthCase,
    pub resolution: ConflictResolution,
}

#[derive(Debug, Clone)]
pub enum ConflictType {
    DifferentExpectedFiles,
    DifferentExpectedCount,
    ConflictingContentRequirements,
    DifferentQueryType,
    InconsistentExpectations,
}

#[derive(Debug, Clone)]
pub enum ConflictResolution {
    KeptOriginal,
    KeptNew,
    Merged(GroundTruthCase),
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct MergeCoverageReport {
    pub query_types_before: HashMap<QueryType, usize>,
    pub query_types_after: HashMap<QueryType, usize>,
    pub coverage_improved: bool,
    pub new_query_types: Vec<QueryType>,
    pub lost_query_types: Vec<QueryType>,
}

impl GroundTruthDataset {
    pub fn merge(&mut self, other: GroundTruthDataset) -> Result<MergeResult> {
        self.merge_with_options(other, MergeOptions::default())
    }
    
    pub fn merge_with_options(&mut self, other: GroundTruthDataset, options: MergeOptions) -> Result<MergeResult> {
        let original_coverage = self.query_type_distribution();
        let mut conflicts = Vec::new();
        let mut duplicates_removed = 0;
        let mut cases_added = 0;
        
        // Create a map of existing queries for efficient lookup
        let mut existing_queries: HashMap<String, usize> = HashMap::new();
        for (idx, case) in self.test_cases.iter().enumerate() {
            existing_queries.insert(case.query.clone(), idx);
        }
        
        // Process each case from the other dataset
        for new_case in other.test_cases {
            if let Some(&existing_idx) = existing_queries.get(&new_case.query) {
                // Handle duplicate query
                let existing_case = &self.test_cases[existing_idx];
                let conflict = self.detect_conflict(existing_case, &new_case)?;
                
                match conflict {
                    Some(conflict_type) => {
                        let resolution = self.resolve_conflict(
                            existing_case,
                            &new_case,
                            &conflict_type,
                            &options
                        )?;
                        
                        let merge_conflict = MergeConflict {
                            query: new_case.query.clone(),
                            conflict_type,
                            original_case: existing_case.clone(),
                            conflicting_case: new_case.clone(),
                            resolution: resolution.clone(),
                        };
                        
                        match resolution {
                            ConflictResolution::KeptOriginal => {
                                duplicates_removed += 1;
                            },
                            ConflictResolution::KeptNew => {
                                self.test_cases[existing_idx] = new_case;
                                duplicates_removed += 1;
                            },
                            ConflictResolution::Merged(merged_case) => {
                                self.test_cases[existing_idx] = merged_case;
                                duplicates_removed += 1;
                            },
                            ConflictResolution::Failed(error) => {
                                if !options.allow_conflicts {
                                    return Err(anyhow!("Merge conflict: {}", error));
                                }
                            },
                        }
                        
                        conflicts.push(merge_conflict);
                        
                        if conflicts.len() > options.max_conflicts_allowed {
                            return Err(anyhow!(
                                "Too many conflicts ({}), aborting merge",
                                conflicts.len()
                            ));
                        }
                    },
                    None => {
                        // Exact duplicate, remove based on strategy
                        duplicates_removed += 1;
                    }
                }
            } else {
                // New query, add it
                if self.should_add_case(&new_case, &options)? {
                    self.test_cases.push(new_case);
                    cases_added += 1;
                }
            }
        }
        
        // Update the existing queries map after modifications
        existing_queries.clear();
        for (idx, case) in self.test_cases.iter().enumerate() {
            existing_queries.insert(case.query.clone(), idx);
        }
        
        let final_coverage = self.query_type_distribution();
        let coverage_report = self.create_coverage_report(&original_coverage, &final_coverage);
        
        Ok(MergeResult {
            merged_dataset: self.clone(),
            conflicts_resolved: conflicts,
            duplicates_removed,
            cases_added,
            coverage_report,
        })
    }
    
    fn detect_conflict(&self, existing: &GroundTruthCase, new: &GroundTruthCase) -> Result<Option<ConflictType>> {
        // Check for exact match first
        if existing == new {
            return Ok(None);
        }
        
        // Detect different types of conflicts
        if existing.expected_files != new.expected_files {
            return Ok(Some(ConflictType::DifferentExpectedFiles));
        }
        
        if existing.expected_count != new.expected_count {
            return Ok(Some(ConflictType::DifferentExpectedCount));
        }
        
        if existing.query_type != new.query_type {
            return Ok(Some(ConflictType::DifferentQueryType));
        }
        
        if existing.must_contain != new.must_contain || existing.must_not_contain != new.must_not_contain {
            return Ok(Some(ConflictType::ConflictingContentRequirements));
        }
        
        // If we get here, there are subtle differences
        Ok(Some(ConflictType::InconsistentExpectations))
    }
    
    fn resolve_conflict(
        &self,
        existing: &GroundTruthCase,
        new: &GroundTruthCase,
        conflict_type: &ConflictType,
        options: &MergeOptions
    ) -> Result<ConflictResolution> {
        match options.strategy {
            MergeStrategy::KeepFirst => Ok(ConflictResolution::KeptOriginal),
            MergeStrategy::KeepLast => Ok(ConflictResolution::KeptNew),
            MergeStrategy::KeepMostSpecific => {
                let existing_specificity = self.calculate_specificity(existing);
                let new_specificity = self.calculate_specificity(new);
                
                if new_specificity > existing_specificity {
                    Ok(ConflictResolution::KeptNew)
                } else {
                    Ok(ConflictResolution::KeptOriginal)
                }
            },
            MergeStrategy::MergeCombined => {
                match self.attempt_merge(existing, new, conflict_type) {
                    Ok(merged) => Ok(ConflictResolution::Merged(merged)),
                    Err(e) => Ok(ConflictResolution::Failed(e.to_string())),
                }
            },
            MergeStrategy::RequireExactMatch => {
                Ok(ConflictResolution::Failed(
                    format!("Exact match required but conflicts found: {:?}", conflict_type)
                ))
            },
        }
    }
    
    fn calculate_specificity(&self, case: &GroundTruthCase) -> usize {
        let mut score = 0;
        
        score += case.expected_files.len();
        score += case.must_contain.len() * 2;
        score += case.must_not_contain.len() * 2;
        
        // Bonus for more specific query types
        match case.query_type {
            QueryType::Proximity | QueryType::Regex => score += 10,
            QueryType::BooleanAnd | QueryType::BooleanOr | QueryType::BooleanNot => score += 5,
            QueryType::Wildcard | QueryType::Phrase => score += 3,
            _ => {}
        }
        
        score
    }
    
    fn attempt_merge(&self, existing: &GroundTruthCase, new: &GroundTruthCase, conflict_type: &ConflictType) -> Result<GroundTruthCase> {
        match conflict_type {
            ConflictType::DifferentExpectedFiles => {
                // Merge expected files, keeping unique entries
                let mut merged_files = existing.expected_files.clone();
                for file in &new.expected_files {
                    if !merged_files.contains(file) {
                        merged_files.push(file.clone());
                    }
                }
                
                Ok(GroundTruthCase {
                    query: existing.query.clone(),
                    expected_files: merged_files.clone(),
                    expected_count: merged_files.len(),
                    must_contain: existing.must_contain.clone(),
                    must_not_contain: existing.must_not_contain.clone(),
                    query_type: existing.query_type.clone(),
                })
            },
            ConflictType::ConflictingContentRequirements => {
                // Merge content requirements, removing conflicts
                let mut merged_must_contain = existing.must_contain.clone();
                let mut merged_must_not_contain = existing.must_not_contain.clone();
                
                for term in &new.must_contain {
                    if !merged_must_not_contain.contains(term) && !merged_must_contain.contains(term) {
                        merged_must_contain.push(term.clone());
                    }
                }
                
                for term in &new.must_not_contain {
                    if !merged_must_contain.contains(term) && !merged_must_not_contain.contains(term) {
                        merged_must_not_contain.push(term.clone());
                    }
                }
                
                Ok(GroundTruthCase {
                    query: existing.query.clone(),
                    expected_files: existing.expected_files.clone(),
                    expected_count: existing.expected_count,
                    must_contain: merged_must_contain,
                    must_not_contain: merged_must_not_contain,
                    query_type: existing.query_type.clone(),
                })
            },
            _ => Err(anyhow!("Cannot automatically merge conflict type: {:?}", conflict_type))
        }
    }
    
    fn should_add_case(&self, case: &GroundTruthCase, options: &MergeOptions) -> Result<bool> {
        // Check confidence threshold if query classification is available
        let classification = QueryType::classify_query(&case.query);
        if classification.confidence < options.min_confidence_threshold {
            return Ok(false);
        }
        
        // Validate the case before adding
        let validation_errors = case.validate()?;
        if !validation_errors.is_empty() {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    fn create_coverage_report(&self, before: &HashMap<QueryType, usize>, after: &HashMap<QueryType, usize>) -> MergeCoverageReport {
        let mut new_query_types = Vec::new();
        let mut lost_query_types = Vec::new();
        
        for query_type in after.keys() {
            if !before.contains_key(query_type) {
                new_query_types.push(query_type.clone());
            }
        }
        
        for query_type in before.keys() {
            if !after.contains_key(query_type) {
                lost_query_types.push(query_type.clone());
            }
        }
        
        let coverage_improved = after.len() > before.len() || 
            after.values().sum::<usize>() > before.values().sum::<usize>();
        
        MergeCoverageReport {
            query_types_before: before.clone(),
            query_types_after: after.clone(),
            coverage_improved,
            new_query_types,
            lost_query_types,
        }
    }
    
    pub fn deduplicate(&mut self) -> Result<usize> {
        let mut seen_queries = HashSet::new();
        let mut unique_cases = Vec::new();
        let mut removed_count = 0;
        
        for case in &self.test_cases {
            if seen_queries.insert(case.query.clone()) {
                unique_cases.push(case.clone());
            } else {
                removed_count += 1;
            }
        }
        
        self.test_cases = unique_cases;
        Ok(removed_count)
    }
    
    pub fn deduplicate_smart(&mut self) -> Result<usize> {
        let mut query_to_case = HashMap::new();
        let mut removed_count = 0;
        
        for case in &self.test_cases {
            if let Some(existing) = query_to_case.get(&case.query) {
                // Keep the more specific case
                let existing_specificity = self.calculate_specificity(existing);
                let new_specificity = self.calculate_specificity(case);
                
                if new_specificity > existing_specificity {
                    query_to_case.insert(case.query.clone(), case.clone());
                }
                removed_count += 1;
            } else {
                query_to_case.insert(case.query.clone(), case.clone());
            }
        }
        
        self.test_cases = query_to_case.into_values().collect();
        Ok(removed_count)
    }
}

impl MergeResult {
    pub fn summary(&self) -> String {
        format!(
            "Merge completed: {} cases added, {} duplicates removed, {} conflicts resolved. Coverage improved: {}",
            self.cases_added,
            self.duplicates_removed,
            self.conflicts_resolved.len(),
            self.coverage_report.coverage_improved
        )
    }
    
    pub fn has_unresolved_conflicts(&self) -> bool {
        self.conflicts_resolved.iter().any(|c| matches!(c.resolution, ConflictResolution::Failed(_)))
    }
}

#[cfg(test)]
mod merge_tests {
    use super::*;
    
    #[test]
    fn test_basic_merge() {
        let mut dataset1 = GroundTruthDataset::new();
        dataset1.add_test(GroundTruthCase {
            query: "test1".to_string(),
            expected_files: vec!["file1.rs".to_string()],
            expected_count: 1,
            must_contain: vec![],
            must_not_contain: vec![],
            query_type: QueryType::Phrase,
        });
        
        let mut dataset2 = GroundTruthDataset::new();
        dataset2.add_test(GroundTruthCase {
            query: "test2".to_string(),
            expected_files: vec!["file2.rs".to_string()],
            expected_count: 1,
            must_contain: vec![],
            must_not_contain: vec![],
            query_type: QueryType::Phrase,
        });
        
        let result = dataset1.merge(dataset2).expect("Merge should succeed");
        assert_eq!(result.cases_added, 1);
        assert_eq!(result.duplicates_removed, 0);
        assert_eq!(dataset1.len(), 2);
    }
    
    #[test]
    fn test_conflict_resolution() {
        let mut dataset1 = GroundTruthDataset::new();
        dataset1.add_test(GroundTruthCase {
            query: "test".to_string(),
            expected_files: vec!["file1.rs".to_string()],
            expected_count: 1,
            must_contain: vec![],
            must_not_contain: vec![],
            query_type: QueryType::Phrase,
        });
        
        let mut dataset2 = GroundTruthDataset::new();
        dataset2.add_test(GroundTruthCase {
            query: "test".to_string(),
            expected_files: vec!["file2.rs".to_string()],
            expected_count: 1,
            must_contain: vec![],
            must_not_contain: vec![],
            query_type: QueryType::Phrase,
        });
        
        let result = dataset1.merge(dataset2).expect("Merge should handle conflicts");
        assert_eq!(result.conflicts_resolved.len(), 1);
        assert!(matches!(result.conflicts_resolved[0].conflict_type, ConflictType::DifferentExpectedFiles));
    }
}
```

## Dependencies
Same as previous tasks - should already be in Cargo.toml

## Success Criteria
- Dataset merging logic compiles without errors
- All merge strategies work correctly with appropriate conflict resolution
- Deduplication removes exact and smart duplicates properly
- Conflict detection identifies all types of inconsistencies
- Merge results provide comprehensive reporting and statistics
- Coverage analysis tracks query type distribution changes
- Smart specificity calculation keeps more detailed test cases
- Unit tests demonstrate merge scenarios and edge cases

## Time Limit
10 minutes maximum