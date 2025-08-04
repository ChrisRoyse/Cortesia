# Task 13: Create Ground Truth Validation Data

## Context
You are completing the test data generation phase (Phase 0, Task 13). Tasks 10-12 created test files with special characters, edge cases, and boundary conditions. Now you need to create ground truth data that defines the expected results for validation and testing.

## Objective
Generate a comprehensive ground truth validation system that defines expected search results, chunk boundaries, and parsing outcomes for all test files. This will enable automated validation of the search system's accuracy.

## Requirements
1. Create ground truth data structure for search validation
2. Define expected results for special character searches
3. Define expected chunk boundaries for semantic parsing
4. Create validation functions for accuracy testing
5. Generate test queries with expected outcomes
6. Create performance baseline expectations

## Implementation for test_data.rs (extend existing)
```rust
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use anyhow::Result;
use tracing::{info, debug};

impl ChunkBoundaryTestGenerator {
    // ... existing methods ...
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    pub version: String,
    pub created_at: String,
    pub test_cases: Vec<TestCase>,
    pub search_queries: Vec<SearchQuery>,
    pub chunk_expectations: Vec<ChunkExpectation>,
    pub performance_baselines: PerformanceBaselines,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub id: String,
    pub file_path: String,
    pub description: String,
    pub content_hash: String,
    pub expected_chunks: Vec<ExpectedChunk>,
    pub search_targets: Vec<SearchTarget>,
    pub validation_rules: Vec<ValidationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub id: String,
    pub query: String,
    pub query_type: QueryType,
    pub expected_results: Vec<ExpectedResult>,
    pub expected_count: usize,
    pub max_latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkExpectation {
    pub file_path: String,
    pub expected_chunk_count: usize,
    pub chunk_boundaries: Vec<ChunkBoundary>,
    pub semantic_units: Vec<SemanticUnit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedChunk {
    pub start_byte: usize,
    pub end_byte: usize,
    pub chunk_type: String,
    pub content_preview: String,
    pub should_be_searchable: bool,
    pub contains_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchTarget {
    pub pattern: String,
    pub should_find: bool,
    pub expected_count: usize,
    pub expected_locations: Vec<LocationRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationRange {
    pub start_byte: usize,
    pub end_byte: usize,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_type: ValidationRuleType,
    pub description: String,
    pub expected_outcome: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    MustParse,
    MustNotCrash,
    MustPreserveBoundaries,
    MustFindPattern(String),
    MustNotSplitSemanticUnit,
    MustHandleUnicode,
    MustHandleSpecialChars,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    ExactMatch,
    BooleanAnd,
    BooleanOr,
    BooleanNot,
    Fuzzy,
    Regex,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedResult {
    pub file_path: String,
    pub match_start: usize,
    pub match_end: usize,
    pub context: String,
    pub relevance_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkBoundary {
    pub start_byte: usize,
    pub end_byte: usize,
    pub boundary_type: BoundaryType,
    pub should_not_split: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryType {
    Function,
    Struct,
    Impl,
    Module,
    Macro,
    Comment,
    Generic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticUnit {
    pub unit_type: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub name: String,
    pub must_preserve_integrity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaselines {
    pub indexing_rate_docs_per_sec: f64,
    pub search_latency_ms: f64,
    pub memory_usage_mb: f64,
    pub concurrent_searches: u32,
    pub max_file_size_mb: f64,
}

pub struct GroundTruthGenerator;

impl GroundTruthGenerator {
    /// Generate comprehensive ground truth validation data
    pub fn generate_ground_truth() -> Result<GroundTruth> {
        info!("Generating ground truth validation data");
        
        let mut ground_truth = GroundTruth {
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            test_cases: Vec::new(),
            search_queries: Vec::new(),
            chunk_expectations: Vec::new(),
            performance_baselines: Self::create_performance_baselines(),
        };
        
        // Generate test cases for special characters
        ground_truth.test_cases.extend(Self::generate_special_char_test_cases()?);
        
        // Generate test cases for edge cases
        ground_truth.test_cases.extend(Self::generate_edge_case_test_cases()?);
        
        // Generate test cases for chunk boundaries
        ground_truth.test_cases.extend(Self::generate_boundary_test_cases()?);
        
        // Generate search queries
        ground_truth.search_queries = Self::generate_search_queries()?;
        
        // Generate chunk expectations
        ground_truth.chunk_expectations = Self::generate_chunk_expectations()?;
        
        // Save ground truth to file
        Self::save_ground_truth(&ground_truth)?;
        
        info!("Ground truth validation data generated successfully");
        Ok(ground_truth)
    }
    
    fn generate_special_char_test_cases() -> Result<Vec<TestCase>> {
        debug!("Generating special character test cases");
        
        let mut test_cases = Vec::new();
        
        // Cargo.toml patterns
        test_cases.push(TestCase {
            id: "cargo_basic_patterns".to_string(),
            file_path: "test_data/special_chars/cargo_basic.toml".to_string(),
            description: "Basic Cargo.toml with all common patterns".to_string(),
            content_hash: "".to_string(), // Would be computed from actual content
            expected_chunks: vec![
                ExpectedChunk {
                    start_byte: 0,
                    end_byte: 500, // Approximate
                    chunk_type: "cargo_manifest".to_string(),
                    content_preview: "[package]\nname = \"test-project\"".to_string(),
                    should_be_searchable: true,
                    contains_patterns: vec![
                        "[package]".to_string(),
                        "[dependencies]".to_string(),
                        "[workspace]".to_string(),
                    ],
                },
            ],
            search_targets: vec![
                SearchTarget {
                    pattern: "[package]".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![LocationRange {
                        start_byte: 1,
                        end_byte: 10,
                        context: "[package]\nname = \"test-project\"".to_string(),
                    }],
                },
                SearchTarget {
                    pattern: "[dependencies]".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![LocationRange {
                        start_byte: 150,
                        end_byte: 164,
                        context: "[dependencies]\nserde = { version = \"1.0\"".to_string(),
                    }],
                },
                SearchTarget {
                    pattern: "[workspace]".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![LocationRange {
                        start_byte: 300,
                        end_byte: 311,
                        context: "[workspace]\nmembers = [".to_string(),
                    }],
                },
            ],
            validation_rules: vec![
                ValidationRule {
                    rule_type: ValidationRuleType::MustParse,
                    description: "Cargo.toml must parse without errors".to_string(),
                    expected_outcome: true,
                },
                ValidationRule {
                    rule_type: ValidationRuleType::MustFindPattern("[package]".to_string()),
                    description: "Must find [package] section".to_string(),
                    expected_outcome: true,
                },
                ValidationRule {
                    rule_type: ValidationRuleType::MustHandleSpecialChars,
                    description: "Must handle brackets in TOML".to_string(),
                    expected_outcome: true,
                },
            ],
        });
        
        // Rust generics patterns
        test_cases.push(TestCase {
            id: "rust_generics_patterns".to_string(),
            file_path: "test_data/special_chars/generics.rs".to_string(),
            description: "Rust code with complex generic patterns".to_string(),
            content_hash: "".to_string(),
            expected_chunks: vec![
                ExpectedChunk {
                    start_byte: 0,
                    end_byte: 200,
                    chunk_type: "use_statements".to_string(),
                    content_preview: "use std::collections::HashMap;".to_string(),
                    should_be_searchable: true,
                    contains_patterns: vec!["use std::".to_string()],
                },
                ExpectedChunk {
                    start_byte: 200,
                    end_byte: 800,
                    chunk_type: "struct_definition".to_string(),
                    content_preview: "pub struct Container<T, E = std::io::Error>".to_string(),
                    should_be_searchable: true,
                    contains_patterns: vec![
                        "Container<T, E".to_string(),
                        "where".to_string(),
                        "Clone + Send + Sync".to_string(),
                    ],
                },
            ],
            search_targets: vec![
                SearchTarget {
                    pattern: "Result<T, E>".to_string(),
                    should_find: true,
                    expected_count: 2,
                    expected_locations: vec![
                        LocationRange {
                            start_byte: 450,
                            end_byte: 462,
                            context: "-> Result<T, E>".to_string(),
                        },
                    ],
                },
                SearchTarget {
                    pattern: "HashMap<String, String>".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![
                        LocationRange {
                            start_byte: 350,
                            end_byte: 373,
                            context: "metadata: HashMap<String, String>".to_string(),
                        },
                    ],
                },
            ],
            validation_rules: vec![
                ValidationRule {
                    rule_type: ValidationRuleType::MustPreserveBoundaries,
                    description: "Generic bounds must not be split".to_string(),
                    expected_outcome: true,
                },
            ],
        });
        
        // Complex patterns test case
        test_cases.push(TestCase {
            id: "complex_patterns".to_string(),
            file_path: "test_data/special_chars/complex_patterns.rs".to_string(),
            description: "Real-world complex Rust patterns".to_string(),
            content_hash: "".to_string(),
            expected_chunks: vec![
                ExpectedChunk {
                    start_byte: 0,
                    end_byte: 1500,
                    chunk_type: "trait_definition".to_string(),
                    content_preview: "#[async_trait::async_trait]\npub trait AsyncRepository".to_string(),
                    should_be_searchable: true,
                    contains_patterns: vec![
                        "#[async_trait::async_trait]".to_string(),
                        "AsyncRepository<T, K, E>".to_string(),
                    ],
                },
            ],
            search_targets: vec![
                SearchTarget {
                    pattern: "#[async_trait::async_trait]".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![
                        LocationRange {
                            start_byte: 50,
                            end_byte: 76,
                            context: "#[async_trait::async_trait]\npub trait".to_string(),
                        },
                    ],
                },
            ],
            validation_rules: vec![
                ValidationRule {
                    rule_type: ValidationRuleType::MustNotSplitSemanticUnit,
                    description: "Trait definition must remain intact".to_string(),
                    expected_outcome: true,
                },
            ],
        });
        
        Ok(test_cases)
    }
    
    fn generate_edge_case_test_cases() -> Result<Vec<TestCase>> {
        debug!("Generating edge case test cases");
        
        let mut test_cases = Vec::new();
        
        // Empty file test
        test_cases.push(TestCase {
            id: "empty_file".to_string(),
            file_path: "test_data/edge_cases/empty.txt".to_string(),
            description: "Completely empty file".to_string(),
            content_hash: "".to_string(),
            expected_chunks: vec![], // No chunks expected
            search_targets: vec![
                SearchTarget {
                    pattern: "anything".to_string(),
                    should_find: false,
                    expected_count: 0,
                    expected_locations: vec![],
                },
            ],
            validation_rules: vec![
                ValidationRule {
                    rule_type: ValidationRuleType::MustNotCrash,
                    description: "Empty file must not crash parser".to_string(),
                    expected_outcome: true,
                },
            ],
        });
        
        // Large file test
        test_cases.push(TestCase {
            id: "large_file".to_string(),
            file_path: "test_data/edge_cases/large_repetitive.rs".to_string(),
            description: "Very large file (~10MB)".to_string(),
            content_hash: "".to_string(),
            expected_chunks: vec![], // Many chunks expected, specific count TBD
            search_targets: vec![
                SearchTarget {
                    pattern: "function_50000".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![],
                },
            ],
            validation_rules: vec![
                ValidationRule {
                    rule_type: ValidationRuleType::MustParse,
                    description: "Large file must parse without memory issues".to_string(),
                    expected_outcome: true,
                },
            ],
        });
        
        // Unicode test
        test_cases.push(TestCase {
            id: "unicode_content".to_string(),
            file_path: "test_data/edge_cases/chinese.rs".to_string(),
            description: "File with Chinese characters".to_string(),
            content_hash: "".to_string(),
            expected_chunks: vec![
                ExpectedChunk {
                    start_byte: 0,
                    end_byte: 200,
                    chunk_type: "function_definition".to_string(),
                    content_preview: "pub fn 函数名() -> String".to_string(),
                    should_be_searchable: true,
                    contains_patterns: vec!["函数名".to_string(), "你好世界".to_string()],
                },
            ],
            search_targets: vec![
                SearchTarget {
                    pattern: "函数名".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![],
                },
                SearchTarget {
                    pattern: "你好世界".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![],
                },
            ],
            validation_rules: vec![
                ValidationRule {
                    rule_type: ValidationRuleType::MustHandleUnicode,
                    description: "Must handle Chinese characters correctly".to_string(),
                    expected_outcome: true,
                },
            ],
        });
        
        // Malformed syntax test
        test_cases.push(TestCase {
            id: "malformed_syntax".to_string(),
            file_path: "test_data/edge_cases/unmatched_bracket.rs".to_string(),
            description: "File with unmatched brackets".to_string(),
            content_hash: "".to_string(),
            expected_chunks: vec![], // Parser should handle gracefully
            search_targets: vec![
                SearchTarget {
                    pattern: "fn test()".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![],
                },
            ],
            validation_rules: vec![
                ValidationRule {
                    rule_type: ValidationRuleType::MustNotCrash,
                    description: "Malformed syntax must not crash parser".to_string(),
                    expected_outcome: true,
                },
            ],
        });
        
        Ok(test_cases)
    }
    
    fn generate_boundary_test_cases() -> Result<Vec<TestCase>> {
        debug!("Generating boundary test cases");
        
        let mut test_cases = Vec::new();
        
        test_cases.push(TestCase {
            id: "function_boundary".to_string(),
            file_path: "test_data/chunk_boundaries/function_preservation.rs".to_string(),
            description: "Function that should not be split across chunks".to_string(),
            content_hash: "".to_string(),
            expected_chunks: vec![
                ExpectedChunk {
                    start_byte: 0,
                    end_byte: 1500,
                    chunk_type: "function_definition".to_string(),
                    content_preview: "pub fn important_function_with_complex_signature".to_string(),
                    should_be_searchable: true,
                    contains_patterns: vec![
                        "important_function_with_complex_signature".to_string(),
                        "Result<HashMap<String, T>".to_string(),
                    ],
                },
            ],
            search_targets: vec![
                SearchTarget {
                    pattern: "important_function_with_complex_signature".to_string(),
                    should_find: true,
                    expected_count: 1,
                    expected_locations: vec![],
                },
            ],
            validation_rules: vec![
                ValidationRule {
                    rule_type: ValidationRuleType::MustNotSplitSemanticUnit,
                    description: "Function signature and body must stay together".to_string(),
                    expected_outcome: true,
                },
            ],
        });
        
        Ok(test_cases)
    }
    
    fn generate_search_queries() -> Result<Vec<SearchQuery>> {
        debug!("Generating search queries");
        
        let queries = vec![
            // Exact match queries
            SearchQuery {
                id: "exact_package_section".to_string(),
                query: "[package]".to_string(),
                query_type: QueryType::ExactMatch,
                expected_results: vec![
                    ExpectedResult {
                        file_path: "test_data/special_chars/cargo_basic.toml".to_string(),
                        match_start: 1,
                        match_end: 10,
                        context: "[package]\nname = \"test-project\"".to_string(),
                        relevance_score: 1.0,
                    },
                ],
                expected_count: 1,
                max_latency_ms: 10,
            },
            
            // Boolean AND queries
            SearchQuery {
                id: "result_and_error".to_string(),
                query: "Result AND Error".to_string(),
                query_type: QueryType::BooleanAnd,
                expected_results: vec![],
                expected_count: 5, // Expected in multiple files
                max_latency_ms: 20,
            },
            
            // Complex generic pattern
            SearchQuery {
                id: "complex_generic".to_string(),
                query: "Result<T, E>".to_string(),
                query_type: QueryType::ExactMatch,
                expected_results: vec![],
                expected_count: 10, // Present in many test files
                max_latency_ms: 15,
            },
            
            // Unicode search
            SearchQuery {
                id: "unicode_function".to_string(),
                query: "函数名".to_string(),
                query_type: QueryType::ExactMatch,
                expected_results: vec![
                    ExpectedResult {
                        file_path: "test_data/edge_cases/chinese.rs".to_string(),
                        match_start: 20,
                        match_end: 23,
                        context: "pub fn 函数名() -> String".to_string(),
                        relevance_score: 1.0,
                    },
                ],
                expected_count: 1,
                max_latency_ms: 10,
            },
            
            // Macro pattern
            SearchQuery {
                id: "derive_macro".to_string(),
                query: "#[derive(Debug)]".to_string(),
                query_type: QueryType::ExactMatch,
                expected_results: vec![],
                expected_count: 15, // Present in many files
                max_latency_ms: 25,
            },
        ];
        
        Ok(queries)
    }
    
    fn generate_chunk_expectations() -> Result<Vec<ChunkExpectation>> {
        debug!("Generating chunk expectations");
        
        let expectations = vec![
            ChunkExpectation {
                file_path: "test_data/chunk_boundaries/function_preservation.rs".to_string(),
                expected_chunk_count: 3,
                chunk_boundaries: vec![
                    ChunkBoundary {
                        start_byte: 500,
                        end_byte: 1500,
                        boundary_type: BoundaryType::Function,
                        should_not_split: true,
                    },
                ],
                semantic_units: vec![
                    SemanticUnit {
                        unit_type: "function".to_string(),
                        start_byte: 500,
                        end_byte: 1500,
                        name: "important_function_with_complex_signature".to_string(),
                        must_preserve_integrity: true,
                    },
                ],
            },
            
            ChunkExpectation {
                file_path: "test_data/chunk_boundaries/struct_preservation.rs".to_string(),
                expected_chunk_count: 2,
                chunk_boundaries: vec![
                    ChunkBoundary {
                        start_byte: 200,
                        end_byte: 1200,
                        boundary_type: BoundaryType::Struct,
                        should_not_split: true,
                    },
                ],
                semantic_units: vec![
                    SemanticUnit {
                        unit_type: "struct".to_string(),
                        start_byte: 200,
                        end_byte: 1200,
                        name: "ComplexStructAtBoundary".to_string(),
                        must_preserve_integrity: true,
                    },
                ],
            },
        ];
        
        Ok(expectations)
    }
    
    fn create_performance_baselines() -> PerformanceBaselines {
        PerformanceBaselines {
            indexing_rate_docs_per_sec: 500.0,
            search_latency_ms: 10.0,
            memory_usage_mb: 1000.0,
            concurrent_searches: 100,
            max_file_size_mb: 10.0,
        }
    }
    
    fn save_ground_truth(ground_truth: &GroundTruth) -> Result<()> {
        let json = serde_json::to_string_pretty(ground_truth)?;
        let path = Path::new("test_data/ground_truth.json");
        std::fs::write(path, json)?;
        debug!("Saved ground truth to {}", path.display());
        Ok(())
    }
    
    /// Load ground truth from file for validation
    pub fn load_ground_truth() -> Result<GroundTruth> {
        let path = Path::new("test_data/ground_truth.json");
        let content = std::fs::read_to_string(path)?;
        let ground_truth: GroundTruth = serde_json::from_str(&content)?;
        Ok(ground_truth)
    }
    
    /// Validate search results against ground truth
    pub fn validate_search_results(
        query_id: &str,
        actual_results: &[SearchResult],
        ground_truth: &GroundTruth,
    ) -> ValidationReport {
        let expected_query = ground_truth.search_queries
            .iter()
            .find(|q| q.id == query_id);
        
        if let Some(expected) = expected_query {
            ValidationReport {
                query_id: query_id.to_string(),
                expected_count: expected.expected_count,
                actual_count: actual_results.len(),
                precision: calculate_precision(actual_results, &expected.expected_results),
                recall: calculate_recall(actual_results, &expected.expected_results),
                latency_within_limit: true, // Would be measured
                passed: actual_results.len() == expected.expected_count,
            }
        } else {
            ValidationReport::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub content: String,
    pub score: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub query_id: String,
    pub expected_count: usize,
    pub actual_count: usize,
    pub precision: f64,
    pub recall: f64,
    pub latency_within_limit: bool,
    pub passed: bool,
}

fn calculate_precision(actual: &[SearchResult], expected: &[ExpectedResult]) -> f64 {
    // Simplified precision calculation
    if actual.is_empty() {
        return 1.0;
    }
    
    let relevant_retrieved = actual.iter()
        .filter(|result| {
            expected.iter().any(|exp| {
                exp.file_path == result.file_path &&
                exp.match_start <= result.start_byte &&
                exp.match_end >= result.end_byte
            })
        })
        .count();
    
    relevant_retrieved as f64 / actual.len() as f64
}

fn calculate_recall(actual: &[SearchResult], expected: &[ExpectedResult]) -> f64 {
    // Simplified recall calculation
    if expected.is_empty() {
        return 1.0;
    }
    
    let relevant_retrieved = expected.iter()
        .filter(|exp| {
            actual.iter().any(|result| {
                exp.file_path == result.file_path &&
                exp.match_start <= result.start_byte &&
                exp.match_end >= result.end_byte
            })
        })
        .count();
    
    relevant_retrieved as f64 / expected.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_ground_truth() {
        let ground_truth = GroundTruthGenerator::generate_ground_truth().unwrap();
        
        assert!(!ground_truth.test_cases.is_empty());
        assert!(!ground_truth.search_queries.is_empty());
        assert!(!ground_truth.chunk_expectations.is_empty());
        
        // Verify file was created
        assert!(Path::new("test_data/ground_truth.json").exists());
    }
    
    #[test]
    fn test_load_ground_truth() {
        // First generate ground truth
        GroundTruthGenerator::generate_ground_truth().unwrap();
        
        // Then load it
        let loaded = GroundTruthGenerator::load_ground_truth().unwrap();
        assert_eq!(loaded.version, "1.0.0");
    }
    
    #[test]
    fn test_validation_functions() {
        let ground_truth = GroundTruthGenerator::generate_ground_truth().unwrap();
        let actual_results = vec![];
        
        let report = GroundTruthGenerator::validate_search_results(
            "exact_package_section",
            &actual_results,
            &ground_truth,
        );
        
        assert_eq!(report.query_id, "exact_package_section");
    }
}
```

## Implementation Steps
1. Add ground truth data structures (GroundTruth, TestCase, SearchQuery, etc.)
2. Add GroundTruthGenerator struct to test_data.rs
3. Implement generate_special_char_test_cases() with expected results
4. Implement generate_edge_case_test_cases() for boundary conditions
5. Implement generate_boundary_test_cases() for chunking validation
6. Implement generate_search_queries() with expected outcomes
7. Implement generate_chunk_expectations() for semantic preservation
8. Add validation functions for accuracy testing
9. Add save/load functionality for ground truth data

## Success Criteria
- [ ] GroundTruthGenerator struct implemented and compiling
- [ ] Complete data structures for validation created
- [ ] Test cases with expected results for special characters
- [ ] Test cases with expected results for edge cases
- [ ] Test cases with expected results for chunk boundaries
- [ ] Search queries with expected outcomes defined
- [ ] Chunk expectations for semantic preservation
- [ ] Performance baselines established
- [ ] Validation functions for accuracy testing
- [ ] Ground truth saved to JSON file for persistence

## Test Command
```bash
cargo test test_generate_ground_truth
cargo test test_load_ground_truth
cargo test test_validation_functions
ls test_data/ground_truth.json
```

## Generated Data Structure
After completion, `test_data/ground_truth.json` should contain:
- **Test Cases**: Detailed expectations for each test file
- **Search Queries**: Queries with expected results and performance limits
- **Chunk Expectations**: Expected chunking behavior for semantic preservation
- **Performance Baselines**: Target metrics for indexing and search speed
- **Validation Rules**: Rules for automated testing

## Validation Capabilities
The ground truth enables validation of:
- Search accuracy (precision/recall)
- Chunk boundary preservation
- Special character handling
- Unicode support
- Performance benchmarks
- Error handling (malformed files)
- Semantic unit integrity

## Time Estimate
10 minutes

## Next Task
Task 14: Create benchmark framework structure for performance testing.