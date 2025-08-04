# Task 064: Create ground_truth.json Template

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates a complete ground_truth.json template with all query types represented, realistic test cases, proper schema validation, and comprehensive documentation of the structure.

## Project Structure
```
ground_truth.json  <- Create this file (in project root)
tests/fixtures/ground_truth/  <- Directory for additional ground truth files
```

## Task Description
Create a comprehensive ground_truth.json template file that defines test cases for all supported query types. Include realistic examples, expected results, validation criteria, and proper schema structure to support the validation system.

## Requirements
1. Include all query types from the validation system (special chars, boolean, proximity, wildcards, regex, vector, hybrid)
2. Provide realistic test cases with expected results
3. Include proper JSON schema validation structure
4. Document content validation requirements (must_contain, must_not_contain)
5. Support both positive and negative test cases

## Expected File Content/Code Structure

### Complete ground_truth.json Template
```json
{
  "metadata": {
    "version": "1.0.0",
    "created": "2024-01-01T00:00:00Z",
    "description": "LLMKG Vector Indexing System - Ground Truth Validation Dataset",
    "total_test_cases": 175,
    "query_type_distribution": {
      "special_characters": 25,
      "boolean_and": 20,
      "boolean_or": 20,
      "boolean_not": 15,
      "proximity": 15,
      "wildcard": 15,
      "phrase": 10,
      "regex": 10,
      "vector": 20,
      "hybrid": 25
    },
    "accuracy_requirement": "100%",
    "platforms": ["windows", "linux", "macos"],
    "schema_version": "2024.1"
  },
  "test_cases": [
    {
      "id": "special_001",
      "category": "special_characters",
      "difficulty": "basic",
      "description": "Search for function with underscore",
      "query": "test_function",
      "query_type": "SpecialCharacters",
      "search_mode": "keyword",
      "expected_files": [
        "src/lib.rs",
        "tests/integration_test.rs"
      ],
      "must_contain": [
        "fn test_function(",
        "test_function()"
      ],
      "must_not_contain": [
        "test function",
        "testfunction"
      ],
      "expected_total_matches": 2,
      "case_sensitive": false,
      "whole_word_only": false,
      "notes": "Basic underscore handling in function names"
    },
    {
      "id": "special_002",
      "category": "special_characters",
      "difficulty": "intermediate",
      "description": "Search for struct with generic parameters",
      "query": "HashMap<String, Vec<u32>>",
      "query_type": "SpecialCharacters",
      "search_mode": "exact",
      "expected_files": [
        "src/collections.rs"
      ],
      "must_contain": [
        "HashMap<String, Vec<u32>>"
      ],
      "must_not_contain": [
        "HashMap<String,Vec<u32>>",
        "HashMap < String , Vec < u32 > >"
      ],
      "expected_total_matches": 1,
      "case_sensitive": true,
      "whole_word_only": false,
      "notes": "Exact match for complex type signatures with angle brackets"
    },
    {
      "id": "boolean_001",
      "category": "boolean_and",
      "difficulty": "basic",
      "description": "Simple AND query for Rust keywords",
      "query": "struct AND impl",
      "query_type": "BooleanAnd",
      "search_mode": "boolean",
      "expected_files": [
        "src/data_structures.rs",
        "src/algorithms.rs"
      ],
      "must_contain": [
        "struct",
        "impl"
      ],
      "must_not_contain": [],
      "expected_total_matches": 2,
      "case_sensitive": false,
      "whole_word_only": true,
      "notes": "Files must contain both struct and impl keywords"
    },
    {
      "id": "boolean_002",
      "category": "boolean_and",
      "difficulty": "advanced",
      "description": "Complex AND query with multiple terms",
      "query": "async AND await AND tokio AND runtime",
      "query_type": "BooleanAnd",
      "search_mode": "boolean",
      "expected_files": [
        "src/async_handler.rs"
      ],
      "must_contain": [
        "async",
        "await", 
        "tokio",
        "runtime"
      ],
      "must_not_contain": [
        "sync",
        "blocking"
      ],
      "expected_total_matches": 1,
      "case_sensitive": false,
      "whole_word_only": true,
      "notes": "All four async-related terms must be present"
    },
    {
      "id": "boolean_or_001",
      "category": "boolean_or",
      "difficulty": "basic",
      "description": "Simple OR query for error types",
      "query": "Error OR Result OR Option",
      "query_type": "BooleanOr",
      "search_mode": "boolean",
      "expected_files": [
        "src/error_handling.rs",
        "src/utils.rs",
        "src/main.rs"
      ],
      "must_contain_any": [
        "Error",
        "Result", 
        "Option"
      ],
      "must_not_contain": [],
      "expected_total_matches": 3,
      "case_sensitive": true,
      "whole_word_only": true,
      "notes": "Files containing any of the three Rust error/option types"
    },
    {
      "id": "boolean_not_001",
      "category": "boolean_not",
      "difficulty": "basic",
      "description": "NOT query excluding test files",
      "query": "function NOT test",
      "query_type": "BooleanNot",
      "search_mode": "boolean",
      "expected_files": [
        "src/lib.rs",
        "src/utils.rs"
      ],
      "must_contain": [
        "function"
      ],
      "must_not_contain": [
        "test",
        "#[test]",
        "test_"
      ],
      "expected_total_matches": 2,
      "case_sensitive": false,
      "whole_word_only": false,
      "notes": "Files with 'function' but without any test-related content"
    },
    {
      "id": "proximity_001",
      "category": "proximity",
      "difficulty": "basic",
      "description": "Words within 5 positions",
      "query": "error NEAR/5 handling",
      "query_type": "Proximity",
      "search_mode": "proximity",
      "expected_files": [
        "src/error_handling.rs"
      ],
      "must_contain": [
        "error",
        "handling"
      ],
      "must_not_contain": [],
      "expected_total_matches": 1,
      "case_sensitive": false,
      "whole_word_only": false,
      "proximity_distance": 5,
      "notes": "'error' and 'handling' must appear within 5 words of each other"
    },
    {
      "id": "wildcard_001",
      "category": "wildcard",
      "difficulty": "basic",
      "description": "Single character wildcard",
      "query": "fn m?in",
      "query_type": "Wildcard",
      "search_mode": "wildcard",
      "expected_files": [
        "src/main.rs"
      ],
      "must_contain": [
        "fn main"
      ],
      "must_not_contain": [
        "fn min",
        "fn miin"
      ],
      "expected_total_matches": 1,
      "case_sensitive": false,
      "whole_word_only": false,
      "notes": "? matches exactly one character, should find 'fn main'"
    },
    {
      "id": "wildcard_002",
      "category": "wildcard",
      "difficulty": "intermediate",
      "description": "Multiple character wildcard",
      "query": "test_*_validation",
      "query_type": "Wildcard",
      "search_mode": "wildcard",
      "expected_files": [
        "tests/test_correctness_validation.rs",
        "tests/test_performance_validation.rs"
      ],
      "must_contain": [
        "test_",
        "_validation"
      ],
      "must_not_contain": [],
      "expected_total_matches": 2,
      "case_sensitive": false,
      "whole_word_only": false,
      "notes": "* matches zero or more characters between test_ and _validation"
    },
    {
      "id": "phrase_001",
      "category": "phrase",
      "difficulty": "basic",
      "description": "Exact phrase search",
      "query": "\"use std::collections::HashMap\"",
      "query_type": "Phrase",
      "search_mode": "phrase",
      "expected_files": [
        "src/collections.rs"
      ],
      "must_contain": [
        "use std::collections::HashMap"
      ],
      "must_not_contain": [
        "use std :: collections :: HashMap",
        "use std::collections::HashMap;"
      ],
      "expected_total_matches": 1,
      "case_sensitive": true,
      "whole_word_only": false,
      "notes": "Exact phrase match including whitespace and punctuation"
    },
    {
      "id": "regex_001",
      "category": "regex",
      "difficulty": "basic",
      "description": "Simple regex pattern for function definitions",
      "query": "fn\\s+\\w+\\s*\\(",
      "query_type": "Regex",
      "search_mode": "regex",
      "expected_files": [
        "src/lib.rs",
        "src/main.rs",
        "src/utils.rs"
      ],
      "must_contain": [
        "fn "
      ],
      "must_not_contain": [],
      "expected_total_matches": 3,
      "case_sensitive": false,
      "whole_word_only": false,
      "regex_flags": ["multiline"],
      "notes": "Matches function definitions: 'fn' + whitespace + identifier + optional whitespace + opening paren"
    },
    {
      "id": "regex_002",
      "category": "regex",
      "difficulty": "advanced",
      "description": "Complex regex for error handling patterns",
      "query": "(Result<[^,>]+,\\s*\\w+Error>|Option<[^>]+>)",
      "query_type": "Regex",
      "search_mode": "regex",
      "expected_files": [
        "src/error_handling.rs"
      ],
      "must_contain_any": [
        "Result<",
        "Option<"
      ],
      "must_not_contain": [],
      "expected_total_matches": 1,
      "case_sensitive": true,
      "whole_word_only": false,
      "regex_flags": ["multiline"],
      "notes": "Matches Result<T, Error> or Option<T> type signatures"
    },
    {
      "id": "vector_001",
      "category": "vector",
      "difficulty": "basic",
      "description": "Vector similarity search for code semantics",
      "query": "implement hash table data structure",
      "query_type": "Vector",
      "search_mode": "vector",
      "expected_files": [
        "src/hash_table.rs",
        "src/collections.rs"
      ],
      "must_contain": [],
      "must_not_contain": [],
      "expected_total_matches": 2,
      "similarity_threshold": 0.7,
      "vector_model": "code-similarity-model",
      "notes": "Semantic similarity search for hash table implementations"
    },
    {
      "id": "vector_002",
      "category": "vector",
      "difficulty": "intermediate",
      "description": "Vector search for error handling patterns",
      "query": "graceful error recovery and logging",
      "query_type": "Vector",
      "search_mode": "vector",
      "expected_files": [
        "src/error_handling.rs",
        "src/logging.rs"
      ],
      "must_contain": [],
      "must_not_contain": [],
      "expected_total_matches": 2,
      "similarity_threshold": 0.75,
      "vector_model": "code-similarity-model",
      "notes": "Semantic search for error handling and logging code"
    },
    {
      "id": "hybrid_001",
      "category": "hybrid",
      "difficulty": "advanced",
      "description": "Hybrid search combining keyword and vector",
      "query": {
        "keyword": "async function",
        "vector": "asynchronous programming patterns",
        "weight_keyword": 0.4,
        "weight_vector": 0.6
      },
      "query_type": "Hybrid",
      "search_mode": "hybrid",
      "expected_files": [
        "src/async_handler.rs",
        "src/concurrent.rs"
      ],
      "must_contain": [
        "async"
      ],
      "must_not_contain": [],
      "expected_total_matches": 2,
      "similarity_threshold": 0.6,
      "keyword_threshold": 0.5,
      "notes": "Combines exact keyword matching with semantic similarity"
    },
    {
      "id": "hybrid_002",
      "category": "hybrid",
      "difficulty": "expert",
      "description": "Complex hybrid with boolean logic and vectors",
      "query": {
        "boolean": "(Result OR Error) AND handling",
        "vector": "robust error management strategies",
        "proximity": "error NEAR/3 handling",
        "weight_boolean": 0.3,
        "weight_vector": 0.4,
        "weight_proximity": 0.3
      },
      "query_type": "Hybrid",
      "search_mode": "hybrid",
      "expected_files": [
        "src/error_handling.rs"
      ],
      "must_contain": [
        "Result",
        "Error",
        "handling"
      ],
      "must_not_contain": [],
      "expected_total_matches": 1,
      "similarity_threshold": 0.7,
      "boolean_threshold": 0.8,
      "proximity_distance": 3,
      "notes": "Advanced hybrid combining boolean logic, proximity, and semantic search"
    }
  ],
  "validation_criteria": {
    "accuracy_requirements": {
      "minimum_accuracy": 100.0,
      "false_positive_tolerance": 0.0,
      "false_negative_tolerance": 0.0
    },
    "performance_requirements": {
      "max_query_time_ms": 1000,
      "max_total_validation_time_minutes": 30,
      "memory_limit_mb": 2048
    },
    "content_validation": {
      "check_must_contain": true,
      "check_must_not_contain": true,
      "case_sensitive_by_default": false,
      "whole_word_matching": false,
      "validate_file_existence": true
    }
  },
  "test_data_sources": {
    "synthetic_rust_files": {
      "count": 50,
      "size_range_kb": [1, 100],
      "complexity_levels": ["basic", "intermediate", "advanced", "expert"]
    },
    "real_rust_projects": {
      "tokio_samples": true,
      "serde_samples": true,
      "actix_samples": true,
      "custom_samples": true
    },
    "edge_cases": {
      "unicode_filenames": true,
      "special_characters": true,
      "large_files": true,
      "empty_files": true,
      "binary_files": false
    }
  },
  "schema_definition": {
    "required_fields": [
      "id",
      "category", 
      "query",
      "query_type",
      "search_mode",
      "expected_files",
      "expected_total_matches"
    ],
    "optional_fields": [
      "difficulty",
      "description",
      "must_contain",
      "must_not_contain", 
      "case_sensitive",
      "whole_word_only",
      "proximity_distance",
      "similarity_threshold",
      "regex_flags",
      "notes"
    ],
    "query_types": [
      "SpecialCharacters",
      "BooleanAnd",
      "BooleanOr", 
      "BooleanNot",
      "Proximity",
      "Wildcard",
      "Phrase",
      "Regex",
      "Vector",
      "Hybrid"
    ],
    "search_modes": [
      "keyword",
      "exact",
      "boolean",
      "proximity",  
      "wildcard",
      "phrase",
      "regex",
      "vector",
      "hybrid"
    ],
    "difficulty_levels": [
      "basic",
      "intermediate", 
      "advanced",
      "expert"
    ]
  }
}
```

### Schema Validation Utility (`validate_ground_truth_schema.rs`)
```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;
use anyhow::{Result, anyhow};
use std::collections::HashSet;

#[derive(Debug, Serialize, Deserialize)]
pub struct GroundTruthSchema {
    pub metadata: Metadata,
    pub test_cases: Vec<TestCase>,
    pub validation_criteria: ValidationCriteria,
    pub test_data_sources: TestDataSources,
    pub schema_definition: SchemaDefinition,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestCase {
    pub id: String,
    pub category: String,
    pub difficulty: Option<String>,
    pub description: Option<String>,
    pub query: Value, // Can be string or object for hybrid queries
    pub query_type: String,
    pub search_mode: String,
    pub expected_files: Vec<String>,
    pub must_contain: Option<Vec<String>>,
    pub must_not_contain: Option<Vec<String>>,
    pub expected_total_matches: u32,
    pub case_sensitive: Option<bool>,
    pub whole_word_only: Option<bool>,
    // Additional optional fields for specific query types
    pub proximity_distance: Option<u32>,
    pub similarity_threshold: Option<f64>,
    pub regex_flags: Option<Vec<String>>,
    pub notes: Option<String>,
}

/// Validates ground truth JSON against schema requirements
pub fn validate_ground_truth_schema(json_content: &str) -> Result<()> {
    let schema: GroundTruthSchema = serde_json::from_str(json_content)
        .map_err(|e| anyhow!("JSON parse error: {}", e))?;
    
    validate_metadata(&schema.metadata)?;
    validate_test_cases(&schema.test_cases)?;
    validate_no_duplicate_ids(&schema.test_cases)?;
    validate_query_type_distribution(&schema.metadata, &schema.test_cases)?;
    
    println!("Ground truth schema validation passed!");
    Ok(())
}

fn validate_metadata(metadata: &Metadata) -> Result<()> {
    if metadata.total_test_cases == 0 {
        return Err(anyhow!("Total test cases must be greater than 0"));
    }
    
    if metadata.accuracy_requirement != "100%" {
        return Err(anyhow!("Accuracy requirement must be 100%"));
    }
    
    Ok(())
}

fn validate_test_cases(test_cases: &[TestCase]) -> Result<()> {
    for (index, test_case) in test_cases.iter().enumerate() {
        if test_case.id.is_empty() {
            return Err(anyhow!("Test case {} has empty ID", index));
        }
        
        if test_case.expected_files.is_empty() {
            return Err(anyhow!("Test case {} has no expected files", test_case.id));
        }
        
        if test_case.expected_total_matches == 0 {
            return Err(anyhow!("Test case {} has zero expected matches", test_case.id));
        }
        
        // Validate query type specific requirements
        match test_case.query_type.as_str() {
            "Proximity" => {
                if test_case.proximity_distance.is_none() {
                    return Err(anyhow!("Proximity query {} missing proximity_distance", test_case.id));
                }
            },
            "Vector" | "Hybrid" => {
                if test_case.similarity_threshold.is_none() {
                    return Err(anyhow!("Vector/Hybrid query {} missing similarity_threshold", test_case.id));
                }
            },
            _ => {}
        }
    }
    
    Ok(())
}

fn validate_no_duplicate_ids(test_cases: &[TestCase]) -> Result<()> {
    let mut ids = HashSet::new();
    
    for test_case in test_cases {
        if !ids.insert(&test_case.id) {
            return Err(anyhow!("Duplicate test case ID: {}", test_case.id));
        }
    }
    
    Ok(())
}

fn validate_query_type_distribution(metadata: &Metadata, test_cases: &[TestCase]) -> Result<()> {
    let mut actual_counts = std::collections::HashMap::new();
    
    for test_case in test_cases {
        *actual_counts.entry(&test_case.category).or_insert(0) += 1;
    }
    
    // Verify the distribution matches metadata
    for (category, expected_count) in &metadata.query_type_distribution {
        let actual_count = actual_counts.get(category).unwrap_or(&0);
        if actual_count != expected_count {
            return Err(anyhow!(
                "Category {} count mismatch: expected {}, found {}", 
                category, expected_count, actual_count
            ));
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_ground_truth_schema() {
        let json_content = include_str!("../ground_truth.json");
        assert!(validate_ground_truth_schema(json_content).is_ok());
    }
    
    #[test]
    fn test_invalid_schema_detection() {
        let invalid_json = r#"{"metadata": {"total_test_cases": 0}}"#;
        assert!(validate_ground_truth_schema(invalid_json).is_err());
    }
}
```

## Success Criteria
- JSON is valid and parseable by serde_json
- All query types are represented with realistic test cases
- Schema validation passes all requirements
- Test case IDs are unique across the dataset
- Expected results are realistic and achievable
- Must_contain and must_not_contain fields are properly specified
- Query type distribution matches metadata counts
- File supports both simple and complex query scenarios

## Time Limit
10 minutes maximum