# Task 003: Create QueryType Enum with All Variants

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 001-002. The QueryType enum categorizes different types of search queries that need validation.

## Project Structure
```
src/
  validation/
    ground_truth.rs  <- Extend this file
  lib.rs
```

## Task Description
Create the `QueryType` enum that categorizes all the different types of queries the system needs to validate. This enum will be used to determine search modes and validation criteria.

## Requirements
1. Add to existing `src/validation/ground_truth.rs`
2. Implement `QueryType` enum with all query categories
3. Add helper methods for query type classification
4. Ensure proper serialization and display support

## Expected Code Structure to Add
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum QueryType {
    SpecialCharacters,
    BooleanAnd,
    BooleanOr,
    BooleanNot,
    Proximity,
    Wildcard,
    Regex,
    Phrase,
    Vector,
    Hybrid,
}

impl QueryType {
    pub fn from_query(query: &str) -> Self {
        // Analyze query string and determine type
        // Implementation needed - basic heuristics:
        // - Contains AND/OR/NOT -> Boolean variants
        // - Contains NEAR/WITHIN -> Proximity
        // - Contains *, ? -> Wildcard
        // - Contains regex chars -> Regex
        // - Contains quotes -> Phrase
        // - Special chars like [, ], <, > -> SpecialCharacters
        // - Default to appropriate type
    }
    
    pub fn description(&self) -> &'static str {
        // Return human-readable description
        // Implementation needed
    }
    
    pub fn requires_vector_search(&self) -> bool {
        // Implementation needed
    }
    
    pub fn requires_text_search(&self) -> bool {
        // Implementation needed
    }
    
    pub fn is_boolean_logic(&self) -> bool {
        // Implementation needed
    }
}

impl std::fmt::Display for QueryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Implementation needed
    }
}
```

## Dependencies
Same as previous tasks

## Success Criteria
- QueryType enum compiles without errors
- All query types are properly categorized
- Helper methods provide accurate query classification
- Display and serialization work correctly
- Integration with GroundTruthCase works properly

## Time Limit
10 minutes maximum